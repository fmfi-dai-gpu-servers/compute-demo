"""CIFAR-10 GPU Training Marathon — long-running distributed training demo.

A single Ray job that saturates the cluster's GPU workers for a configurable
wall-clock budget (default ~2 hours, scalable arbitrarily):

  Stage 0  GPU worker pool boots (triggers Ray autoscaler scale-from-zero).
           Each worker actor pulls CIFAR-10 from S3 once and caches it in RAM.
  Stage 1  Population-based hyperparameter optimization: an evolving pool of
           ResNet-family configs is trained on every GPU in parallel; each
           trial streams per-epoch metrics to MLflow.
  Stage 2  The best config is retrained with hand-rolled multi-GPU DDP (gloo)
           across all workers for the remaining budget; rank 0 uploads final
           weights to S3.
  Stage 3  Summary run in MLflow + RESULT_JSON on stdout.

Design constraints of this cluster baked into the code:

  * The head node has no torch — the driver NEVER imports torch and every
    actor return value is plain JSON primitives (str/int/float/bool/None).
    Never return torch objects: e.g. torch.__version__ is a TorchVersion
    str-subclass defined inside the torch module — unpickling it on the head
    raises ModuleNotFoundError and kills the job.
  * Worker pods: 1.5 CPU, 3.5 GiB RAM, 1 HAMi vGPU with ~2 GiB usable VRAM.
    Batch sizes / model widths are constrained so the worst config fits.
  * Workers scale to zero after 120 s idle — trial actors stay alive for the
    whole job so GPUs never idle mid-run.

Tuning knobs (CLI args, overridable via environment variables):

  --runtime-min   total job budget in minutes (default 120; env MARATHON_RUNTIME_MIN)
  --num-gpus      GPU trial workers (default 4; env MARATHON_NUM_GPUS)
  --trial-min     target minutes per HPO trial (default 3; env MARATHON_TRIAL_MIN)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

import ray

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

S3_PREFIX = "cifar10"
EXPERIMENT = "cifar-gpu-marathon"

# Rough VRAM guard: bs256/width64 peaked at ~1.05 GiB on the TITAN Xp probe;
# cap width inversely with batch size to stay under the ~2 GiB vGPU cap.
MAX_WIDTH_FOR_BATCH = {64: 96, 128: 80, 256: 48}

HPO_FRACTION = 0.60  # of runtime budget
DDP_FRACTION = 0.25  # of runtime budget (rest = scale-up + finalization margin)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--runtime-min", type=float,
        default=float(os.environ.get("MARATHON_RUNTIME_MIN", "120")),
        help="total wall-clock budget in minutes",
    )
    p.add_argument(
        "--num-gpus", type=int,
        default=int(os.environ.get("MARATHON_NUM_GPUS", "4")),
        help="number of GPU trial workers (capped at cluster capacity)",
    )
    p.add_argument(
        "--trial-min", type=float,
        default=float(os.environ.get("MARATHON_TRIAL_MIN", "3")),
        help="target minutes per HPO trial",
    )
    return p.parse_args()


# ----------------------------------------------------------------------------
# GPU trial worker (runs on ray-ml GPU worker pods; returns primitives ONLY)
# ----------------------------------------------------------------------------


@ray.remote(num_gpus=1, max_restarts=0)
class TrialWorker:
    """Owns one GPU for the whole job: runs HPO trials, then one DDP rank."""

    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    def __init__(self, worker_idx: int) -> None:
        import boto3
        import numpy as np

        self.worker_idx = worker_idx
        self.np = np

        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_URL"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            region_name="us-east-1",
        )
        bucket = os.getenv("S3_BUCKET")

        import pickle
        from io import BytesIO

        def load_xy(key: str):
            raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            batch = pickle.load(BytesIO(raw), encoding="latin1")
            x = np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            y = np.asarray(batch["labels"], dtype=np.int64)
            return x, y

        xs, ys = zip(*(load_xy(f"{S3_PREFIX}/data_batch_{i}") for i in range(1, 6)))
        self.train_x = np.concatenate(xs)  # (50000, 3, 32, 32) uint8
        self.train_y = np.concatenate(ys)
        self.test_x, self.test_y = load_xy(f"{S3_PREFIX}/test_batch")

        from ray.util import get_node_ip_address

        self.node_ip = get_node_ip_address()

    def ping(self) -> str:
        return self.node_ip

    # -- model -----------------------------------------------------------------

    def _build_model(self, width: int, blocks: int):
        import torch
        from torch import nn

        def stage(cin: int, cout: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            for i in range(blocks):
                layers += [
                    nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(),
                ]
            return nn.Sequential(*layers)

        class ResNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, width, 3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                )
                self.s1 = stage(width, width)
                self.s2 = stage(width, width * 2)
                self.s3 = stage(width * 2, width * 4)
                self.head = nn.Linear(width * 4, 10)

            def forward(self, x):
                x = self.stem(x)
                x = x + self.s1(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = self.s2(x)
                x = torch.nn.functional.max_pool2d(x, 2)
                x = self.s3(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
                return self.head(x)

        return ResNet()

    # -- data plumbing -----------------------------------------------------------

    def _augment(self, x):
        """Random pad-crop + horizontal flip on a uint8 NCHW batch (numpy)."""
        np = self.np
        rng = np.random.default_rng()
        n, c, h, w = x.shape
        padded = np.pad(x, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="edge")
        tops = rng.integers(0, 9, size=n)
        lefts = rng.integers(0, 9, size=n)
        rows = (tops[:, None] + np.arange(h)[None, :])[:, None, :, None]  # (n,1,h,1)
        cols = (lefts[:, None] + np.arange(w)[None, :])[:, None, None, :]  # (n,1,1,w)
        out = padded[
            np.arange(n)[:, None, None, None], np.arange(c)[None, :, None, None], rows, cols
        ]
        flip = rng.random(n) < 0.5
        out[flip] = out[flip, :, :, ::-1]
        return out

    def _to_tensor(self, x_uint8):
        import torch

        t = torch.from_numpy(x_uint8.copy()).float().div_(255.0)
        mean = torch.tensor(self.MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.STD).view(1, 3, 1, 1)
        return (t - mean) / std

    def _evaluate(self, model, dev) -> float:
        import torch

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for s in range(0, len(self.test_x), 512):
                x = self._to_tensor(self.test_x[s : s + 512]).to(dev)
                pred = model(x).argmax(1).cpu().numpy()
                correct += int((pred == self.test_y[s : s + 512]).sum())
                total += len(pred)
        return correct / total

    # -- HPO trial -----------------------------------------------------------------

    def run_trial(self, config: dict, trial_id: str, deadline_ts: float) -> dict:
        """Train one config until deadline_ts; returns primitives only."""
        import torch
        from torch import nn

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.np.random.seed(42 + hash(trial_id) % 10000)

        result: dict = {
            "trial_id": trial_id,
            "config": config,
            "status": "failed",
            "best_val_acc": 0.0,
            "epochs": 0,
            "train_seconds": 0.0,
        }
        mlflow_run = None
        try:
            dev = torch.device("cuda")
            model = self._build_model(int(config["width"]), int(config["blocks"])).to(dev)
            opt = torch.optim.SGD(
                model.parameters(), lr=float(config["lr"]),
                momentum=0.9, weight_decay=5e-4, nesterov=True,
            )
            bs = int(config["batch"])
            smooth = float(config.get("label_smoothing", 0.0))
            steps_per_epoch = math.ceil(len(self.train_x) / bs)

            def lr_at(epoch: int) -> float:
                return float(config["lr"]) * (0.5 ** (epoch // 8))

            if os.getenv("MLFLOW_TRACKING_URI"):
                try:
                    import mlflow

                    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
                    mlflow.set_experiment(EXPERIMENT)
                    mlflow_run = mlflow.start_run(run_name=trial_id)
                    mlflow.set_tag("stage", "hpo")
                    mlflow.set_tag("worker", str(self.worker_idx))
                    mlflow.log_params({k: str(v) for k, v in config.items()})
                except BaseException as e:  # noqa: BLE001 — tracking must never kill training
                    mlflow_run = None
                    print(f"  (mlflow unavailable, training without tracking: {e!r})", flush=True)

            epoch = 0
            t_start = time.time()
            while time.time() < deadline_ts:
                for g in opt.param_groups:
                    g["lr"] = lr_at(epoch)
                model.train()
                perm = self.np.random.permutation(len(self.train_x))
                loss_sum, seen = 0.0, 0
                for s in range(steps_per_epoch):
                    idx = perm[s * bs : (s + 1) * bs]
                    x = self._to_tensor(self._augment(self.train_x[idx])).to(dev, non_blocking=True)
                    y = torch.from_numpy(self.train_y[idx]).to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x), y, label_smoothing=smooth)
                    loss.backward()
                    opt.step()
                    loss_sum += float(loss.item()) * len(idx)
                    seen += len(idx)
                    if time.time() >= deadline_ts:
                        break
                epoch += 1

                val_acc = self._evaluate(model, dev)
                result["epochs"] = epoch
                result["best_val_acc"] = max(result["best_val_acc"], val_acc)
                result["status"] = "ok"
                if mlflow_run is not None:
                    mlflow.log_metric("train_loss", loss_sum / max(seen, 1), step=epoch)
                    mlflow.log_metric("val_acc", val_acc, step=epoch)
            result["train_seconds"] = round(time.time() - t_start, 1)
        except BaseException as e:  # noqa: BLE001 — never leak torch objects to the driver
            import traceback

            result["status"] = "oom" if "out of memory" in str(e).lower() else "error"
            result["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            result["traceback"] = traceback.format_exc()[-900:]
            try:
                torch.cuda.empty_cache()
            except BaseException:  # noqa: BLE001,S110 — best-effort cleanup
                pass
        finally:
            if mlflow_run is not None:
                try:
                    import mlflow

                    mlflow.end_run()
                except BaseException:  # noqa: BLE001,S110 — best-effort cleanup
                    pass
        return result

    # -- DDP final training -----------------------------------------------------------

    def ddp_train(
        self, rank: int, world_size: int, master_addr: str, master_port: int,
        config: dict, deadline_ts: float,
    ) -> dict:
        """One rank of the final multi-GPU DDP training of the best config."""
        from datetime import timedelta

        import torch
        import torch.distributed as dist
        from torch import nn

        out: dict = {"rank": rank, "status": "failed"}
        try:
            dev = torch.device("cuda")
            torch.manual_seed(42)
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                rank=rank, world_size=world_size,
                timeout=timedelta(seconds=600),
            )
            model = self._build_model(int(config["width"]), int(config["blocks"])).to(dev)
            ddp = nn.parallel.DistributedDataParallel(
                model, device_ids=[0] if dev.type == "cuda" else None
            )

            bs = int(config["batch"])
            base_lr = float(config["lr"]) * world_size  # linear LR scaling
            opt = torch.optim.SGD(ddp.parameters(), lr=base_lr, momentum=0.9,
                                  weight_decay=5e-4, nesterov=True)
            smooth = float(config.get("label_smoothing", 0.0))

            my_idx = self.np.arange(rank, len(self.train_x), world_size)
            steps_per_epoch = math.ceil(len(my_idx) / bs)
            epoch, best_acc = 0, 0.0
            t_start = time.time()
            while time.time() < deadline_ts:
                lr = base_lr * (0.5 ** (epoch // 10))
                for g in opt.param_groups:
                    g["lr"] = lr
                ddp.train()
                perm = self.np.random.permutation(len(my_idx))
                for s in range(steps_per_epoch):
                    idx = my_idx[perm[s * bs : (s + 1) * bs]]
                    x = self._to_tensor(self._augment(self.train_x[idx])).to(dev, non_blocking=True)
                    y = torch.from_numpy(self.train_y[idx]).to(dev)
                    opt.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(ddp(x), y, label_smoothing=smooth)
                    loss.backward()
                    opt.step()
                    if time.time() >= deadline_ts:
                        break
                epoch += 1

                # distributed eval: every rank scores the full test set, allreduce
                model.eval()
                local_correct = local_total = 0
                with torch.no_grad():
                    for s0 in range(0, len(self.test_x), 512):
                        x = self._to_tensor(self.test_x[s0 : s0 + 512]).to(dev)
                        pred = model(x).argmax(1).cpu().numpy()
                        local_correct += int((pred == self.test_y[s0 : s0 + 512]).sum())
                        local_total += len(pred)
                t_correct = torch.tensor([local_correct], dtype=torch.float64, device=dev)
                t_total = torch.tensor([local_total], dtype=torch.float64, device=dev)
                dist.all_reduce(t_correct)
                dist.all_reduce(t_total)
                acc = float(t_correct.item() / t_total.item())
                best_acc = max(best_acc, acc)
                print(
                    f"[ddp rank{rank}] epoch={epoch} lr={lr:.4f} val_acc={acc:.4f} "
                    f"elapsed={time.time() - t_start:.0f}s",
                    flush=True,
                )

            out.update(
                status="ok",
                epochs=epoch,
                best_val_acc=round(best_acc, 4),
                seconds=round(time.time() - t_start, 1),
            )

            if rank == 0:
                from io import BytesIO

                import boto3

                buf = BytesIO()
                torch.save({"state_dict": model.state_dict(), "config": config}, buf)
                s3 = boto3.client(
                    "s3",
                    endpoint_url=os.getenv("S3_URL"),
                    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
                    region_name="us-east-1",
                )
                s3.put_object(Bucket=os.getenv("S3_BUCKET"),
                              Key=f"{S3_PREFIX}/best_model.pt", Body=buf.getvalue())
                out["model_s3_key"] = f"{S3_PREFIX}/best_model.pt"
            dist.destroy_process_group()
        except BaseException as e:  # noqa: BLE001
            out["status"] = "error"
            out["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            try:
                dist.destroy_process_group()
            except BaseException:  # noqa: BLE001,S110 — best-effort cleanup
                pass
        return out


# ----------------------------------------------------------------------------
# HPO: (mu + lambda) evolution over the config space
# ----------------------------------------------------------------------------


def random_config(rng: random.Random) -> dict:
    batch = rng.choice([64, 128, 256])
    width = rng.choice([w for w in (16, 24, 32, 48, 64, 80, 96) if w <= MAX_WIDTH_FOR_BATCH[batch]])
    return {
        "width": width,
        "blocks": rng.choice([1, 2, 3]),
        "lr": round(10 ** rng.uniform(-3, -0.7), 5),
        "batch": batch,
        "label_smoothing": rng.choice([0.0, 0.1]),
    }


def mutate(cfg: dict, rng: random.Random) -> dict:
    out = dict(cfg)
    batch = rng.choice([out["batch"], rng.choice([64, 128, 256])])
    width = out["width"] if batch == out["batch"] else min(out["width"], MAX_WIDTH_FOR_BATCH[batch])
    out["batch"] = batch
    out["width"] = max(16, min(width, MAX_WIDTH_FOR_BATCH[batch]))
    out["blocks"] = max(1, min(3, out["blocks"] + rng.choice([-1, 0, 1])))
    out["lr"] = round(min(0.3, max(1e-3, out["lr"] * 10 ** rng.uniform(-0.4, 0.4))), 5)
    if rng.random() < 0.3:
        out["label_smoothing"] = rng.choice([0.0, 0.1])
    return out

def _boot_worker_pool(n_requested: int) -> tuple[list, list]:
    """Create TrialWorker actors and wait for the autoscaler to place them.

    The autoscaler only launches workers for pending GPU demand, so the full
    request must exist up front. If the cluster's worker cap is below the
    request, placement stalls; after a stability window the unplaced surplus
    actors are killed and the job proceeds with the workers that did place.
    Returns ``(workers, ready_pod_ips)``.
    """
    workers = [TrialWorker.remote(i) for i in range(n_requested)]
    ref_to_idx = {w.ping.remote(): i for i, w in enumerate(workers)}
    pending = list(ref_to_idx)
    placed: dict[int, str] = {}
    last_n, stable_since = -1, None
    t0 = time.time()
    while pending and time.time() - t0 < 1500:
        done, pending = ray.wait(pending, num_returns=len(pending), timeout=20)
        for ref in done:
            placed[ref_to_idx[ref]] = ray.get(ref)
        n = len(placed)
        if n and n == last_n:
            stable_since = stable_since or time.time()
            if time.time() - stable_since > 240:
                print(f"[stage 0] placement stalled at {n}/{n_requested} GPUs "
                      f"(cluster worker cap); continuing with {n}", flush=True)
                break
        else:
            last_n, stable_since = n, None
        print(f"[stage 0] {n}/{n_requested} workers up ({time.time() - t0:.0f}s)",
              flush=True)
    for ref in pending:  # kill unplaced surplus actors
        ray.kill(workers[ref_to_idx[ref]])
    if not placed:
        raise RuntimeError("no GPU workers could be placed on this cluster")
    order = sorted(placed)
    return [workers[i] for i in order], [placed[i] for i in order]


# ----------------------------------------------------------------------------
# Driver (head node — must not import torch)
# ----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    t_job = time.time()
    total_s = args.runtime_min * 60

    ray.init()
    print(f"Ray cluster resources: {dict(ray.cluster_resources())}", flush=True)

    # Workers scale from zero on demand and the autoscaler only launches what
    # the actor pool asks for, so a request beyond the cluster's worker cap
    # would deadlock. The pool below adapts to what the cluster can place.
    print(
        f"Budget: {args.runtime_min:.0f} min total | requesting up to "
        f"{args.num_gpus} GPU workers (cluster currently advertises "
        f"{int(ray.cluster_resources().get('GPU', 0))} GPU)",
        flush=True,
    )

    # -- Stage 0: boot the GPU worker pool --------------------------------------
    print(f"\n[stage 0] booting up to {args.num_gpus} GPU trial workers "
          f"(each pulls CIFAR-10 from S3)...", flush=True)
    t0 = time.time()
    workers, ready_ips = _boot_worker_pool(args.num_gpus)
    n_gpu = len(workers)
    print(f"[stage 0] {n_gpu} workers up in {time.time() - t0:.0f}s, "
          f"pods on {sorted(set(ready_ips))}", flush=True)

    # -- Stage 1: HPO marathon ----------------------------------------------------
    hpo_deadline = t_job + total_s * HPO_FRACTION
    ddp_deadline = t_job + total_s * (HPO_FRACTION + DDP_FRACTION)
    trial_len = args.trial_min * 60
    rng = random.Random(42)

    queue = [random_config(rng) for _ in range(2 * n_gpu)]
    results: list[dict] = []
    trial_counter = 0
    free = list(range(n_gpu))
    pending: dict[int, tuple] = {}  # worker idx -> (future, trial_id, cfg)

    print(
        f"\n[stage 1] HPO evolution until t+{(hpo_deadline - t_job) / 60:.0f} min "
        f"(~{args.trial_min:.0f} min trials, {n_gpu} in parallel)",
        flush=True,
    )
    print("Watch trials appear live in the MLflow UI: https://mlflow.c.dai.fmph.uniba.sk\n", flush=True)

    def submit_next(widx: int) -> None:
        nonlocal trial_counter
        if not queue:
            return
        now = time.time()
        room = hpo_deadline - now - 20  # leave margin before the DDP handoff
        if room < max(90, min(trial_len, 150)):
            return  # not enough room for setup + at least one meaningful epoch
        cfg = queue.pop(0)
        trial_counter += 1
        tid = f"trial-{trial_counter:03d}"
        fut = workers[widx].run_trial.remote(cfg, tid, min(now + max(trial_len, 90), now + room))
        pending[widx] = (fut, tid, cfg)
        print(f"  -> {tid} on worker {widx}: {cfg}", flush=True)

    while True:
        for widx in list(free):
            submit_next(widx)
        if not pending:
            break
        done, _ = ray.wait([f for f, _, _ in pending.values()], num_returns=1, timeout=30)
        if not done:
            continue
        done_fut = done[0]
        widx = next(w for w, (f, _, _) in pending.items() if f == done_fut)
        fut, tid, cfg = pending.pop(widx)
        try:
            res = ray.get(fut, timeout=120)
            results.append(res)
            note = {
                "ok": f"val_acc={res['best_val_acc']:.4f}",
                "oom": "OOM (config too big — skipped)",
                "error": f"ERROR {res.get('error', '')[:80]}\n{res.get('traceback', '')}",
            }.get(res["status"], res["status"])
            print(f"  <- {tid}: {note} ({res['epochs']} epochs in {res['train_seconds']}s)", flush=True)
        except BaseException as e:  # noqa: BLE001 — dead actor etc.
            results.append({"trial_id": tid, "config": cfg, "status": "actor-died",
                            "best_val_acc": 0.0, "epochs": 0, "train_seconds": 0.0,
                            "error": repr(e)[:120]})
            print(f"  <- {tid}: ACTOR DIED ({repr(e)[:80]}) — dropping worker {widx}", flush=True)
            workers[widx] = None
        free.append(widx)
        free = [w for w in free if workers[w] is not None]

        if not queue and time.time() < hpo_deadline - 60:
            ok = sorted((r for r in results if r["status"] == "ok"),
                        key=lambda r: r["best_val_acc"], reverse=True)
            parents = [r["config"] for r in ok[:4]]
            if parents:
                queue.extend(mutate(rng.choice(parents), rng) for _ in range(2 * n_gpu))
                print(f"  ** evolved {2 * n_gpu} children from top {len(parents)} "
                      f"(best so far {ok[0]['best_val_acc']:.4f})", flush=True)
            else:
                queue.extend(random_config(rng) for _ in range(n_gpu))

    t_hpo_end = time.time()
    alive = [w for w in workers if w is not None]
    ok_results = sorted((r for r in results if r["status"] == "ok"),
                        key=lambda r: r["best_val_acc"], reverse=True)
    if not ok_results or not alive:
        print(f"\nNo usable HPO results ({len(results)} trials) — aborting before DDP.", flush=True)
        ray.shutdown()
        raise SystemExit(1)
    best_cfg = ok_results[0]["config"]
    best_acc = ok_results[0]["best_val_acc"]
    print(
        f"\n[stage 1] done at t+{(t_hpo_end - t_job) / 60:.0f} min: {len(results)} trials, "
        f"{len(ok_results)} ok, best val_acc={best_acc:.4f} with {best_cfg}",
        flush=True,
    )

    # -- Stage 2: final DDP training of the best config ------------------------------
    world = len(alive)
    master_addr = ready_ips[0]
    master_port = 29500
    ddp_deadline = max(time.time() + 120, ddp_deadline)  # at least 2 min of training
    print(
        f"\n[stage 2] final DDP training on {world} GPUs until "
        f"t+{(ddp_deadline - t_job) / 60:.0f} min (gloo via tcp://{master_addr}:{master_port})",
        flush=True,
    )
    ddp_results = []
    for f in [w.ddp_train.remote(i, world, master_addr, master_port, best_cfg, ddp_deadline)
              for i, w in enumerate(alive)]:
        try:
            ddp_results.append(ray.get(f, timeout=3600))
        except BaseException as e:  # noqa: BLE001
            ddp_results.append({"rank": -1, "status": "error", "error": repr(e)[:120]})
    for r in ddp_results:
        print(
            f"  ddp rank {r['rank']}: {r['status']} ({r.get('epochs', 0)} epochs, "
            f"best val_acc={r.get('best_val_acc', 0):.4f}, {r.get('seconds', 0)}s)",
            flush=True,
        )
    final = max((r for r in ddp_results if r["status"] == "ok"),
                key=lambda r: r.get("best_val_acc", 0), default=None)

    # -- Stage 3: summary ------------------------------------------------------------
    summary = {
        "runtime_budget_min": args.runtime_min,
        "gpus_used": world,
        "hpo_trials": len(results),
        "hpo_ok": len(ok_results),
        "hpo_seconds": round(t_hpo_end - t_job, 1),
        "best_config": best_cfg,
        "best_hpo_val_acc": round(best_acc, 4),
        "ddp_epochs": final.get("epochs") if final else 0,
        "ddp_final_val_acc": final.get("best_val_acc") if final else None,
        "model_s3_key": final.get("model_s3_key") if final else None,
        "total_seconds": round(time.time() - t_job, 1),
    }
    print(f"\nRESULT_JSON:{json.dumps(summary)}", flush=True)

    try:
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="marathon-summary"):
            mlflow.set_tag("stage", "summary")
            mlflow.log_params({k: str(v) for k, v in best_cfg.items()})
            mlflow.log_metric("best_hpo_val_acc", best_acc)
            mlflow.log_metric("hpo_trials", len(results))
            if final:
                mlflow.log_metric("ddp_final_val_acc", final.get("best_val_acc", 0))
                mlflow.log_metric("ddp_epochs", final.get("epochs", 0))
            mlflow.log_text(json.dumps(summary, indent=2), "summary.json")
    except BaseException as e:  # noqa: BLE001
        print(f"  (mlflow summary logging skipped: {e!r})", flush=True)

    print(f"\n{'=' * 60}")
    print("Ray Dashboard:  https://ray.c.dai.fmph.uniba.sk")
    print("MLflow UI:      https://mlflow.c.dai.fmph.uniba.sk")
    print("Datasets:       https://datasets.c.dai.fmph.uniba.sk")
    print(f"{'=' * 60}", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()

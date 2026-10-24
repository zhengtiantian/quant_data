#!/usr/bin/env python3
"""Run a command as the only LM Studio user, with the model it needs loaded and sized for this Mac.

    lm_session.py run --model qwen3.5-9b-mlx --workers-env ENRICH_WORKERS -- python script.py
    lm_session.py calibrate --model qwen3.5-9b-mlx [--force]
    lm_session.py plan --model qwen3.5-9b-mlx          # what run would load, without loading

Why (2026-10-24): two scheduled tasks used LM Studio at once - company match on qwen3.5-4b and
llm_enrich on gemma, the day before job_agent on gemma against GDELT on qwen - and LM Studio
swapped the two models in and out on every request ("Channel Error", "Operation canceled";
69 of 76 job_agent judge calls failed). Separately every model was loaded with parallel=1, so
the 8 client workers of llm_enrich queued behind one prediction at a time (0.8 article/s).

What `run` does:
  1. Takes one file lock shared by every caller (Airflow host tasks and job_agent), so only one
     LM Studio task runs at a time and the rest wait their turn.
  2. Unloads the LLMs the command does not need (embedding models stay: search uses them), then
     loads the one it needs under its plain model key, with the number of parallel slots this
     machine can actually use (see below), and a TTL so it unloads itself once idle. Unloading
     also frees the MLX engine's deleted-but-open temp file (57 GB on 2026-10-22).
  3. Exports LM_PARALLEL / LM_WORKERS (and the task's own worker variable) and runs the command.

Slots and instances are decided per machine and model, never below one:
  - Throughput is measured once (`calibrate`, run automatically the first time a model is used
    on a machine) at 1 / 2 / 4 / 8 slots; the smallest count within GOOD_ENOUGH of the best wins,
    since more slots past that only hold memory. Two instances of the model are measured too:
    on one GPU they usually lose to one instance with more slots (one read of the weights per
    batched step), but it is measured rather than assumed. Results are kept per machine, so the
    other Mac or a new one measures its own.
  - Memory caps the slot count every time: what macOS has free plus what the unloads release,
    and LM Studio's GPU working set, minus headroom. When the budget is short it still loads one
    slot so the task runs.
Standard library only: job_agent's Python runs this too.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

LMS = os.getenv("LMS_BIN", str(Path.home() / ".lmstudio" / "bin" / "lms"))
API = os.getenv("LM_API_URL", "http://127.0.0.1:1234/v1").rstrip("/")
STATE_DIR = Path(os.getenv("LM_SESSION_DIR", str(Path.home() / "Library" / "Application Support" / "lm_session")))
LOCK_FILE = STATE_DIR / "lmstudio.lock"
WAITING_DIR = STATE_DIR / "waiting"       # one file per session waiting for the lock: its model
CALIBRATION_FILE = STATE_DIR / "calibration.json"

CONTEXT = int(os.getenv("LM_CONTEXT", "8192"))   # what every task loads today (lms ps, 2026-10-24)
# A session unloads its model when the command ends; the TTL only covers a session that was
# killed. It must outlast a quiet stretch inside a task (GDELT downloads for minutes before its
# first model call), or LM Studio would drop the model and reload it on demand with one slot.
TTL_SECONDS = int(os.getenv("LM_TTL", "7200"))
GIB = 1024 ** 3
HEADROOM = 4 * GIB              # left to macOS, Docker and the apps on top of what is free
GPU_RESERVE = 3 * GIB           # LM Studio's own reserve under the GPU working set (its log)
GPU_SHARE = 0.78                # default GPU working set: LM Studio logged 37.44 GiB of 48 GiB
SLOT_STEPS = (1, 2, 4, 8)
GOOD_ENOUGH = 0.9
WORKERS_PER_SLOT = 2            # keeps a request queued behind every busy slot
# Full KV cache per token, from LM Studio's context_fit log lines (2026-10-23). Unknown
# architectures get the conservative default.
KV_BYTES_PER_TOKEN = {"qwen3_5": 32768, "gemma4": 16384}
DEFAULT_KV_BYTES_PER_TOKEN = 65536


def log(msg: str) -> None:
    print(f"[lm_session {datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- LM Studio
def lms(*args: str, timeout: int = 900) -> str:
    out = subprocess.run([LMS, *args], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"lms {' '.join(args)} failed: {(out.stderr or out.stdout).strip()[-400:]}")
    return out.stdout


def loaded() -> List[Dict]:
    return json.loads(lms("ps", "--json") or "[]")


def catalog() -> Dict[str, Dict]:
    return {m["modelKey"]: m for m in json.loads(lms("ls", "--json") or "[]")}


def busy(m: Dict) -> bool:
    """Serving someone right now. Sessions are serialised by the lock, but quant_ai's questions,
    R.11 runs and the other Mac's job_agent call LM Studio without it."""
    return (m.get("status") or "idle") != "idle" or int(m.get("queued") or 0) > 0


def ensure_server() -> None:
    try:
        urllib.request.urlopen(f"{API}/models", timeout=5).read()
    except Exception:  # noqa: BLE001 - any failure means: start it
        log("LM Studio server not answering; starting it")
        subprocess.run([LMS, "daemon", "up"], capture_output=True, timeout=120)
        lms("server", "start", timeout=120)


def unload(identifier: str) -> None:
    lms("unload", identifier, timeout=120)


def load(key: str, identifier: str, parallel: int) -> None:
    lms("load", key, "--identifier", identifier, "--parallel", str(parallel),
        "-c", str(CONTEXT), "--ttl", str(TTL_SECONDS), "-y")


# ---------------------------------------------------------------- memory
def _sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def machine_key() -> str:
    return f"{_sysctl('hw.model')}|{_sysctl('machdep.cpu.brand_string')}|{int(_sysctl('hw.memsize')) // GIB}GB"


def free_bytes() -> int:
    """What macOS can hand out without swapping: free + inactive + speculative + purgeable."""
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", text).group(1))
    pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in re.findall(r"Pages ([^:]+):\s+(\d+\.?)", text)}
    return page * sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable"))


def gpu_working_set() -> int:
    limit_mb = int(_sysctl("iogpu.wired_limit_mb") or 0)
    return limit_mb * 1024 * 1024 if limit_mb > 0 else int(int(_sysctl("hw.memsize")) * GPU_SHARE)


def model_bytes(info: Dict) -> int:
    """LM Studio's own estimate for one instance (it ignores --parallel), else 1.3x the files."""
    try:
        text = lms("load", info["modelKey"], "-c", str(CONTEXT), "--estimate-only", timeout=60)
        m = re.search(r"Estimated Total Memory:\s*([\d.]+)\s*GiB", text)
        if m:
            return int(float(m.group(1)) * GIB)
    except Exception:  # noqa: BLE001
        pass
    return int(info.get("sizeBytes", 0) * 1.3)


def slot_bytes(info: Dict) -> int:
    return KV_BYTES_PER_TOKEN.get(info.get("architecture"), DEFAULT_KV_BYTES_PER_TOKEN) * CONTEXT


def max_slots(budget: int, base: int, per_slot: int, instances: int = 1) -> int:
    """Slots per instance that fit the budget; never below 1 (the task must run)."""
    room = budget - base * instances
    fit = room // (per_slot * instances) if room > 0 else 0
    return int(max(1, min(fit, SLOT_STEPS[-1])))


def budget_for(keep: List[Dict], released: int) -> int:
    """Bytes available for the new load: the tighter of macOS memory and the GPU working set."""
    kept = sum(int(m.get("sizeBytes", 0) * 1.3) for m in keep)
    by_ram = free_bytes() + released - HEADROOM
    by_gpu = gpu_working_set() - kept - GPU_RESERVE
    return min(by_ram, by_gpu)


# ---------------------------------------------------------------- throughput
_ARTICLE = " ".join(
    f"Paragraph {i}: the company reported quarterly revenue of ${10 + i}.{i} billion, guided the next "
    f"quarter below consensus on weaker memory pricing, and said capital spending on AI accelerators "
    f"would rise; shares moved {i % 7 - 3}% after hours as analysts revised targets." for i in range(18))
_PROMPT = ("Read the news article and answer in JSON with keys company, sentiment (-1..1), "
           "event_type and a one-sentence summary.\n\n" + _ARTICLE)


def _one_request(identifier: str, max_tokens: int) -> int:
    body = json.dumps({"model": identifier, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": _PROMPT}]}).encode()
    req = urllib.request.Request(f"{API}/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return int(json.load(resp).get("usage", {}).get("completion_tokens", 0))


def bench(identifiers: List[str], slots_total: int, max_tokens: int = 160) -> Dict:
    """Generated tokens per second with every slot busy (two requests per slot, warm model)."""
    for ident in identifiers:
        _one_request(ident, 8)                              # warm-up, not timed
    n = max(8, slots_total * 2)
    errors, tokens = 0, 0
    lock = threading.Lock()

    def work(i: int) -> None:
        nonlocal errors, tokens
        try:
            got = _one_request(identifiers[i % len(identifiers)], max_tokens)
            with lock:
                tokens += got
        except Exception:  # noqa: BLE001
            with lock:
                errors += 1
    start = time.time()
    with ThreadPoolExecutor(max_workers=slots_total) as ex:
        list(ex.map(work, range(n)))
    wall = time.time() - start
    return {"tok_s": round(tokens / wall, 1) if wall > 0 else 0.0, "errors": errors, "requests": n}


def pick(results: Dict[int, Dict]) -> int:
    """Smallest error-free slot count within GOOD_ENOUGH of the best throughput."""
    clean = {s: r["tok_s"] for s, r in results.items() if r["errors"] == 0 and r["tok_s"] > 0}
    if not clean:
        return 1
    best = max(clean.values())
    return min(s for s, v in clean.items() if v >= GOOD_ENOUGH * best)


def load_calibration() -> Dict:
    try:
        return json.loads(CALIBRATION_FILE.read_text())
    except (OSError, ValueError):
        return {}


def calibrate(key: str, cap: Optional[int] = None) -> Dict:
    """Measure slot counts (and one two-instance layout) for this model on this machine."""
    info = catalog()[key]
    for m in loaded():
        if m["type"] == "llm" and busy(m):
            log(f"calibrate {key}: {m['identifier']} is serving someone else; left loaded (numbers may read low)")
        elif m["type"] == "llm":
            unload(m["identifier"])
    base, per_slot = model_bytes(info), slot_bytes(info)
    cap = cap or max_slots(budget_for([], 0), base, per_slot)
    results: Dict[int, Dict] = {}
    for slots in [s for s in SLOT_STEPS if s <= cap]:
        load(key, key, slots)
        try:
            results[slots] = bench([key], slots)
        finally:
            unload(key)
        log(f"calibrate {key}: {slots} slot(s) -> {results[slots]}")
        if len(results) >= 2:
            prev = results[SLOT_STEPS[SLOT_STEPS.index(slots) - 1]]["tok_s"]
            if results[slots]["tok_s"] < prev * 1.05:      # flat or worse: more slots will not help
                break
    best = pick(results)
    record = {"slots": best, "instances": 1, "context": CONTEXT, "measured_at": datetime.now().isoformat(timespec="seconds"),
              "results": {str(s): r for s, r in results.items()}}
    # Two instances sharing the same slot total, if memory allows it.
    per_instance = max(1, best // 2)
    if max_slots(budget_for([], 0), base, per_slot, instances=2) >= per_instance:
        ids = [key, f"{key}:2"]
        try:
            for ident in ids:
                load(key, ident, per_instance)
            two = bench(ids, per_instance * 2)
            record["two_instances"] = {"slots_each": per_instance, **two}
            log(f"calibrate {key}: 2 instances x {per_instance} slot(s) -> {two}")
            if two["errors"] == 0 and two["tok_s"] > results[best]["tok_s"] / GOOD_ENOUGH:
                record.update(instances=2, slots=per_instance)
        except Exception as e:  # noqa: BLE001 - the one-instance answer stands
            record["two_instances"] = {"error": str(e)[:200]}
        finally:
            for ident in ids:
                try:
                    unload(ident)
                except Exception:  # noqa: BLE001
                    pass
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_calibration()
    data.setdefault(machine_key(), {})[key] = record
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    log(f"calibrate {key}: chose {record['instances']} instance(s) x {record['slots']} slot(s)")
    return record


# ---------------------------------------------------------------- session
def plan(key: str, calibrate_if_missing: bool = True) -> Dict:
    """Instances and slots for this model now: the calibrated choice, capped by free memory."""
    info = catalog().get(key)
    if not info:
        raise SystemExit(f"lm_session: model {key!r} is not downloaded in LM Studio")
    record = load_calibration().get(machine_key(), {}).get(key)
    if record is None and calibrate_if_missing:
        record = calibrate(key)
    record = record or {"slots": 1, "instances": 1}
    current = loaded()
    others = [m for m in current if m["type"] == "llm"]
    keep = [m for m in current if m["type"] != "llm"]
    released = sum(int(m.get("sizeBytes", 0) * 1.3) for m in others)
    budget = budget_for(keep, released)
    instances, slots = record["instances"], record["slots"]
    base, per_slot = model_bytes(info), slot_bytes(info)
    if instances > 1 and max_slots(budget, base, per_slot, instances) < slots:
        instances, slots = 1, slots * instances           # same slot total in one instance
    slots = min(slots, max_slots(budget, base, per_slot, instances))
    return {"model": key, "instances": instances, "slots": max(1, slots),
            "ids": [key] + [f"{key}:{i}" for i in range(2, instances + 1)],
            "budget_gib": round(budget / GIB, 1), "calibrated": record.get("measured_at")}


def prepare(p: Dict) -> Dict:
    """Load what the plan needs, unload the LLMs it does not. A model already loaded the right
    way is kept; one in use by someone else is never pulled away from them: another model then
    stays loaded, and this model is used as loaded (its slot count replaces the plan's)."""
    for ident, m in {m["identifier"]: m for m in loaded()}.items():
        if m["type"] != "llm" or (ident in p["ids"] and m.get("parallel") == p["slots"]):
            continue
        if busy(m):
            if ident in p["ids"]:
                p = {**p, "slots": int(m.get("parallel") or 1)}
            log(f"{ident} is serving someone else; left as it is")
            continue
        unload(ident)
    current = {m["identifier"] for m in loaded()}
    for ident in p["ids"]:
        if ident not in current:
            load(p["model"], ident, p["slots"])
    return p


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def wanted_next(model: str) -> bool:
    """A session waiting for the lock needs this model too: keep it loaded for that one."""
    for f in WAITING_DIR.glob("*") if WAITING_DIR.exists() else []:
        try:
            if f.read_text().strip() == model and _pid_alive(int(f.name)):
                return True
        except (OSError, ValueError):
            continue
    return False


def release(p: Dict) -> None:
    """Unload the session's model unless a waiting session needs it or someone else is using it;
    either way the TTL unloads it once it really goes idle."""
    current = {m["identifier"]: m for m in loaded()}
    for ident in p["ids"]:
        m = current.get(ident)
        if m is None:
            continue
        if wanted_next(p["model"]) or busy(m):
            log(f"{ident} kept loaded (still in use, or the next task needs it); TTL {TTL_SECONDS}s")
            continue
        unload(ident)


def worker_env(specs: List[str], workers: int) -> Dict[str, str]:
    """NAME sets NAME=workers; NAME/6 sets NAME=ceil(workers/6) for scripts whose own worker
    count multiplies it (GDELT: BATCH_WORKERS=6 batches, SLM_MAX_CONCURRENCY calls each)."""
    env = {}
    for spec in specs:
        name, _, div = spec.partition("/")
        env[name] = str(max(1, math.ceil(workers / int(div or 1))))
    return env


@contextmanager
def exclusive(what: str, model: str = ""):
    """The one LM Studio lock. flock is released by the kernel when the process dies, so a
    killed task never leaves it held. While waiting, the model is noted in WAITING_DIR so the
    session ahead keeps it loaded for us."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fh.seek(0)
            log(f"LM Studio busy ({fh.read().strip() or 'another task'}); waiting")
            WAITING_DIR.mkdir(parents=True, exist_ok=True)
            note = WAITING_DIR / str(os.getpid())
            note.write_text(model)
            try:
                fcntl.flock(fh, fcntl.LOCK_EX)
            finally:
                note.unlink(missing_ok=True)
        fh.seek(0); fh.truncate()
        fh.write(f"pid {os.getpid()} since {datetime.now():%H:%M}: {what[:160]}")
        fh.flush()
        yield


def run(args) -> int:
    with exclusive(f"{args.model} {' '.join(args.command)}", args.model):
        ensure_server()
        p = prepare(plan(args.model))
        workers = p["slots"] * len(p["ids"]) * WORKERS_PER_SLOT
        env = {**os.environ, "LM_PARALLEL": str(p["slots"]), "LM_WORKERS": str(workers),
               "LM_MODEL_IDS": ",".join(p["ids"]), **worker_env(args.workers_env, workers)}
        if args.ids_env:
            env[args.ids_env] = ",".join(p["ids"])
        if args.second_id_env:
            env[args.second_id_env] = p["ids"][1] if len(p["ids"]) > 1 else ""
        log(f"{p['model']}: {len(p['ids'])} instance(s) x {p['slots']} slot(s), {workers} workers, "
            f"budget {p['budget_gib']} GiB, calibrated {p['calibrated']}")
        try:
            return subprocess.call(args.command, env=env)
        finally:
            try:
                release(p)
            except Exception as e:  # noqa: BLE001 - the TTL unloads it later
                log(f"could not unload {p['model']}: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--workers-env", action="append", default=[], help="NAME or NAME/divisor")
    r.add_argument("--ids-env", help="variable to receive the comma-separated instance ids")
    r.add_argument("--second-id-env", help="variable to receive the second instance id, or ''")
    r.add_argument("command", nargs=argparse.REMAINDER)
    c = sub.add_parser("calibrate")
    c.add_argument("--model", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if args.command[:1] == ["--"]:
            args.command = args.command[1:]
        if not args.command:
            ap.error("run needs a command after --")
        return run(args)
    ensure_server()
    if args.cmd == "calibrate":
        with exclusive(f"calibrate {args.model}"):
            print(json.dumps(calibrate(args.model), indent=2))
    else:
        print(json.dumps(plan(args.model, calibrate_if_missing=False), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

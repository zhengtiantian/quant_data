"""lm_session: slot choice, memory caps, worker variables, which models get unloaded. No LM Studio."""
from unittest import mock

from tools import lm_session as lm

GIB = lm.GIB
QWEN9 = {"modelKey": "qwen3.5-9b-mlx", "type": "llm", "architecture": "qwen3_5", "sizeBytes": 6 * GIB}
GEMMA = {"modelKey": "gemma-4-e4b-it-mlx", "identifier": "gemma-4-e4b-it-mlx", "type": "llm",
         "architecture": "gemma4", "sizeBytes": 7 * GIB, "parallel": 1}
EMBED = {"modelKey": "nomic", "identifier": "nomic", "type": "embedding", "sizeBytes": GIB // 10}


def test_pick_takes_the_smallest_slot_count_near_the_best():
    results = {1: {"tok_s": 30, "errors": 0}, 2: {"tok_s": 55, "errors": 0},
               4: {"tok_s": 80, "errors": 0}, 8: {"tok_s": 84, "errors": 0}}
    assert lm.pick(results) == 4                       # 80 >= 0.9 * 84
    results[4]["errors"] = 2                           # errors disqualify a count
    assert lm.pick(results) == 8
    assert lm.pick({}) == 1


def test_max_slots_never_below_one_and_capped():
    assert lm.max_slots(budget=2 * GIB, base=8 * GIB, per_slot=GIB // 4) == 1      # short: still one
    assert lm.max_slots(budget=9 * GIB, base=8 * GIB, per_slot=GIB // 4) == 4
    assert lm.max_slots(budget=40 * GIB, base=8 * GIB, per_slot=GIB // 4) == lm.SLOT_STEPS[-1]
    assert lm.max_slots(budget=17 * GIB, base=8 * GIB, per_slot=GIB // 4, instances=2) == 2


def test_worker_env_divides_for_scripts_that_multiply():
    assert lm.worker_env(["ENRICH_WORKERS"], 8) == {"ENRICH_WORKERS": "8"}
    assert lm.worker_env(["SLM_MAX_CONCURRENCY/6"], 8) == {"SLM_MAX_CONCURRENCY": "2"}
    assert lm.worker_env(["SLM_MAX_CONCURRENCY/6"], 2) == {"SLM_MAX_CONCURRENCY": "1"}


def _plan(record, budget, loaded=()):
    with mock.patch.object(lm, "catalog", return_value={QWEN9["modelKey"]: QWEN9}), \
         mock.patch.object(lm, "load_calibration", return_value={"m": {QWEN9["modelKey"]: record}}), \
         mock.patch.object(lm, "machine_key", return_value="m"), \
         mock.patch.object(lm, "loaded", return_value=list(loaded)), \
         mock.patch.object(lm, "budget_for", return_value=budget), \
         mock.patch.object(lm, "model_bytes", return_value=8 * GIB):
        return lm.plan(QWEN9["modelKey"], calibrate_if_missing=False)


def test_plan_caps_the_calibrated_slots_by_memory():
    p = _plan({"slots": 4, "instances": 1}, budget=30 * GIB)
    assert (p["instances"], p["slots"], p["ids"]) == (1, 4, ["qwen3.5-9b-mlx"])
    assert _plan({"slots": 4, "instances": 1}, budget=8.5 * GIB)["slots"] == 2
    assert _plan({"slots": 4, "instances": 1}, budget=1 * GIB)["slots"] == 1     # the task still runs


def test_two_instances_fall_back_to_one_when_memory_is_short():
    p = _plan({"slots": 2, "instances": 2}, budget=30 * GIB)
    assert p["ids"] == ["qwen3.5-9b-mlx", "qwen3.5-9b-mlx:2"]
    p = _plan({"slots": 2, "instances": 2}, budget=10 * GIB)
    assert (p["instances"], p["slots"]) == (1, 4)


def test_uncalibrated_model_runs_with_one_slot():
    with mock.patch.object(lm, "load_calibration", return_value={}), \
         mock.patch.object(lm, "catalog", return_value={QWEN9["modelKey"]: QWEN9}), \
         mock.patch.object(lm, "machine_key", return_value="m"), \
         mock.patch.object(lm, "loaded", return_value=[]), \
         mock.patch.object(lm, "budget_for", return_value=30 * GIB), \
         mock.patch.object(lm, "model_bytes", return_value=8 * GIB):
        assert lm.plan(QWEN9["modelKey"], calibrate_if_missing=False)["slots"] == 1


def test_prepare_unloads_other_llms_keeps_embeddings_and_reloads_wrong_parallel():
    stale = {"identifier": "qwen3.5-9b-mlx", "type": "llm", "parallel": 1}
    calls = []
    states = [[GEMMA, EMBED, stale], [EMBED]]
    with mock.patch.object(lm, "loaded", side_effect=lambda: states.pop(0)), \
         mock.patch.object(lm, "unload", side_effect=lambda i: calls.append(("unload", i))), \
         mock.patch.object(lm, "load", side_effect=lambda k, i, p: calls.append(("load", i, p))):
        lm.prepare({"model": "qwen3.5-9b-mlx", "ids": ["qwen3.5-9b-mlx"], "slots": 4})
    assert ("unload", "gemma-4-e4b-it-mlx") in calls
    assert ("unload", "qwen3.5-9b-mlx") in calls       # loaded with 1 slot, wanted 4
    assert ("unload", "nomic") not in calls
    assert calls[-1] == ("load", "qwen3.5-9b-mlx", 4)


def test_prepare_keeps_a_model_already_right():
    right = {"identifier": "qwen3.5-9b-mlx", "type": "llm", "parallel": 4}
    with mock.patch.object(lm, "loaded", return_value=[right]), \
         mock.patch.object(lm, "unload") as un, mock.patch.object(lm, "load") as ld:
        lm.prepare({"model": "qwen3.5-9b-mlx", "ids": ["qwen3.5-9b-mlx"], "slots": 4})
    un.assert_not_called(); ld.assert_not_called()


def test_run_holds_the_lock_exports_workers_and_unloads_after(tmp_path):
    seen = {}

    def fake_call(cmd, env):
        seen.update(env=env, locked=(tmp_path / "lmstudio.lock").read_text())
        return 3
    p = {"model": "qwen3.5-9b-mlx", "instances": 1, "slots": 4, "ids": ["qwen3.5-9b-mlx"],
         "budget_gib": 20, "calibrated": "x"}
    with mock.patch.object(lm, "STATE_DIR", tmp_path), mock.patch.object(lm, "LOCK_FILE", tmp_path / "lmstudio.lock"), \
         mock.patch.object(lm, "ensure_server"), mock.patch.object(lm, "plan", return_value=p), \
         mock.patch.object(lm, "prepare", side_effect=lambda plan: plan), mock.patch.object(lm, "unload") as un, \
         mock.patch.object(lm, "loaded", return_value=[{"identifier": "qwen3.5-9b-mlx", "type": "llm", "status": "idle"}]), \
         mock.patch.object(lm.subprocess, "call", side_effect=fake_call):
        rc = lm.main(["run", "--model", "qwen3.5-9b-mlx", "--workers-env", "ENRICH_WORKERS", "--", "python", "x.py"])
    assert rc == 3                                     # the command's exit code
    assert seen["env"]["ENRICH_WORKERS"] == "8" and seen["env"]["LM_PARALLEL"] == "4"
    assert "qwen3.5-9b-mlx" in seen["locked"]
    un.assert_called_once_with("qwen3.5-9b-mlx")


def test_job_agent_style_session_unloads_gemma_only_when_nobody_needs_it(tmp_path):
    p = {"model": "gemma-4-e4b-it-mlx", "ids": ["gemma-4-e4b-it-mlx"], "slots": 2}
    idle = {**GEMMA, "status": "idle", "queued": 0}
    serving = {**GEMMA, "status": "generating", "queued": 3}
    with mock.patch.object(lm, "WAITING_DIR", tmp_path), mock.patch.object(lm, "unload") as un:
        with mock.patch.object(lm, "loaded", return_value=[idle]):
            lm.release(p)
        un.assert_called_once_with("gemma-4-e4b-it-mlx")
        un.reset_mock()
        with mock.patch.object(lm, "loaded", return_value=[serving]):
            lm.release(p)                                  # someone else is using gemma
        un.assert_not_called()
        (tmp_path / str(lm.os.getpid())).write_text("gemma-4-e4b-it-mlx")
        with mock.patch.object(lm, "loaded", return_value=[idle]):
            lm.release(p)                                  # the next queued task wants gemma too
        un.assert_not_called()


def test_prepare_never_pulls_a_busy_model_away():
    other_busy = {"identifier": "qwen3.5-4b-mlx", "type": "llm", "status": "generating", "parallel": 1}
    own_busy = {**GEMMA, "status": "processingPrompt", "parallel": 1}
    with mock.patch.object(lm, "loaded", return_value=[other_busy, own_busy]), \
         mock.patch.object(lm, "unload") as un, mock.patch.object(lm, "load") as ld:
        out = lm.prepare({"model": "gemma-4-e4b-it-mlx", "ids": ["gemma-4-e4b-it-mlx"], "slots": 4})
    un.assert_not_called(); ld.assert_not_called()
    assert out["slots"] == 1                               # used as loaded, workers follow it

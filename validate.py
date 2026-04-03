"""
AUTORE Pre-Submission Validation
==================================

Mirrors the official bash validator (validate-submission.sh) checks:

  Step 1  HF Space ping        POST /reset → HTTP 200
  Step 2  Docker build         docker build succeeds
  Step 3  openenv validate     openenv validate CLI passes

Plus the internal Python checks the automated grader will run:

  [A] openenv.yaml compliance
  [B] Typed models (AutoreAction, AutoreObservation)
  [C] step() / reset() / state() interface
  [D] Reward logic
  [E] Yellow phase transition
  [F] Rush-hour traffic spike
  [G] Emergency vehicle handling & penalty
  [H] Episode termination at MAX_STEPS
  [I] Determinism (same seed → same trajectory)
  [J] 3+ tasks with scores in [0.0, 1.0]
  [K] inference.py exists in project root
  [L] inference.py imports cleanly and main() returns valid scores

Usage
-----
  # Internal checks only (no network/Docker required):
  python validate.py

  # Full validation including live HF Space ping:
  python validate.py --ping-url https://your-space.hf.space

  # Full validation with custom repo dir:
  python validate.py --ping-url https://your-space.hf.space --repo-dir /path/to/repo
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colour output ──────────────────────────────────────────────────────────
_USE_COLOUR = sys.stdout.isatty()
GREEN  = "\033[92m" if _USE_COLOUR else ""
RED    = "\033[91m" if _USE_COLOUR else ""
YELLOW = "\033[93m" if _USE_COLOUR else ""
BOLD   = "\033[1m"  if _USE_COLOUR else ""
NC     = "\033[0m"  if _USE_COLOUR else ""

results: list = []


def _check(name: str, cond: bool, detail: str = "") -> bool:
    ok = bool(cond)
    tag = f"{GREEN}[PASS]{NC}" if ok else f"{RED}[FAIL]{NC}"
    line = f"  {tag} {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append((name, ok))
    return ok


def _section(title: str):
    print(f"\n{BOLD}{title}{NC}")


def _stop(msg: str):
    print(f"\n{RED}{BOLD}Stopped: {msg}{NC}")
    _summary()
    sys.exit(1)


def _summary():
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'='*62}")
    if passed == total:
        print(f"{GREEN}{BOLD}  {passed}/{total} checks passed — ready to submit!{NC}")
    else:
        print(f"{RED}{BOLD}  {passed}/{total} checks passed.{NC}")
        failed = [n for n, ok in results if not ok]
        for f in failed:
            print(f"    {RED}✗{NC} {f}")
    print(f"{'='*62}")


# ══════════════════════════════════════════════════════════════════════════
# External checks (mirrors validate-submission.sh)
# ══════════════════════════════════════════════════════════════════════════

def check_hf_space(ping_url: str):
    """Step 1 — POST <ping_url>/reset and expect HTTP 200."""
    _section("Step 1/3: HF Space ping")
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(
            f"{ping_url.rstrip('/')}/reset",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            code = resp.getcode()
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as exc:
        _check("HF Space /reset reachable", False, str(exc))
        _stop("Fix Step 1 before continuing.")
        return

    ok = _check(f"POST /reset → HTTP {code}", code == 200, f"expected 200, got {code}")
    if not ok:
        _stop("Fix Step 1 before continuing.")


def check_docker_build(repo_dir: str):
    """Step 2 — docker build must succeed within 600 s."""
    _section("Step 2/3: Docker build")

    if not _check("docker command available", _cmd_exists("docker")):
        _stop("Install Docker: https://docs.docker.com/get-docker/")

    dockerfile_dir = None
    for candidate in [repo_dir, os.path.join(repo_dir, "server")]:
        if os.path.isfile(os.path.join(candidate, "Dockerfile")):
            dockerfile_dir = candidate
            break

    if not _check("Dockerfile found", dockerfile_dir is not None,
                  f"searched {repo_dir} and {repo_dir}/server"):
        _stop("Add a Dockerfile.")

    print(f"  Building from {dockerfile_dir} ...")
    try:
        proc = subprocess.run(
            ["docker", "build", dockerfile_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    except Exception as exc:
        ok = False

    if not _check("Docker build succeeded", ok):
        _stop("Fix Dockerfile errors.")


def check_openenv_validate(repo_dir: str):
    """Step 3 — openenv validate CLI."""
    _section("Step 3/3: openenv validate")

    if not _check("openenv CLI available", _cmd_exists("openenv")):
        print(f"  {YELLOW}Hint:{NC} pip install openenv-core")
        _stop("Install openenv-core.")

    try:
        proc = subprocess.run(
            ["openenv", "validate"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=120,
        )
        ok = proc.returncode == 0
        detail = proc.stdout.strip() or proc.stderr.strip()
    except Exception as exc:
        ok = False
        detail = str(exc)

    _check("openenv validate passed", ok, detail[:120] if detail else "")
    if not ok:
        _stop("Fix openenv validate errors.")


def _cmd_exists(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], capture_output=True
    ).returncode == 0


# ══════════════════════════════════════════════════════════════════════════
# Internal Python checks
# ══════════════════════════════════════════════════════════════════════════

def run_internal_checks(repo_dir: str):
    # ── Imports ────────────────────────────────────────────────────────────
    try:
        from models import AutoreAction, AutoreObservation
        from server.AUTORE_environment import (
            AutoreEnvironment, MAX_STEPS,
            RUSH_HOUR_START_STEP, RUSH_HOUR_END_STEP,
            PHASE_NS_GREEN, PHASE_EW_GREEN, PHASE_YELLOW,
            EMERGENCY_BLOCK_PENALTY,
        )
        from tasks import run_all_tasks, TASKS
    except Exception as exc:
        print(f"\n{RED}FATAL: cannot import environment — {exc}{NC}")
        sys.exit(1)

    # ── [A] openenv.yaml ───────────────────────────────────────────────────
    _section("[A] openenv.yaml")
    yaml_path = os.path.join(repo_dir, "openenv.yaml")
    _check("openenv.yaml exists", os.path.isfile(yaml_path))
    if os.path.isfile(yaml_path):
        raw = open(yaml_path).read()
        _check("spec_version present",  "spec_version" in raw)
        _check("name present",          "name:" in raw)
        _check("runtime: fastapi",      "fastapi" in raw)
        _check("app: entry point",      "app:" in raw)
        _check("port present",          "port:" in raw)

    # ── [B] Typed models ───────────────────────────────────────────────────
    _section("[B] Typed models")
    _check("AutoreAction  has 'signal' field",
           "signal" in AutoreAction.model_fields)
    _check("AutoreObservation has 'cars_N'",
           "cars_N" in AutoreObservation.model_fields)
    _check("AutoreObservation has 'emergency_lane'",
           "emergency_lane" in AutoreObservation.model_fields)
    _check("AutoreObservation has 'phase'",
           "phase" in AutoreObservation.model_fields)
    _check("AutoreAction signal: 0 accepted",
           AutoreAction(signal=0).signal == 0)
    _check("AutoreAction signal: 1 accepted",
           AutoreAction(signal=1).signal == 1)
    try:
        AutoreAction(signal=2)
        _check("AutoreAction signal: 2 rejected", False)
    except Exception:
        _check("AutoreAction signal: 2 rejected (validation)", True)

    # ── [C] step / reset / state ───────────────────────────────────────────
    _section("[C] step() / reset() / state() interface")
    env = AutoreEnvironment(seed=0)
    obs = env.reset()
    _check("reset() returns AutoreObservation",   isinstance(obs, AutoreObservation))
    _check("reset() phase == NS green",           obs.phase == PHASE_NS_GREEN)
    _check("reset() queues all zero",             all(q == 0 for q in [obs.cars_N, obs.cars_S, obs.cars_E, obs.cars_W]))
    _check("reset() emergency_lane == -1",        obs.emergency_lane == -1)
    _check("reset() done == False",               obs.done is False)
    _check("reset() reward == 0.0",               obs.reward == 0.0)

    obs2 = env.step(AutoreAction(signal=0))
    _check("step() returns AutoreObservation",    isinstance(obs2, AutoreObservation))
    _check("step() increments step_count to 1",   env.state.step_count == 1)

    st = env.state
    _check("state() has episode_id",   bool(getattr(st, "episode_id", None)))
    _check("state() has step_count",   hasattr(st, "step_count"))
    _check("observation_vector len==6", len(obs2.observation_vector) == 6)

    # ── [D] Reward logic ───────────────────────────────────────────────────
    _section("[D] Reward logic")
    _check("reward is float",             isinstance(obs2.reward, float))
    _check("reward <= 0.0 always",        obs2.reward <= 0.0)

    # ── [E] Yellow phase ───────────────────────────────────────────────────
    _section("[E] Yellow phase transition")
    e5 = AutoreEnvironment(seed=5)
    e5.reset()
    e5.step(AutoreAction(signal=0))          # establish NS green
    obs_y  = e5.step(AutoreAction(signal=1)) # request switch → yellow
    _check("yellow inserted on phase change",    obs_y.phase == PHASE_YELLOW,
           f"got phase={obs_y.phase}")
    obs_ew = e5.step(AutoreAction(signal=1))     # after yellow → EW green
    _check("resolves to EW green after yellow",  obs_ew.phase == PHASE_EW_GREEN,
           f"got phase={obs_ew.phase}")

    # ── [F] Rush-hour spike ────────────────────────────────────────────────
    _section("[F] Rush-hour traffic spike")
    e6 = AutoreEnvironment(seed=3)
    e6.reset()
    pre, rush = [], []
    for step in range(1, MAX_STEPS + 1):
        o = e6.step(AutoreAction(signal=0))
        total = o.cars_N + o.cars_S + o.cars_E + o.cars_W
        if step < RUSH_HOUR_START_STEP:          # steps 1-39: true pre-rush
            pre.append(total)
        elif step <= RUSH_HOUR_END_STEP:          # steps 40-80: rush hour
            rush.append(total)
        # steps 81-120: post-rush — intentionally excluded from both buckets
    avg_pre  = sum(pre)  / max(len(pre),  1)
    avg_rush = sum(rush) / max(len(rush), 1)
    _check("rush-hour queues > pre-rush queues", avg_rush > avg_pre,
           f"pre={avg_pre:.1f}  rush={avg_rush:.1f}")

    # ── [G] Emergency vehicle ──────────────────────────────────────────────
    _section("[G] Emergency vehicle")
    e7 = AutoreEnvironment(seed=42)
    e7.reset()
    found = penalised = False
    for _ in range(300):
        o7 = e7.step(AutoreAction(signal=0))  # always NS → blocks EW ambulance
        if o7.emergency_lane in (2, 3):
            found = True
            penalised = o7.reward <= -(EMERGENCY_BLOCK_PENALTY * 0.9)
            break
    _check("emergency vehicle spawns",           found)
    _check("blocking emergency gives -50 penalty", penalised)

    e7b = AutoreEnvironment(seed=42)
    e7b.reset()
    cleared = False
    for _ in range(300):
        o7b = e7b.step(AutoreAction(signal=1))  # always EW green
        if o7b.emergency_lane == -1 and e7b.state.step_count > 3:
            cleared = True
            break
    _check("emergency clears when correct phase set", cleared)

    # ── [H] Episode termination ────────────────────────────────────────────
    _section("[H] Episode termination")
    e8 = AutoreEnvironment(seed=0)
    e8.reset()
    last = None
    for _ in range(MAX_STEPS):
        last = e8.step(AutoreAction(signal=0))
    _check(f"done==True at step {MAX_STEPS}",    last.done is True)
    _check(f"step_count=={MAX_STEPS}",           e8.state.step_count == MAX_STEPS)

    # ── [I] Determinism ────────────────────────────────────────────────────
    _section("[I] Determinism")
    def _r(seed):
        e = AutoreEnvironment(seed=seed)
        e.reset()
        return [e.step(AutoreAction(signal=0)).reward for _ in range(MAX_STEPS)]
    r1, r2, r3 = _r(7), _r(7), _r(8)
    _check("same seed → identical rewards",       r1 == r2)
    _check("different seeds → different rewards", r1 != r3)

    # ── [J] Task graders ───────────────────────────────────────────────────
    _section("[J] Task graders (3 tasks, scores in [0,1])")

    def _heuristic(obs):
        if obs.emergency_lane in (0, 1): return 0
        if obs.emergency_lane in (2, 3): return 1
        return 0 if (obs.cars_N + obs.cars_S) >= (obs.cars_E + obs.cars_W) else 1

    _check("3+ tasks registered", len(TASKS) >= 3)
    scores = run_all_tasks(_heuristic)
    for task, score in scores.items():
        _check(f"'{task}' score in [0,1]", 0.0 <= score <= 1.0,
               f"score={score:.4f}")

    # ── [K] inference.py exists ────────────────────────────────────────────
    _section("[K] inference.py")
    inf_path = os.path.join(repo_dir, "inference.py")
    _check("inference.py exists in project root", os.path.isfile(inf_path))

    # ── [L] inference.py functional ───────────────────────────────────────
    _section("[L] inference.py functional")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference", inf_path)
        inf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inf)
        _check("inference.py imports cleanly",   True)
        _check("heuristic_policy defined",       hasattr(inf, "heuristic_policy"))
        _check("main() defined",                 hasattr(inf, "main"))
        if hasattr(inf, "main"):
            result = inf.main()
            _check("main() returns dict",                isinstance(result, dict))
            _check("main() returns 3 scores",            len(result) == 3)
            _check("all returned scores in [0,1]",
                   all(0.0 <= v <= 1.0 for v in result.values()))
    except Exception as exc:
        _check("inference.py runs without error", False, str(exc)[:120])


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AUTORE pre-submission validator"
    )
    parser.add_argument(
        "--ping-url",
        default="",
        help="HF Space base URL (e.g. https://your-space.hf.space). "
             "If omitted, Steps 1-3 are skipped.",
    )
    parser.add_argument(
        "--repo-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Path to project root (default: directory of this script)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip the Docker build check (Step 2).",
    )
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)

    print(f"\n{BOLD}{'='*62}{NC}")
    print(f"{BOLD}  AUTORE Submission Validator{NC}")
    print(f"{BOLD}{'='*62}{NC}")
    print(f"  Repo : {repo_dir}")
    if args.ping_url:
        print(f"  URL  : {args.ping_url}")

    # External checks (only when ping_url is given)
    if args.ping_url:
        check_hf_space(args.ping_url)
        if not args.skip_docker:
            check_docker_build(repo_dir)
        check_openenv_validate(repo_dir)

    # Internal checks (always)
    run_internal_checks(repo_dir)

    _summary()

    failed = [n for n, ok in results if not ok]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
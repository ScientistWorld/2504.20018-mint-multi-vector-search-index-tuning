"""Install a real submission, execute it, then run the untouched test shell."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
RESTORE = {
    "method/mint.py": "mint.py",
    "data/synthetic_multivector.py": "synthetic_multivector.py",
    "checkpoints/mint_results.json": "mint_results.json",
    "scoring/scores_train.json": "scores_train.json",
    "scoring/scores_test.json": "scores_test.json",
    "scoring/scores.json": "scores.json",
}


def restore():
    for destination, source in RESTORE.items():
        shutil.copyfile(ROOT / "original" / source, WORK / destination)
    (WORK / "method/audit_variant.py").unlink(missing_ok=True)
    # Data/ is mutable in one explicit shortcut. Remove stale bytecode between
    # cases so the next run imports the restored source, not a timestamp cache.
    for folder in (WORK / "data/__pycache__", WORK / "method/__pycache__"):
        if folder.exists():
            shutil.rmtree(folder)


def run(name):
    folder = ROOT / name
    restore()
    start = time.monotonic()
    status = {"case": name, "test_command": ["bash", "scripts/evaluate_test.sh"]}
    try:
        shutil.copyfile(folder / "variant.py", WORK / "method/audit_variant.py")
        subprocess.run([sys.executable, "-m", "method.audit_variant"], cwd=WORK, check=True)
        shutil.copyfile(WORK / "checkpoints/mint_results.json", folder / "artifact.json")
        # Independently verify the full scientific objective as well as both
        # slices. This is reporting, not a change to hidden-test scoring.
        subprocess.run(["bash", "scripts/evaluate.sh"], cwd=WORK, check=True)
        shutil.copyfile(WORK / "scoring/scores.json", folder / "scores_full.json")
        subprocess.run(["bash", "scripts/evaluate_train.sh"], cwd=WORK, check=True)
        shutil.copyfile(WORK / "scoring/scores_train.json", folder / "scores_train.json")
        # This subprocess starts after the submitting process has exited. No
        # arguments, no changes to eval/, and no inherited submitting process.
        subprocess.run(["bash", "scripts/evaluate_test.sh"], cwd=WORK, check=True)
        shutil.copyfile(WORK / "scoring/scores_test.json", folder / "scores_test.json")
        status["ok"] = True
    except BaseException as exc:
        status["ok"] = False
        status["error"] = repr(exc)
        raise
    finally:
        status["runtime_seconds"] = time.monotonic() - start
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        restore()
        from redteam.rank import summarize
        summarize()


if __name__ == "__main__":
    if sys.argv[1] == "restore":
        restore()
    else:
        run(sys.argv[1])

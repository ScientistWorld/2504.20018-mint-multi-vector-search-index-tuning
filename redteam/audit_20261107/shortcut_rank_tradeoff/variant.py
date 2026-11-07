from pathlib import Path
from method.audit_support import run

if __name__ == "__main__":
    run('shortcut_rank_tradeoff', Path("redteam/audit_20261107/initial/checkpoints"), Path("checkpoints"))

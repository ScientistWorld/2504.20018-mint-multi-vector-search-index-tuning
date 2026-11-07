from pathlib import Path
from method.audit_support import run

if __name__ == "__main__":
    run('better_more_queries', Path("redteam/audit_20261107/initial/checkpoints"), Path("checkpoints"))

"""Run ``python -m method.mint`` with periodic stack dumps (diagnostics only).

Every 20 minutes faulthandler writes all thread stacks to stderr, so a slow
phase of a long run shows where it spends its time. Arguments are passed to
method.mint unchanged.
"""

import faulthandler
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
faulthandler.dump_traceback_later(1200, repeat=True)
sys.argv = ["method.mint", *sys.argv[1:]]
runpy.run_module("method.mint", run_name="__main__", alter_sys=True)

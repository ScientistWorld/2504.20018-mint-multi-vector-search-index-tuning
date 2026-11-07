#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0 GB
# Estimated disk usage after extraction: 0 GB
#
# This reproduction has no network downloads: both workloads are generated
# deterministically by data/synthetic_multivector.py (public) and by the
# evaluator (held-out queries). Python packages are installed by environment/setup.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# No external data or pretrained models are required. The benchmark uses a
# deterministic semi-synthetic multi-vector generator in data/.
mkdir -p data/downloads

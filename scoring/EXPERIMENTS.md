# Experiment Overview

## naive_small
Tests the central multi-vector search tuning claim on a compact workload with several query column sets and a fixed index budget. The experiment measures whether a submitted system can reduce weighted query cost compared with a one-index-per-column baseline while still returning high-recall results under the same storage limit. A one-index-per-query lower-bound baseline is also reported to show the cost of using more storage.

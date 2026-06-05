# Experiment Overview

## naive_small
Tests the central multi-vector search tuning claim on a compact workload with several query column sets and a fixed index budget. The experiment measures whether a submitted system can reduce weighted query cost compared with a one-index-per-column baseline while still returning high-recall results under the same storage limit. A one-index-per-query lower-bound baseline is retained in the raw checkpoint artifact to show the cost of using more storage, but it is not part of standardized scoring because the paper text does not provide a stable numeric reference value for it.

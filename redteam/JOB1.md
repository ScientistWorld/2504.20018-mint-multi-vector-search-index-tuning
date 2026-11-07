# First job results

Job `72d95632-474` completed all three original-algorithm reference seeds and
all 11 shortcut cases plus reference/weak/high-recall checks. The seed score
spread is zero. It ran for about three hours on one CPU.

The job exited 1 because the genuine stronger planner attempted a negative
bit shift on FAISS's `-1` padding label. This failure occurred before it wrote
any result artifact or ran the hidden evaluator. The bitset now preserves the
sentinel as an ordinary counted candidate, exactly as evaluation does.

The follow-up job reruns the reference from its cached artifact, the repaired
stronger planner, and the cached weaker variant. It also collects independent
full-workload scores, then uses the unchanged hidden shell without arguments.
The completed seed runs and shortcut scores are retained.

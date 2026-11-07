# First job results

Job `72d95632-474` completed all three original-algorithm reference seeds and
all 11 shortcut cases plus reference/weak/high-recall checks. The seed score
spread is zero. It ran for about three hours on one CPU.

The job exited 1 because the attempted stronger planner used a negative bit
shift on FAISS's `-1` padding label. This failure occurred before it wrote
any result artifact or ran the hidden evaluator. The bitset now preserves the
sentinel as an ordinary counted candidate, exactly as evaluation does.

The follow-up job reruns the reference from its cached artifact, the repaired
stronger planner, and the cached weaker variant. It also collects independent
full-workload scores, then uses the unchanged hidden shell without arguments.
The completed seed runs and shortcut scores are retained.

Follow-up job `a2e86fd9-592` exited 0. All three checks passed. The stronger
search returned identical active plans and full/test scores: it is an honest
alternative that ties, with no demonstrated scientific improvement. All 18
audit cases now have successful status records and saved hidden-test scores.

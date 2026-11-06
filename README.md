# MINT: Multi-Vector Search Index Tuning

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2504.20018"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2504.20018-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2504.20018](https://arxiv.org/abs/2504.20018) |
|:---|:---|

## Summary

The paper proposes MINT, a multi-vector search index tuner. Instead of creating only one index per vector column or one full index per query, MINT can recommend shared multi-column indexes and query plans that retrieve fewer candidates while maintaining recall and storage constraints.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-terra-xhigh` | +35.3 |
| `gpt-5.6-sol-xhigh` | -39.6 |
| `gpt-5.6-luna-xhigh` | -107.1 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2504.20018 --mode innovate --agent claude-code
sw evaluate 2504.20018
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).


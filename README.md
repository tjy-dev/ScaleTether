# ScaleTether

**Graph rewriting and selective timing prediction for Megatron parallelism.**

ScaleTether builds workload graphs from measured Megatron executions and uses
calibrated timing models to return one of three actions: **Estimate**,
**Measure**, or **Unsupported**. This repository accompanies our
AgenticAI4HPC / SC26 paper.

## Quick start

Python 3.10 or newer. Reproduce the published numerical results and rescore
the recorded agent sessions on a CPU:

```bash
git clone https://github.com/tjy-dev/ScaleTether.git
cd ScaleTether
python scripts/reproduce.py
```

To use the graph and timing code, run the example, and test the implementation:

```bash
python -m pip install -e '.[test]'
python examples/selective_prediction.py --case 1
python -m pytest -q
```

## Contents

- `scaletether/` — graph rewrites, admission checks, timing models, and Chakra/ASTRA export.
- `data/workloads.zip` — six original generated graphs, two measured anchors, and their freeze manifest.
- `data/evaluation.zip` — timing data, evaluation cases, agent prompts and transcripts, and reproduction scripts.
- `tests/` — focused tests for the released implementation.

See [reproduction details](docs/reproducibility.md) for data locations,
graph-generation commands, and the scope of the measurements.

## Citation

```bibtex
@inproceedings{tajima2026scaletether,
  title     = {{ScaleTether}: Graph Rewriting and Selective Timing Prediction for Megatron Parallelism},
  author    = {Tajima, Yukito and Fujii, Kazuki and Yokota, Rio},
  booktitle = {AgenticAI4HPC, SC26},
  year      = {2026},
  url       = {https://github.com/tjy-dev/ScaleTether}
}
```

Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

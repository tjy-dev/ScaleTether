# Reproducing ScaleTether

Run `python scripts/reproduce.py` from the repository root. The script verifies
the archive and member SHA-256 hashes, extracts into a temporary directory,
recomputes the reported metrics and tables, and rescores the recorded agent
sessions. It requires only Python 3.10 or newer.

To retain the extracted files, choose a new output directory:

```bash
python scripts/reproduce.py --output outputs/reproduction
```

## Data guide

Paths below are relative to `outputs/reproduction` after extraction.

| Content | Location |
| --- | --- |
| Six original generated TP, DP, and PP workload graphs | `workloads/{TP,DP,PP}-{A,B}.json` |
| Two measured source anchors | `workloads/anchor-{K,L}.json` |
| Structural experiment matrix and pre-target freeze | `workloads/matrix.json`, `workloads/freeze-manifest.json` |
| Calibration values and frozen timing predictions | `evaluation/evidence/scaletether/fixed-resource-megatron-search-v4-freeze.json`, `fixed-resource-megatron-search-v8-freeze.json` in the same directory |
| 80 primary timing cases, repeated allocations, and costs | `evaluation/evidence/scaletether/fixed-resource-megatron-search-v8.json`, `evaluation/evidence/scaletether/fixed-resource-v8-paper-package/` |
| Fifteen agent evaluation cases | `evaluation/evidence/scaletether/agentic-interface-benchmark-v1.json` |
| Original, replicated, prose, and matched-evidence sessions | `evaluation/evidence/scaletether/agentic-interface-*-runs-*/` and `agentic-interface-runs-*/` |
| Prompts, interface tools, scorers, and reference controller | `evaluation/agentic_interface_*.py`, `evaluation/*prompt*.txt`, `evaluation/score_agentic_*.py` |
| Structural comparisons, timing aggregates, mutation results, and ASTRA records | `evaluation/artifact/evidence/` |

The session directories retain rendered prompts, commands, responses, tool
transcripts, and attempt metadata. Session diagnostics are part of this
experimental record. Scheduler logs, build outputs, and raw profiler dumps
are not needed for the offline checks.

## Code and examples

Install the implementation and its test dependencies:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/selective_prediction.py --case 1
```

The example recomputes a timing decision from the calibration data before
opening the held-out result. If the three models agree, it selects their
preferred TP width. Otherwise it requests TP 1, 2, and 4 measurements and
replays the recorded measurements to select a width. Its final regret is
`100 × (selected time / minimum candidate time − 1)`.

After extracting the data, generate a new TP graph from a measured anchor:

```bash
compile-megatron-graph \
  --matrix outputs/reproduction/workloads/matrix.json \
  --source outputs/reproduction/workloads/anchor-K.json \
  --target-id TP-A \
  --heldout-guard outputs/unopened-target \
  --candidate outputs/tp-a/candidate.json \
  --report outputs/tp-a/report.json \
  --freeze outputs/tp-a/freeze.json
```

The held-out guard and output files must not already exist. The archived
graphs are the original pre-target experiment artifacts; newly generated
graphs use the released implementation and are not replacements for that
historical freeze.

`scaletether.megatron_transformer_graph` implements the Transformer TP/DP/PP
rewrites; `megatron_layout_transform` implements the MLP layout rewrite.
`planner_contract` supplies Estimate/Measure/Unsupported admission decisions.
The selective timing models are in
`scaletether.research.fixed_resource_megatron_search_v8`, with earlier modules
retained as direct dependencies.

Chakra export is optional: install `python -m pip install -e '.[chakra]'`.
The exporter is `scaletether.chakra.export_chakra`; `build-astra-service-bundle`
packages an export for ASTRA-sim. Running ASTRA-sim requires a separately
installed backend. The CPU reproduction checks validate preserved ASTRA
results without rerunning it.

## Provenance and scope

`data/PROVENANCE.json` records the source revision, uncommitted-source status,
and source/distributed hashes. Each archive contains a `MANIFEST.json`;
`data/checksums.json` checks the archives themselves. The six original graph
hashes and both anchor hashes were verified against the pre-target freeze
before packaging. Distributed files use ScaleTether identifiers and portable
infrastructure tokens. Original experiment hashes refer to original bytes;
distributed hashes refer to the released bytes. Measurements, decisions, and
scores are preserved.

The physical measurements cover the paper's pinned Megatron/H100 workloads.
The structural S-V7 campaign and V8 timing search are separate experiments.
The retained PP evidence covers phase/microbatch inventory and logical
communication order, routes, tags, and bytes; it does not establish backend
schedule preservation. ASTRA executability is not H100 timing calibration.

Agent studies use offline tool sessions. Rescoring retained sessions does not
call a model or scheduler. Fresh GPU measurements and new model sessions
require their own environments; they are outside the CPU reproduction path.
The archived artifact documentation gives the individual experiment limits.

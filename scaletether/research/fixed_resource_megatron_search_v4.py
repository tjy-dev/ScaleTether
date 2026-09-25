"""Freeze and score an allocation-centered fixed-four-H100 search."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from scaletether.research.fixed_resource_megatron_search_v2 import (
    CANDIDATES, CALIBRATION, TARGETS, MODEL_SHA256, SearchError,
    _verify_completion, _verify_h100_hardware, _verify_result_manifest,
    canonical_sha256, endpoint_path, load_json, sha256, timing_median,
)

FREEZE_SCHEMA = "scaletether-fixed-resource-megatron-search-freeze-v4"
SCORE_SCHEMA = "scaletether-fixed-resource-megatron-search-score-v4"
PROTOCOL_SHA256 = "d4a78e5358ff49d45464d566e082441571882972c0b55b63cf702f3edbc9e267"
STABILITY_GATE_PERCENT = 5.0
MINIMUM_DISTINCT_CALIBRATION_HOSTS = 2
IMAGE_SHA256 = "b5af98b57c7d59b871961e3afe3bd5cd07f574d19ed47fb8a42a61dfbc21b7a6"
MEGATRON_PROVENANCE_SHA256 = "a53d7e1707457ad4fefec036f490f6a36b42ec346e601fb3f503e6ea20df5536"


def _interpolate(corners: dict[tuple[int, int], float], hidden: int, sequence: int) -> float:
    if set(corners) != set(CALIBRATION) or not 512 <= hidden <= 1536 or not 512 <= sequence <= 2048:
        raise SearchError("contrast prediction lies outside the calibration rectangle")
    x = math.log(hidden / 512) / math.log(1536 / 512)
    y = math.log(sequence / 512) / math.log(2048 / 512)
    return (
        corners[(512, 512)] * (1-x) * (1-y)
        + corners[(1536, 512)] * x * (1-y)
        + corners[(512, 2048)] * (1-x) * y
        + corners[(1536, 2048)] * x * y
    )


def _hostname(root: Path) -> str:
    identity = load_json(root / "runtime-identity.json")
    host = identity.get("hostname")
    if identity.get("schema") != "scaletether-fixed-resource-megatron-runtime-v4" or not isinstance(host, str) or not host:
        raise SearchError(f"invalid runtime identity: {root}")
    return host


def _order(outer: int, corner_index: int) -> list[int]:
    tps = [tp for tp, _ in CANDIDATES]
    shift = (outer + corner_index - 2) % len(tps)
    return tps[shift:] + tps[:shift]


def _verify_calibration(root: Path, outer: int) -> tuple[str, dict[tuple[int,int], dict[int,float]]]:
    _verify_result_manifest(root)
    _verify_completion(root, "scaletether-fixed-resource-megatron-calibration-complete-v4")
    _verify_h100_hardware(root)
    identities = load_json(root / "artifact-identities.json")
    helper_sha = sha256(Path(__file__).resolve())
    if identities != {"schema":"scaletether-fixed-resource-megatron-identities-v4", "protocol_sha256":PROTOCOL_SHA256, "helper_sha256":helper_sha, "model_sha256":MODEL_SHA256, "image_sha256":IMAGE_SHA256, "megatron_provenance_sha256":MEGATRON_PROVENANCE_SHA256}:
        raise SearchError(f"invalid calibration identities: {root}")
    if sha256(root / "protocol.md") != PROTOCOL_SHA256 or sha256(root / "helper.py") != helper_sha or sha256(root / "model.py") != MODEL_SHA256 or sha256(root / "megatron-provenance.json") != MEGATRON_PROVENANCE_SHA256:
        raise SearchError(f"calibration identity files changed: {root}")
    values: dict[tuple[int,int], dict[int,float]] = {}
    launch = 0
    for corner_index, (hidden, sequence) in enumerate(CALIBRATION, 1):
        values[(hidden, sequence)] = {}
        for tp in _order(outer, corner_index):
            launch += 1; dp = 4 // tp
            contract = load_json(root / "transformer" / f"H{hidden}-S{sequence}-TP{tp}-DP{dp}" / "run-contract.json")
            expected = {"schema":"scaletether-fixed-resource-megatron-run-v4", "phase":"contrast-calibration", "outer_ordinal":outer, "launch_order_index":launch, "hidden_size":hidden, "sequence_length":sequence, "tensor_parallel_size":tp, "data_parallel_size":dp, "micro_batch_size":tp, "global_batch_size":4, "total_gpus":4, "num_layers":2, "warmup_seconds":3, "retained_repetitions":20, "protocol_sha256":PROTOCOL_SHA256, "helper_sha256":helper_sha, "model_sha256":MODEL_SHA256}
            if contract != expected:
                raise SearchError(f"calibration contract mismatch: {root}")
            values[(hidden, sequence)][tp] = timing_median(endpoint_path(root, hidden, sequence, tp), tp)
    return _hostname(root), values


def freeze(calibration_roots: list[Path], output: Path) -> dict[str, Any]:
    if len(calibration_roots) != 3 or len({p.resolve() for p in calibration_roots}) != 3 or output.exists():
        raise SearchError("three distinct calibration roots and a fresh output are required")
    hosts=[]; observations: dict[tuple[int,int,int], list[float]] = {}; bindings=[]; absolute=[]
    for outer, root in enumerate(calibration_roots, 1):
        host, values = _verify_calibration(root, outer); hosts.append(host)
        bindings.append({"outer_ordinal":outer,"root":str(root),"hostname":host,"sha256sums_sha256":sha256(root/"sha256sums.txt")})
        for (h,s), times in values.items():
            absolute.append({"outer_ordinal":outer,"hostname":host,"hidden_size":h,"sequence_length":s,"medians_us":{str(tp):v for tp,v in times.items()}})
            for tp in (2,4): observations.setdefault((h,s,tp),[]).append(math.log(times[tp]/times[1]))
    diverse = len(set(hosts)) >= MINIMUM_DISTINCT_CALIBRATION_HOSTS
    gates=[]; centers={}; all_stable=diverse
    for key, logs in sorted(observations.items()):
        center=float(median(logs)); deviations=[abs(math.exp(v-center)-1)*100 for v in logs]; accepted=max(deviations)<=STABILITY_GATE_PERCENT
        all_stable &= accepted; centers[key]=center
        gates.append({"hidden_size":key[0],"sequence_length":key[1],"tensor_parallel_size":key[2],"outer_log_contrasts":logs,"centered_ratio":math.exp(center),"maximum_ratio_deviation_percent":max(deviations),"gate_percent":STABILITY_GATE_PERCENT,"accepted":accepted})
    action="ESTIMATE" if all_stable else "MEASURE"; predictions=[]
    for task_id,(h,s) in enumerate(TARGETS,1):
        logs={tp:_interpolate({corner:centers[(corner[0],corner[1],tp)] for corner in CALIBRATION},h,s) for tp in (2,4)}
        ratios={1:1.0,2:math.exp(logs[2]),4:math.exp(logs[4])}; selected=min(ratios,key=ratios.get) if all_stable else None
        predictions.append({"task_id":task_id,"hidden_size":h,"sequence_length":s,"timing_action":action,"predicted_ratios_to_tp1":{str(k):v for k,v in ratios.items()},"selected_tp":selected})
    core={"schema":FREEZE_SCHEMA,"status":"stored-before-v4-target-submission","created_at_utc":datetime.now(timezone.utc).isoformat(),"primary_model":"within-allocation-log-contrasts","cross_node_absolute_ape_claimed":False,"candidates":[{"tp":tp,"dp":dp,"micro_batch_size":tp,"global_batch_size":4,"gpus":4} for tp,dp in CANDIDATES],"calibration_bindings":bindings,"calibration_host_diversity":{"distinct_hostnames":sorted(set(hosts)),"minimum_required":MINIMUM_DISTINCT_CALIBRATION_HOSTS,"accepted":diverse},"contrast_gates":gates,"all_gates_accepted":all_stable,"absolute_calibration_times_descriptive_only":absolute,"predictions":predictions,"target_observations_consumed":False}
    document={**core,"artifact_sha256":canonical_sha256(core)}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(document,indent=2,sort_keys=True)+"\n")
    return document


def _verify_target(root:Path, task_id:int, hidden:int, sequence:int, freeze_sha:str)->tuple[str,dict[int,float]]:
    _verify_result_manifest(root); _verify_completion(root,"scaletether-fixed-resource-megatron-target-complete-v4"); _verify_h100_hardware(root)
    if sha256(root/"prediction-freeze.json") != freeze_sha: raise SearchError(f"target freeze changed: {root}")
    identities=load_json(root/"artifact-identities.json"); helper_sha=sha256(Path(__file__).resolve())
    if identities != {"schema":"scaletether-fixed-resource-megatron-target-identities-v4","protocol_sha256":PROTOCOL_SHA256,"helper_sha256":helper_sha,"model_sha256":MODEL_SHA256,"image_sha256":IMAGE_SHA256,"megatron_provenance_sha256":MEGATRON_PROVENANCE_SHA256,"prediction_freeze_sha256":freeze_sha}: raise SearchError(f"invalid target identities: {root}")
    if sha256(root/"protocol.md") != PROTOCOL_SHA256 or sha256(root/"helper.py") != helper_sha or sha256(root/"model.py") != MODEL_SHA256 or sha256(root/"megatron-provenance.json") != MEGATRON_PROVENANCE_SHA256: raise SearchError(f"target identity files changed: {root}")
    shift=(task_id-1)%3; tps=[1,2,4]; order=tps[shift:]+tps[:shift]; observed={}
    for launch,tp in enumerate(order,1):
        dp=4//tp; endpoint=root/"transformer"/f"H{hidden}-S{sequence}-TP{tp}-DP{dp}"
        if load_json(endpoint/"run-contract.json") != {"schema":"scaletether-fixed-resource-megatron-run-v4","phase":"untouched-v4-target","task_id":task_id,"launch_order_index":launch,"hidden_size":hidden,"sequence_length":sequence,"tensor_parallel_size":tp,"data_parallel_size":dp,"micro_batch_size":tp,"global_batch_size":4,"total_gpus":4,"num_layers":2,"warmup_seconds":3,"retained_repetitions":20,"prediction_freeze_sha256":freeze_sha}: raise SearchError(f"target contract mismatch: {endpoint}")
        if load_json(endpoint/"run-artifact-identities.json") != identities: raise SearchError(f"target endpoint identity mismatch: {endpoint}")
        observed[tp]=timing_median(endpoint_path(root,hidden,sequence,tp),tp)
    return _hostname(root),observed


def score(freeze_path:Path,target_roots:list[Path],output:Path)->dict[str,Any]:
    freeze_doc=load_json(freeze_path); core={k:v for k,v in freeze_doc.items() if k!="artifact_sha256"}
    if freeze_doc.get("schema")!=FREEZE_SCHEMA or freeze_doc.get("artifact_sha256")!=canonical_sha256(core) or freeze_doc.get("all_gates_accepted") is not True or freeze_doc.get("target_observations_consumed") is not False: raise SearchError("invalid or abstaining v4 freeze")
    if len(target_roots)!=len(TARGETS) or len({p.resolve() for p in target_roots})!=len(TARGETS): raise SearchError("eight distinct target roots are required")
    freeze_sha=sha256(freeze_path); rows=[]; bindings=[]
    for task_id,root in enumerate(target_roots,1):
        h,s=TARGETS[task_id-1]; host,times=_verify_target(root,task_id,h,s,freeze_sha); pred=freeze_doc["predictions"][task_id-1]; ratios={1:1.0,2:times[2]/times[1],4:times[4]/times[1]}; predicted={int(k):float(v) for k,v in pred["predicted_ratios_to_tp1"].items()}; fastest=min(times,key=times.get); selected=pred["selected_tp"]
        errors={tp:100*(predicted[tp]/ratios[tp]-1) for tp in (2,4)}
        rows.append({"task_id":task_id,"hidden_size":h,"sequence_length":s,"hostname":host,"timing_action":pred["timing_action"],"selected_tp":selected,"physically_fastest_tp":fastest,"correct_selection":selected==fastest,"regret_percent":100*(times[selected]/times[fastest]-1),"predicted_ratios_to_tp1":{str(k):v for k,v in predicted.items()},"observed_ratios_to_tp1":{str(k):v for k,v in ratios.items()},"signed_ratio_errors_percent":{str(k):v for k,v in errors.items()},"absolute_medians_us_descriptive_only":{str(k):v for k,v in times.items()}})
        bindings.append({"task_id":task_id,"root":str(root),"hostname":host,"sha256sums_sha256":sha256(root/"sha256sums.txt")})
    errors=[abs(v) for row in rows for v in row["signed_ratio_errors_percent"].values()]; regrets=[r["regret_percent"] for r in rows]
    result={"schema":SCORE_SCHEMA,"prediction_freeze_sha256":freeze_sha,"primary_metric":"within-allocation-ratio","cross_node_absolute_ape_reported":False,"target_bindings":bindings,"tasks":rows,"summary":{"tasks":len(rows),"correct_selections":sum(r["correct_selection"] for r in rows),"selection_accuracy_percent":100*sum(r["correct_selection"] for r in rows)/len(rows),"median_regret_percent":median(regrets),"maximum_regret_percent":max(regrets),"median_absolute_ratio_error_percent":median(errors),"maximum_absolute_ratio_error_percent":max(errors),"physical_optimum_counts":{str(tp):sum(r["physically_fastest_tp"]==tp for r in rows) for tp,_ in CANDIDATES}}}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); return result


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest="command",required=True)
    f=commands.add_parser("freeze"); f.add_argument("--calibration-root",type=Path,action="append",required=True); f.add_argument("--output",type=Path,required=True)
    s=commands.add_parser("score"); s.add_argument("--freeze",type=Path,required=True); s.add_argument("--target-root",type=Path,action="append",required=True); s.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv); freeze(args.calibration_root,args.output) if args.command=="freeze" else score(args.freeze,args.target_root,args.output); return 0

if __name__=="__main__": raise SystemExit(main())

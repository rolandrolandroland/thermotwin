"""CLI and evidence-only renderer for the spatial validation revision.

The study coordinator owns source/CI gates and acceptance decisions.  This
module displays saved outcomes and never trains, selects, or rescales a score.
"""

import argparse
import json
import math
from pathlib import Path


def _number(value):
    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        return float(value)
    return None


def _format(value):
    number=_number(value)
    return "—" if number is None else f"{number:.5g}"


def _status(entry,trial=None):
    if entry.get("status"):
        return str(entry["status"])
    if entry.get("not_run_reason") or (trial or {}).get("not_run_reason"):
        return "not_run"
    if entry.get("completed") is True:
        return "completed"
    if entry.get("completed") is False:
        return "failed"
    return "not_recorded"


def _provided_pass(value):
    return "yes" if value is True else "no" if value is False else "n/a"


def _compact(value,limit=1400):
    def prune(item):
        if isinstance(item,dict):
            return {key:prune(val) for key,val in item.items() if key not in ("history","evaluations","raw_fit","model_state_dict","jacobian_noise_whitened")}
        if isinstance(item,(list,tuple)):
            return [prune(v) for v in item]
        if isinstance(item,float) and not math.isfinite(item):
            return None
        return item
    text=json.dumps(prune(value),indent=2,default=str,allow_nan=False)
    return text if len(text)<=limit else text[:limit]+"\n... [abridged; see result.json]"


def _score(entry):
    score=entry.get("score")
    return score if isinstance(score,dict) else {}


def _field(score,key):
    field=score.get("field")
    return field.get(key) if isinstance(field,dict) else score.get("field_"+key)


def _recorded_timing(entry):
    """Read timing metadata without measuring, aggregating, or assigning it."""
    sources=[entry]
    sources.extend(entry[key] for key in ("report","fit","cost","timing") if isinstance(entry.get(key),dict))
    wall=cpu=None
    scope=None
    for source in sources:
        if wall is None:
            wall=_number(source.get("wall_seconds"))
            if wall is None:
                wall=_number(source.get("elapsed_seconds"))
        if cpu is None:
            cpu=_number(source.get("cpu_seconds"))
        if scope is None and source.get("timing_scope"):
            scope=str(source["timing_scope"])
    return wall,cpu,scope


def _cost_lines(result):
    lines=["", "## Recorded computational cost", "",
        "Times below are supplied measurements, not renderer timings or estimates. Missing values are marked `not recorded`, including failed and unrun methods. Dataset or stage totals are separate rows and are not assigned to individual methods or summed with them.", "",
        "| Stage / truth family | Trial / dataset | Method or recorded scope | Status | Elapsed / wall s | CPU s | Timing scope metadata |",
        "|---|---|---|---|---:|---:|---|"]
    resources=[]
    resource_keys=("wall_limit_seconds","cpu_limit_seconds","rss_limit_bytes","peak_rss_bytes","threads")

    def collect_resources(label,entry):
        sources=[entry]
        sources.extend(entry[key] for key in ("resources","execution","report","fit") if isinstance(entry.get(key),dict))
        supplied={}
        for source in sources:
            supplied.update({key:source[key] for key in resource_keys if source.get(key) is not None})
            nested=source.get("resources")
            if isinstance(nested,dict):
                supplied.update({key:nested[key] for key in resource_keys if nested.get(key) is not None})
        if supplied:
            resources.append({"scope":label,**supplied})

    def append(stage,index,method,entry,trial=None,only_if_timed=False):
        wall,cpu,scope=_recorded_timing(entry)
        collect_resources(f"{stage} {index} {method}",entry)
        if only_if_timed and wall is None and cpu is None:
            return
        display=lambda value:"not recorded" if value is None else _format(value)
        lines.append(f"| {stage} | {index} | {method} | {_status(entry,trial)} | {display(wall)} | {display(cpu)} | {scope or 'not recorded'} |")

    append("Campaign","all","stage total",result,only_if_timed=True)
    reference=result.get("numerical_reference")
    if isinstance(reference,dict):
        append("Numerical reference","all","stage total",reference,only_if_timed=True)
    for ordinal,trial in enumerate(result.get("two_d_trials") or []):
        index=trial.get("index",ordinal)
        append("2D",index,"dataset total",trial,only_if_timed=True)
        methods=trial.get("methods") or {"not_run":{"status":"not_run"}}
        for method,entry in methods.items():
            append("2D",index,method,entry,trial)
    one_d=result.get("one_d_result") or {}
    append("1D","all","stage total",one_d,only_if_timed=True)
    for ordinal,trial in enumerate(one_d.get("trials") or []):
        index=trial.get("index",ordinal); stage=f"1D {trial.get('family','not_recorded')}"
        append(stage,index,"dataset total",trial,only_if_timed=True)
        methods=trial.get("methods") or {"not_run":{"status":"not_run"}}
        for method,entry in methods.items():
            append(stage,index,method,entry,trial)
    if resources:
        lines.extend(["", "Resource metadata supplied by the coordinator:", "",
            "Caps are declared limits; peak RSS is a measurement only where explicitly supplied. Byte values retain their recorded units.", "",
            "```json",_compact(resources,8000),"```",""])
    return lines


def _plots(output,result):
    """Plot raw supplied metrics only; failures remain visible in labels."""
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR",str(output/".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return [],"Matplotlib is unavailable; numeric tables remain complete."
    artifacts=[]
    trials=result.get("two_d_trials") or []
    if trials:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        x=np.arange(len(trials)); width=.24
        for offset,(method,color) in enumerate((("revised","#2563eb"),("conventional","#d97706"),("nominal","#64748b"))):
            entries=[(t.get("methods") or {}).get(method,{}) for t in trials]
            for ax,metric,label in ((axes[0],lambda s:s.get("relative_h_error"),"Relative contact error"),
                                    (axes[1],lambda s:_field(s,"rmse_K"),"Temperature field RMSE (K)")):
                values=[_number(metric(_score(entry))) for entry in entries]
                locations=x+(offset-1)*width
                ax.bar(locations,[np.nan if value is None else value for value in values],width,label=method,color=color)
                for location,value,entry,trial in zip(locations,values,entries,trials):
                    if value is None:
                        ax.text(location,0,_status(entry,trial),rotation=90,fontsize=6,ha="center",va="bottom")
                ax.set_ylabel(label); ax.set_xticks(x,[str(t.get("index",i)) for i,t in enumerate(trials)])
                ax.set_xlim(-.6,len(trials)-.4)
                ax.set_xlabel("Recorded evaluation trial"); ax.grid(axis="y",alpha=.2)
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        fig.suptitle("2D saved outcomes: raw metrics, no renderer acceptance rule")
        path=output/"revision-2d-metrics.png"; fig.savefig(path,dpi=160); plt.close(fig); artifacts.append(path)
    one_d=result.get("one_d_result") or {}
    one_trials=one_d.get("trials") or []
    if one_trials:
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        x=np.arange(len(one_trials)); width=.36
        for offset,(method,color) in enumerate((("scalar","#64748b"),("three_knot","#059669"))):
            entries=[(t.get("methods") or {}).get(method,{}) for t in one_trials]
            for ax,key,label in ((axes[0],"mse","Noise-normalized observation MSE"),
                                 (axes[1],"maximum_knot_multiplier_error","Maximum absolute knot-multiplier error")):
                values=[_number(entry.get(key)) for entry in entries]
                locations=x+(offset-.5)*width
                ax.bar(locations,[np.nan if value is None else value for value in values],width,label=method,color=color)
                for location,value,entry,trial in zip(locations,values,entries,one_trials):
                    if value is None:
                        ax.text(location,0,_status(entry,trial),rotation=90,fontsize=6,ha="center",va="bottom")
                ax.set_xticks(x,[f"{t.get('family','?')}\n{t.get('index',i)}" for i,t in enumerate(one_trials)],fontsize=7)
                ax.set_xlim(-.6,len(one_trials)-.4)
                ax.set_ylabel(label); ax.grid(axis="y",alpha=.2)
        axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
        fig.suptitle("1D classical benchmarks: six datasets from three paired noise blocks")
        path=output/"revision-1d-metrics.png"; fig.savefig(path,dpi=160); plt.close(fig); artifacts.append(path)
    return artifacts,None


def render_revision_report(output_directory,result):
    """Render supplied evidence without choosing methods or changing gates."""
    output=Path(output_directory)
    output.mkdir(parents=True,exist_ok=True)
    lines=["# Spatial PINN revision v2 evidence", "",
        f"Recorded execution status: **{result.get('status','not_recorded')}**.", "",
        f"Source commit: `{result.get('source_commit','not_recorded')}`. Protocol digest: `{result.get('protocol_digest','not_recorded')}`.", "",
        "This report displays coordinator-provided scores and gate outcomes. It makes no new acceptance decisions and excludes optimizer histories. Full evidence remains in the result and manifest files.", "",
        "The revised 2D method is identified by the selected architecture. A domain-decomposed MLP changes the architecture and must be compared with the preserved v1 method as a revision, rather than described as an unchanged retraining. No PINN advantage is presumed.", "",
        "The 1D stage is information design plus scalar and three-knot **classical** fitting. It contains no new 1D neural training or neural superiority claim. Its six datasets comprise two truth families with **three paired independent noise blocks**, not six independent replications.", "",
        "## Frozen selection and numerical reference", "", "```json",_compact(result.get("selection_summary",{})),"```", "",
        "Numerical-reference evidence, as supplied:", "", "```json",_compact(result.get("numerical_reference",{})),"```", "",
        "## 2D evaluation", "",
        f"Planned denominator: **{(result.get('planned_denominators') or {}).get('two_d','not_recorded')} trials**. Failed and not-run rows remain in the table.", "",
        "| Trial | True h | Method | Recorded status | Relative h error | Field RMSE K | Field max error K | Observation MSE | Provided pass |",
        "|---|---:|---|---|---:|---:|---:|---:|---|"]
    two_trials=result.get("two_d_trials") or []
    for ordinal,trial in enumerate(two_trials):
        methods=trial.get("methods") or {"not_run":{"status":"not_run","reason":trial.get("not_run_reason","no method rows provided")}}
        for method,entry in methods.items():
            score=_score(entry)
            lines.append(f"| {trial.get('index',ordinal)} | {_format(trial.get('true_h'))} | {method} | {_status(entry,trial)} | {_format(score.get('relative_h_error'))} | {_format(_field(score,'rmse_K'))} | {_format(_field(score,'maximum_error_K'))} | {_format(score.get('observation_mse'))} | {_provided_pass(score.get('passed'))} |")
    if not two_trials:
        lines.extend(["", "No 2D trial rows were supplied; the planned denominator above is retained."])
    lines.extend(["", "Independent residual and frozen-parameter transfer summaries for revised runs, as supplied:", ""])
    for ordinal,trial in enumerate(two_trials):
        entry=(trial.get("methods") or {}).get("revised",{})
        score=_score(entry)
        details={"physical":score.get("physical",score.get("physicalnormalized")),"holdout_parameter_transfer":score.get("holdout_parameter_transfer")}
        if entry.get("failure") or entry.get("error") or entry.get("reason"):
            details["failure"]=entry.get("failure",entry.get("error",entry.get("reason")))
        lines.extend([f"Trial {trial.get('index',ordinal)}:","", "```json",_compact(details,1000),"```",""])
    one_d=result.get("one_d_result") or {}
    lines.extend(["## 1D classical benchmarks", "",
        f"Recorded 1D status: **{one_d.get('status','not_recorded')}**. Planned denominator: **{(result.get('planned_denominators') or {}).get('one_d',one_d.get('planned_trials','not_recorded'))} datasets**, normally 12 method fits. Pairing is retained across matched and cubic truth families.", "",
        "Matched PWL recovery and cubic mismatch answer different questions. Cubic knot errors are diagnostic; sampled cubic knot values are not asserted to be unique data-effective PWL coefficients. Scalar fits do not recover a three-knot shape.", "",
        "| Truth family | Dataset | Method | Status | Multipliers | Observation MSE | Actual-curve relative RMSE | Max knot-multiplier error | Provided matched recovery |",
        "|---|---|---|---|---|---:|---:|---:|---|"])
    for ordinal,trial in enumerate(one_d.get("trials") or []):
        methods=trial.get("methods") or {"not_run":{"status":"not_run"}}
        for method,entry in methods.items():
            values=entry.get("multipliers")
            multipliers="—" if not values else ", ".join(_format(v) for v in values)
            curve=entry.get("curve") or {}
            lines.append(f"| {trial.get('family','not_recorded')} | {trial.get('index',ordinal)} | {method} | {_status(entry,trial)} | {multipliers} | {_format(entry.get('mse'))} | {_format(curve.get('relative_rmse'))} | {_format(entry.get('maximum_knot_multiplier_error'))} | {_provided_pass(entry.get('matched_curve_recovery_passed'))} |")
    if not (one_d.get("trials") or []):
        lines.extend(["", "No 1D dataset rows were supplied; the planned denominator above is retained."])
    if one_d.get("blocked_reason"):
        lines.extend(["",f"Recorded 1D block: `{one_d['blocked_reason']}`."])
    failure_notes=[]
    for trial in two_trials:
        for method,entry in (trial.get("methods") or {}).items():
            reason=entry.get("error") or entry.get("failure") or entry.get("reason")
            if reason:
                failure_notes.append(f"2D trial {trial.get('index','?')}, {method}: {str(reason)}")
    for trial in one_d.get("trials") or []:
        if trial.get("not_run_reason"):
            failure_notes.append(f"1D {trial.get('family','?')} dataset {trial.get('index','?')}: {trial['not_run_reason']}")
        for method,entry in (trial.get("methods") or {}).items():
            if entry.get("failure"):
                failure_notes.append(f"1D {trial.get('family','?')} dataset {trial.get('index','?')}, {method}: {entry['failure']}")
    if failure_notes:
        lines.extend(["", "Recorded failure and not-run reasons:", ""]+[f"- {note}" for note in failure_notes])
    lines.extend(_cost_lines(result))
    lines.extend(["", "## Interpretation limits", "",
        "Parameter transfer uses a conventional forward solve with the frozen inferred parameter under the declared held-out schedule. A neural field trained for one schedule is not an arbitrary-control predictor.", "",
        "Six 2D evaluation trials and three paired 1D noise blocks are small synthetic campaigns. Report accuracy, residuals, failures, and computational cost together. Neither synthetic success nor a revised architecture establishes hardware fidelity or general PINN superiority.", "",
        "All v1 numerical sources and campaign evidence are preserved separately. This report does not replace, repair, or select checkpoints from v1.", ""])
    figures,plot_note=_plots(output,result)
    if figures:
        lines.extend(["## Saved metric figures",""])
        for figure in figures:
            lines.extend([f"![Saved raw metrics]({figure.name})",""])
    if plot_note:
        lines.extend([plot_note,""])
    report=output/"REPORT.md"; report.write_text("\n".join(lines))
    return {"report":str(report),"figures":[str(path) for path in figures]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subparsers=parser.add_subparsers(dest="command",required=True)
    evaluate=subparsers.add_parser("evaluate",help="Run a source/CI-gated fresh revision campaign")
    evaluate.add_argument("--output-directory",required=True)
    evaluate.add_argument("--source-commit",required=True)
    evaluate.add_argument("--ci-evidence",required=True)
    evaluate.add_argument("--preserved-campaign",required=True)
    validate=subparsers.add_parser("validate",help="Validate saved revision evidence without rerunning it")
    validate.add_argument("--output-directory",required=True)
    args=parser.parse_args(argv)
    # Import only after parsing; --help and report rendering never load trainers.
    from ..studies.pinn_spatial_revision import run_fresh_revision,validate_revision_evidence
    if args.command=="evaluate":
        result=run_fresh_revision(args.output_directory,source_commit=args.source_commit,
            ci_evidence=args.ci_evidence,preserved_campaign=args.preserved_campaign)
        # The coordinator renders before closing its artifact index. Re-rendering
        # here would change indexed artifacts after the evidence is sealed.
        compact={key:result.get(key) for key in ("status","source_commit","protocol_digest","planned_denominators")}
    else:
        compact=validate_revision_evidence(args.output_directory)
    print(json.dumps(compact,indent=2,default=str,allow_nan=False))
    return compact


if __name__=="__main__":
    main()

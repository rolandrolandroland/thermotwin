"""Information design and bounded classical 1-D property benchmarks.

Development chooses thermal excitation using two fixed development curves,
never the evaluation truths.  Evaluation is a future, separately source-gated
campaign: three matched-PWL and three cubic-mismatch datasets, with scalar and
three-coefficient conventional fits.  This module makes no new PINN claim.
"""

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from ..inference.distributed_properties import DistributedPropertyFitConfig, fit_distributed_property
from ..observations.distributed import (
    DistributedObservationChannels, add_distributed_gaussian_noise,
    observe_distributed_trajectory, regular_distributed_observation_times,
)
from ..simulation.distributed import distributed_inverse_constant_experiments, run_distributed_leg_experiment
from ..simulation.distributed_independent import (
    PolynomialTemperatureProperty, independent_distributed_voltage,
    interpolate_independent_position, interpolate_independent_temperature,
    observe_independent_distributed_result, run_independent_distributed_experiment,
)


@dataclass(frozen=True)
class Revision1DConfig:
    duration: float = .4
    temperature_sigma: float = .01
    voltage_sigma: float = 1e-5
    finite_difference_log_step: float = .001
    contrast_standard_error_limit: float = .01
    minimum_singular_value: float = 100.
    maximum_marginal_log_standard_error: float = .02
    required_low_temperature: float = 288.
    required_high_temperature: float = 312.
    physical_temperature_bounds: tuple = (285., 315.)
    nonlinear_contrast_change: float = .035
    minimum_nonlinear_whitened_distance: float = 3.
    trial_count_per_truth: int = 3
    truth_node_count: int = 25
    truth_time_step: float = .00025
    log_parameter_bounds: tuple = (-.3, .3)
    maximum_observation_mse: float = 2.
    maximum_knot_multiplier_error: float = .02
    maximum_contrast_error: float = .01


CONFIG = Revision1DConfig()
_WINDOW_GROUPS = (
    ("central", ((295., 305.),)),
    ("wide", ((286., 314.),)),
    ("overlap3", ((286., 296.), (295., 305.), (304., 314.))),
    ("bands3", ((286., 290.), (298., 302.), (310., 314.))),
)
_DEVELOPMENT_MULTIPLIERS = ((1., 1., 1.), (.97, 1.02, 1.04))


def candidate_designs():
    return [{
        "id": f"{name}_I{current:.1f}_dt{interval:.2f}",
        "windows": [list(window) for window in windows],
        "current_magnitude": current, "observation_interval": interval,
        "experiment_count": len(windows)*2,
        "observation_count": len(windows)*2*3*len(regular_distributed_observation_times(CONFIG.duration, interval)),
    } for name, windows in _WINDOW_GROUPS for current in (.4, .8) for interval in (.08, .04)]


SPEC = {
    "version": "revision-v2-1d-information-design-20261021",
    "claim": "thermal information design and conventional scalar/three-knot benchmarks; no 1D PINN retraining",
    "config": asdict(CONFIG),
    "candidates": candidate_designs(),
    "development_multipliers": _DEVELOPMENT_MULTIPLIERS,
    "development_target_exclusion": "Evaluation matched/cubic truths are never arguments to candidate selection.",
    "known_inputs": "nominal alpha(T), k(T), geometry, face capacities/conductances, initial states and controls",
    "unknown": "rho(T), either one common log multiplier or three bounded knot log multipliers",
    "channels": ("cold_face_temperature", "hot_face_temperature", "voltage"),
    "candidate_controls": "signed +/- constant current for each initial/reservoir cold-hot window; no new sensor channels",
    "selection_order": "Among passing candidates: experiment count, observation count, current magnitude, worst contrast SE, negative minimum singular value, candidate id.",
    "information_gate": "At both fixed development curves: SE(contrast)<=.01; min singular value>=100; all marginal log SE<=.02; temperatures stay285..315K and cover below288/above312K.",
    "nonlinear_gate": "Each contrast +/- .035 perturbation along the locally least-informative contrast direction must give whitened observation distance>=3; remains within frozen log bounds.",
    "selected_refinement": "Recheck selected Jacobians at log step .0003 (relative Frobenius discrepancy<=.001), and conventional5cell/dt.0015 versus10cell/dt.00075 observable discrepancy<=.1sigma. Failure is unresolved, without selecting another candidate.",
    "positive_control": {"kind": "matched PWL constitutive basis, independent nodal numerical truth", "multipliers": (1., 1.035, 1.)},
    "mismatch_control": {"kind": "new cubic matching positive-control knots plus .02e-5*z*(z*z-1)", "reference_temperature": 300., "temperature_scale": 15., "coefficients": (1.035e-5, -.065e-5, -.030e-5, .020e-5)},
    "evaluation": "3 paired fresh-noise datasets per truth family; both scalar and three-knot classical fits, failures retained; no unique data-effective cubic truth claimed",
    "scalar_fit": {"log_bounds": CONFIG.log_parameter_bounds, "maximum_iterations": 48, "x_tolerance": 1e-8},
    "three_knot_fit": {"initial_log_multipliers": (0.,0.,0.), "log_bounds": CONFIG.log_parameter_bounds, "coordinate_passes": 1, "golden_section_iterations": 6, "gauss_newton_iterations": 6, "smoothness_weight": 0., "shrinkage_weight": 0.},
    "truth_refinement": "All selected regimes and both truth families:25nodes/dt.00025 versus49nodes/dt.0000625, max differences<=.001K and1e-6V on21times and9positions; all coarse/fine truth temperatures remain285..315K. Numerical exception or failed check blocks all6trials before fitting.",
    "matched_recovery_gates": "Three-knot fit: data MSE<=2, max absolute multiplier error<=.02, absolute signed center-contrast error<=.01. Scalar reports fit/prediction; scalar is not a curve-shape recovery method.",
    "mismatch_metrics": "Data likelihood, actual cubic dense-curve errors and projection floors, knot diagnostics labeled as mismatch diagnostics; no exact cubic-knots recovery gate.",
}


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "_asdict"):
        return _jsonable(value._asdict())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def _canonical(value):
    return json.dumps(_jsonable(value),sort_keys=True,separators=(",",":"))


def _write(path,value):
    path.write_text(json.dumps(_jsonable(value),indent=2,allow_nan=False)+"\n")


def _source_hashes():
    root=Path(__file__).resolve().parents[2]
    names=("thermotwin/studies/pinn_spatial_revision_1d.py", "thermotwin/simulation/distributed.py", "thermotwin/simulation/distributed_independent.py", "thermotwin/physics/distributed.py", "thermotwin/observations/distributed.py", "thermotwin/inference/distributed_properties.py", "thermotwin/core/controls.py", "thermotwin/numerics/integration.py", "thermotwin/numerics/matrices.py", "thermotwin/inference/distributed_regularization.py")
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def design_experiments(candidate):
    base=distributed_inverse_constant_experiments()[1]
    return tuple(replace(base,initial_cold_face_temperature=low,initial_hot_face_temperature=high,
        cold_reservoir_temperature=low,hot_reservoir_temperature=high,duration=CONFIG.duration,current=sign*candidate["current_magnitude"])
        for low,high in candidate["windows"] for sign in (1.,-1.))


@lru_cache(maxsize=96)
def _trajectory(experiment,values):
    baseline=experiment.material.electrical_resistivity
    changed=replace(experiment,material=replace(experiment.material,electrical_resistivity=baseline.with_values(values)))
    return changed,run_distributed_leg_experiment(changed)


def _forward(experiments,theta,interval):
    predictions, rows, ranges = [], [], []
    for index,experiment in enumerate(experiments):
        baseline=experiment.material.electrical_resistivity
        prop_values=tuple(v*math.exp(float(t)) for v,t in zip(baseline.values,theta))
        changed,result=_trajectory(experiment,prop_values)
        dataset=observe_distributed_trajectory(changed,result.trajectory,
            times=regular_distributed_observation_times(experiment.duration,interval))
        for item in dataset.observations:
            predictions.append(item.value)
            rows.append({"experiment":index,"time":item.time,"channel":item.channel,
                "sigma":CONFIG.temperature_sigma if item.channel.endswith("temperature") else CONFIG.voltage_sigma})
        all_t=(*result.trajectory.cold_face,*result.trajectory.hot_face,
               *(v for state in result.trajectory.cells for v in state))
        ranges.append((min(all_t),max(all_t)))
    return np.asarray(predictions),rows,ranges


def noise_whitened_jacobian(experiments,log_multipliers,observation_interval,step=1e-3):
    theta=np.asarray(log_multipliers,dtype=float)
    if theta.shape!=(3,) or not np.isfinite(theta).all() or not math.isfinite(step) or step<=0:
        raise ValueError("three finite log multipliers and a positive finite difference step required")
    predicted,rows,_=_forward(tuple(experiments),theta,observation_interval)
    sigma=np.asarray([r["sigma"] for r in rows])
    jac=np.column_stack([(_forward(tuple(experiments),theta+np.eye(3)[j]*step,observation_interval)[0]
        -_forward(tuple(experiments),theta-np.eye(3)[j]*step,observation_interval)[0])/(2*step)/sigma for j in range(3)])
    return jac,rows,predicted


def contrast_information(jacobian,multipliers):
    jac=np.asarray(jacobian,dtype=float)
    m=np.asarray(multipliers,dtype=float)
    if jac.ndim!=2 or jac.shape[1]!=3 or m.shape!=(3,) or not np.isfinite(jac).all() or not np.isfinite(m).all() or np.min(m)<=0:
        raise ValueError("finite N-by-3 Jacobian and three positive multipliers required")
    _,s,vt=np.linalg.svd(jac,full_matrices=False)
    full=bool(len(s)==3 and s[-1]>max(1e-12,s[0]*1e-10))
    weak=vt[-1].copy()
    if weak[np.argmax(abs(weak))]<0:
        weak*=-1
    if full:
        covariance=vt.T@np.diag(1/s**2)@vt
        gradient=np.array([-.5*m[0],m[1],-.5*m[2]])
        variance=float(gradient@covariance@gradient)
        se=math.sqrt(variance)
        marginal=np.sqrt(np.diag(covariance))
        direction=covariance@gradient/variance
    else:
        covariance=None; se=math.inf; marginal=np.full(3,math.inf); direction=None
    return {"singular_values":s.tolist(),"full_rank":full,"condition_number":float(s[0]/s[-1]) if full else math.inf,
        "contrast_standard_error":se,"max_marginal_log_standard_error":float(max(marginal)),
        "marginal_log_standard_errors":marginal.tolist(),"covariance":None if covariance is None else covariance.tolist(),
        "weakest_direction":weak.tolist(),"least_informative_contrast_direction":None if direction is None else direction.tolist(),
        "minimum_singular_value_per_sqrt_observation":float(s[-1]/math.sqrt(len(jac))) if len(s)==3 else 0.}


def _contrast(m):
    return float(m[1]-.5*(m[0]+m[2]))


def _nonlinear_check(experiments,theta,interval,information):
    if not information["full_rank"]:
        return {"passed":False,"reason":"rank_deficient"}
    reference,rows,_=_forward(experiments,theta,interval)
    sigma=np.asarray([r["sigma"] for r in rows])
    direction=np.asarray(information["least_informative_contrast_direction"])
    m=np.exp(theta); checks=[]
    for sign in (-1.,1.):
        delta=sign*CONFIG.nonlinear_contrast_change*direction
        changed=theta+delta
        valid=bool(np.min(changed)>=CONFIG.log_parameter_bounds[0] and np.max(changed)<=CONFIG.log_parameter_bounds[1])
        if valid:
            prediction,_,_=_forward(experiments,changed,interval)
            norm=float(np.linalg.norm((prediction-reference)/sigma))
            actual_change=_contrast(np.exp(changed))-_contrast(m)
            sign_correct=bool(actual_change*sign>0)
            acceptable=bool(norm>=CONFIG.minimum_nonlinear_whitened_distance and sign_correct and abs(actual_change)>=.9*CONFIG.nonlinear_contrast_change)
            checks.append({"sign":sign,"requested_linear_contrast_change":sign*CONFIG.nonlinear_contrast_change,
                "actual_nonlinear_contrast_change":actual_change,"whitened_observation_distance":norm,
                "log_multipliers":changed.tolist(),"within_parameter_bounds":True,"passed":acceptable})
        else:
            checks.append({"sign":sign,"within_parameter_bounds":False,"passed":False})
    return {"passed":all(c["passed"] for c in checks),"checks":checks,
        "interpretation":"Nonlinear forward check along locally least-informative contrast direction, not a nonlinear confidence interval."}


def _conventional_refinement(experiments,theta,interval):
    coarse,rows,_=_forward(experiments,theta,interval)
    refined=tuple(replace(e,cell_count=2*e.cell_count,time_step=e.time_step/2,initial_cell_temperatures=None) for e in experiments)
    fine,_,_=_forward(refined,theta,interval)
    sigma=np.asarray([r["sigma"] for r in rows])
    maximum=float(max(abs((coarse-fine)/sigma)))
    return {"maximum_observable_difference_sigma":maximum,"passed":maximum<=.1,
        "coarse_cells":experiments[0].cell_count,"fine_cells":refined[0].cell_count,
        "coarse_time_step":experiments[0].time_step,"fine_time_step":refined[0].time_step}


def run_development(output_directory):
    output=Path(output_directory); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    # This file exists before the first forward solve or candidate score.
    manifest={"status":"started","created_utc":datetime.now(timezone.utc).isoformat(),"spec":SPEC,
        "spec_sha256":hashlib.sha256(_canonical(SPEC).encode()).hexdigest(),"source_sha256":_source_hashes(),
        "python":sys.version,"executable":sys.executable}
    _write(output/"predeclared-development-manifest.json",manifest)
    results=[]
    for candidate in candidate_designs():
        print("1d development",candidate["id"],flush=True)
        experiments=design_experiments(candidate); curves=[]
        for multipliers in _DEVELOPMENT_MULTIPLIERS:
            theta=np.log(multipliers)
            jac,rows,_=noise_whitened_jacobian(experiments,theta,candidate["observation_interval"],CONFIG.finite_difference_log_step)
            info=contrast_information(jac,multipliers)
            _,_,ranges=_forward(experiments,theta,candidate["observation_interval"])
            tmin=min(r[0] for r in ranges); tmax=max(r[1] for r in ranges)
            numerical=bool(CONFIG.physical_temperature_bounds[0]<=tmin and tmax<=CONFIG.physical_temperature_bounds[1])
            coverage=bool(tmin<=CONFIG.required_low_temperature and tmax>=CONFIG.required_high_temperature)
            information_pass=bool(info["full_rank"] and min(info["singular_values"])>=CONFIG.minimum_singular_value
                and info["contrast_standard_error"]<=CONFIG.contrast_standard_error_limit
                and info["max_marginal_log_standard_error"]<=CONFIG.maximum_marginal_log_standard_error)
            nonlinear=_nonlinear_check(experiments,theta,candidate["observation_interval"],info) if information_pass else {"passed":False,"reason":"information_gate_failed"}
            curves.append({"development_multipliers":multipliers,"information":info,"temperature_ranges_K":ranges,
                "physical_domain_passed":numerical,"temperature_coverage_passed":coverage,"information_passed":information_pass,
                "nonlinear_contrast_check":nonlinear,"passed":bool(numerical and coverage and information_pass and nonlinear["passed"])})
        row={"candidate":candidate,"curves":curves,"passed":all(c["passed"] for c in curves),
            "worst_contrast_standard_error":max(c["information"]["contrast_standard_error"] for c in curves),
            "worst_max_marginal_log_standard_error":max(c["information"]["max_marginal_log_standard_error"] for c in curves),
            "worst_minimum_singular_value":min(min(c["information"]["singular_values"]) for c in curves)}
        results.append(row); _write(output/"candidate-results.json",results)
    passing=[r for r in results if r["passed"]]
    choice=min(passing,key=lambda r:(r["candidate"]["experiment_count"],r["candidate"]["observation_count"],r["candidate"]["current_magnitude"],r["worst_contrast_standard_error"],-r["worst_minimum_singular_value"],r["candidate"]["id"])) if passing else None
    refinement=[]
    if choice:
        experiments=design_experiments(choice["candidate"])
        for multipliers in _DEVELOPMENT_MULTIPLIERS:
            theta=np.log(multipliers)
            coarse,_,_=noise_whitened_jacobian(experiments,theta,choice["candidate"]["observation_interval"],.001)
            fine,_,_=noise_whitened_jacobian(experiments,theta,choice["candidate"]["observation_interval"],.0003)
            discrepancy=float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine))
            forward_refinement=_conventional_refinement(experiments,theta,choice["candidate"]["observation_interval"])
            refinement.append({"development_multipliers":multipliers,"relative_jacobian_discrepancy":discrepancy,
                "conventional_refinement":forward_refinement,"passed":discrepancy<=.001 and forward_refinement["passed"]})
    selected=bool(choice and all(r["passed"] for r in refinement))
    selection={"status":"selected" if selected else "unresolved","spec_sha256":manifest["spec_sha256"],
        "source_sha256":manifest["source_sha256"],"candidate_count":len(results),"passing_count":len(passing),
        "candidate":None if choice is None else choice["candidate"],"candidate_score":choice,
        "selected_fd_refinement":refinement,"noise_assumptions":{"temperature_sigma_K":CONFIG.temperature_sigma,"voltage_sigma_V":CONFIG.voltage_sigma},
        "elapsed_seconds":time.monotonic()-started,"named_data_generated":False,"evaluation_truths_used_for_selection":False}
    _write(output/"selection.json",selection)
    manifest["status"]="completed"; manifest["source_unchanged"]=(manifest["source_sha256"]==_source_hashes())
    _write(output/"development-manifest.json",manifest)
    return selection


def _with_property(experiment,prop):
    return replace(experiment,material=replace(experiment.material,electrical_resistivity=prop))


def _refinement(experiments,results):
    rows=[]
    for experiment,coarse in zip(experiments,results):
        fine=run_independent_distributed_experiment(experiment,node_count=49,time_step=.0000625)
        temperatures=[]; voltages=[]
        for time_value in np.linspace(0,CONFIG.duration,21):
            c=interpolate_independent_temperature(coarse.trajectory,float(time_value))
            f=interpolate_independent_temperature(fine.trajectory,float(time_value))
            for x in np.linspace(0,experiment.geometry.length,9):
                temperatures.append(interpolate_independent_position(coarse.trajectory.positions,c,float(x))-interpolate_independent_position(fine.trajectory.positions,f,float(x)))
            voltages.append(independent_distributed_voltage(experiment,c,current=float(experiment.current))-independent_distributed_voltage(experiment,f,current=float(experiment.current)))
        coarse_range=(min(min(s) for s in coarse.trajectory.temperature),max(max(s) for s in coarse.trajectory.temperature))
        fine_range=(min(min(s) for s in fine.trajectory.temperature),max(max(s) for s in fine.trajectory.temperature))
        domain=bool(all(CONFIG.physical_temperature_bounds[0]<=low and high<=CONFIG.physical_temperature_bounds[1] for low,high in (coarse_range,fine_range)))
        row={"maximum_temperature_difference_K":max(map(abs,temperatures)),"maximum_voltage_difference_V":max(map(abs,voltages)),
            "coarse_temperature_range_K":coarse_range,"fine_temperature_range_K":fine_range,"physical_domain_passed":domain}
        row["passed"]=domain and row["maximum_temperature_difference_K"]<=.001 and row["maximum_voltage_difference_V"]<=1e-6
        rows.append(row)
    return {"passed":all(r["passed"] for r in rows),"regimes":rows}


def _curve_metrics(prop,truth):
    temperatures=np.linspace(285,315,601)
    target=np.asarray([truth.value(float(t)) for t in temperatures])
    estimated=np.asarray([prop.value(float(t)) for t in temperatures])
    relative=(estimated-target)/target
    basis=np.asarray([[prop.with_values(tuple(float(i==j) for j in range(3))).value(float(t)) for i in range(3)] for t in temperatures])
    projection,*_=np.linalg.lstsq(basis/target[:,None],np.ones(len(target)),rcond=None)
    projected=basis@projection
    return {"relative_rmse":float(np.sqrt(np.mean(relative**2))),"maximum_relative_error":float(max(abs(relative))),
        "projection_relative_rmse":float(np.sqrt(np.mean(((projected-target)/target)**2))),"projection_values":projection.tolist(),
        "projection_is_data_effective_truth":False}


def _seed(namespace,trial,experiment):
    digest=hashlib.sha256(f"{namespace}\0trial={trial}\0experiment={experiment}".encode()).digest()
    return 1_000_000+int.from_bytes(digest[:8],"big")%(2**31-1_000_001)


def _blocked_evaluation(output,selection,namespace,reason,error=None):
    result={"status":"blocked","blocked_reason":reason,"numerical_reference_passed":False,
        "planned_trials":6,"completed_trials":0,"not_run_count":6,
        "trials":[{"family":family,"index":index,"completed":False,"not_run_reason":reason}
                  for family in ("matched_pwl","cubic_mismatch") for index in range(CONFIG.trial_count_per_truth)]}
    if error is not None:
        result["numerical_failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
    _write(output/"result.json",result)
    _write(output/"completed-manifest.json",{"status":"blocked","blocked_reason":reason,
        "namespace":namespace,"source_sha256":_source_hashes(),"selection":selection})
    return result


def run_evaluation(output_directory,selection,namespace):
    """Generate and fit future named data only after the caller's source gate."""
    from scipy.optimize import minimize_scalar
    import scipy
    if isinstance(selection,(str,Path)):
        selection=json.loads(Path(selection).read_text())
    if not isinstance(namespace,str) or not namespace.strip():
        raise ValueError("a fresh caller-declared evaluation namespace is required")
    if selection.get("status")!="selected" or selection.get("spec_sha256")!=hashlib.sha256(_canonical(SPEC).encode()).hexdigest():
        raise ValueError("a selected development result for this frozen SPEC is required")
    if selection.get("source_sha256")!=_source_hashes() or selection.get("candidate") not in candidate_designs():
        raise ValueError("selected source or candidate changed before evaluation")
    candidate=selection["candidate"]; baselines=design_experiments(candidate)
    seed_grid=[[_seed(namespace,i,j) for j in range(len(baselines))] for i in range(CONFIG.trial_count_per_truth)]
    if len(set(seed for row in seed_grid for seed in row))!=CONFIG.trial_count_per_truth*len(baselines):
        raise ValueError("distinct trial/experiment streams collide before data generation")
    output=Path(output_directory); output.mkdir(parents=True,exist_ok=False)
    _write(output/"manifest.json",{"status":"started","spec":SPEC,"selection":selection,"namespace":namespace,
        "source_sha256":_source_hashes(),"paired_noise_across_truth_families":True,"seed_grid":seed_grid,
        "runtime":{"python":sys.version,"executable":sys.executable,"numpy":np.__version__,"scipy":scipy.__version__,
                   "thread_environment":{name:os.environ.get(name) for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS")}}})
    baseline=baselines[0].material.electrical_resistivity
    matched=baseline.with_values(tuple(v*m for v,m in zip(baseline.values,SPEC["positive_control"]["multipliers"])))
    cubic=PolynomialTemperatureProperty(300.,15.,SPEC["mismatch_control"]["coefficients"])
    families={"matched_pwl":matched,"cubic_mismatch":cubic}
    fixtures={}; refinements={}
    try:
        for name,truth in families.items():
            experiments=tuple(_with_property(e,truth) for e in baselines)
            results=tuple(run_independent_distributed_experiment(e,node_count=CONFIG.truth_node_count,time_step=CONFIG.truth_time_step) for e in experiments)
            refinements[name]=_refinement(experiments,results)
            clean=tuple(observe_independent_distributed_result(e,r,observation_interval=candidate["observation_interval"]) for e,r in zip(experiments,results))
            fixtures[name]=(experiments,clean)
            _write(output/f"clean-{name}.json",{"datasets":clean,"truth_values_at_knots":[truth.value(t) for t in baseline.temperatures]})
    except (RuntimeError,ValueError,FloatingPointError,OverflowError) as error:
        _write(output/"truth-refinement.json",refinements)
        return _blocked_evaluation(output,selection,namespace,"independent_truth_numerical_failure",error)
    _write(output/"truth-refinement.json",refinements)
    if not all(r["passed"] for r in refinements.values()):
        return _blocked_evaluation(output,selection,namespace,"independent_truth_refinement_failed")
    trials=[]
    scales={"cold_face_temperature":CONFIG.temperature_sigma,"hot_face_temperature":CONFIG.temperature_sigma,"voltage":CONFIG.voltage_sigma}
    for family,truth in families.items():
        for index in range(CONFIG.trial_count_per_truth):
            directory=output/f"{family}-trial-{index:02d}"; directory.mkdir()
            seeds=seed_grid[index]
            observations=tuple(add_distributed_gaussian_noise(clean,standard_deviations=scales,seed=seed) for clean,seed in zip(fixtures[family][1],seeds))
            _write(directory/"observations.json",{"seeds":seeds,"datasets":observations})
            target=np.asarray([v for dataset in observations for v in dataset.values()])
            _,rows,_=_forward(baselines,np.zeros(3),candidate["observation_interval"])
            sigma=np.asarray([r["sigma"] for r in rows])
            trial={"family":family,"index":index,"methods":{},"completed":False}
            for method in ("scalar","three_knot"):
                stage="fit"
                try:
                    if method=="scalar":
                        evaluations=[]
                        def objective(value):
                            prediction=_forward(baselines,np.full(3,value),candidate["observation_interval"])[0]
                            mse=float(np.mean(((prediction-target)/sigma)**2)); evaluations.append({"log_multiplier":float(value),"mse":mse}); return mse
                        fit=minimize_scalar(objective,method="bounded",bounds=CONFIG.log_parameter_bounds,options={"maxiter":48,"xatol":1e-8})
                        theta=np.full(3,fit.x); fit_mse=float(fit.fun); converged=bool(fit.success)
                        raw={"evaluations":evaluations,"optimizer_success":converged,"message":str(fit.message)}
                    else:
                        fit=fit_distributed_property(baselines,observations,DistributedPropertyFitConfig(
                            property_name="electrical_resistivity",observation_interval=candidate["observation_interval"],
                            initial_log_multipliers=(0.,0.,0.),log_multiplier_bounds=CONFIG.log_parameter_bounds,
                            coordinate_passes=1,golden_section_iterations=6,gauss_newton_iterations=6,
                            temperature_standard_deviation=CONFIG.temperature_sigma,voltage_standard_deviation=CONFIG.voltage_sigma,
                            smoothness_weight=0.,shrinkage_weight=0.))
                        theta=np.asarray(fit.log_multipliers); fit_mse=fit.mean_normalized_squared_error; converged=True
                        raw={"evaluations":fit.evaluations,"fixed_budget_returned":True}
                    trial["methods"][method]={"completed":False,"fit_completed":True,"log_multipliers":theta.tolist(),"mse":fit_mse,"raw_fit":raw}
                    _write(directory/f"{method}-raw-fit.json",trial["methods"][method])
                    stage="diagnostics"
                    multipliers=np.exp(theta); inferred=baseline.with_values(np.asarray(baseline.values)*multipliers)
                    jac,_,_=noise_whitened_jacobian(baselines,theta,candidate["observation_interval"])
                    information=contrast_information(jac,multipliers)
                    level_derivative=jac@np.ones(3)
                    fit_covariance=[[float(1/(level_derivative@level_derivative))]] if method=="scalar" else information["covariance"]
                    actual_knot=np.asarray([truth.value(t)/v for t,v in zip(baseline.temperatures,baseline.values)])
                    knot_error=float(max(abs(multipliers-actual_knot))); contrast_error=abs(_contrast(multipliers)-_contrast(actual_knot))
                    matched_pass=bool(method=="three_knot" and family=="matched_pwl" and fit_mse<=CONFIG.maximum_observation_mse
                        and knot_error<=CONFIG.maximum_knot_multiplier_error and contrast_error<=CONFIG.maximum_contrast_error)
                    trial["methods"][method].update({"completed":True,"multipliers":multipliers.tolist(),"mse":fit_mse,
                        "optimizer_convergence_claimed":converged if method=="scalar" else None,"raw_fit":raw,"information":information,
                        "information_parameterization":"three hypothetical free log knot coefficients, including for scalar estimates",
                        "fit_covariance_log_parameters":fit_covariance,"fit_free_parameter_count":1 if method=="scalar" else 3,
                        "curve":_curve_metrics(inferred,truth),"maximum_knot_multiplier_error":knot_error,"signed_center_contrast":_contrast(multipliers),
                        "actual_knot_contrast_error":contrast_error,"matched_curve_recovery_passed":matched_pass if family=="matched_pwl" and method=="three_knot" else None,
                        "interpretation":"cubic knot diagnostics do not define exact data-effective PWL truth" if family=="cubic_mismatch" else "matched constitutive positive control"})
                except (RuntimeError,ValueError,FloatingPointError,OverflowError) as error:
                    trial["methods"].setdefault(method,{})
                    trial["methods"][method].update({"completed":False,"failure_stage":stage,"failure":str(error),"traceback":traceback.format_exc()})
            trial["completed"]=all(m["completed"] for m in trial["methods"].values())
            trials.append(trial); _write(directory/"result.json",trial)
            print("1d evaluation",family,index,"completed",trial["completed"],flush=True)
    result={"status":"completed","numerical_reference_passed":True,"planned_trials":6,"completed_trials":sum(t["completed"] for t in trials),
        "matched_three_knot_success_count":sum(t["methods"]["three_knot"].get("matched_curve_recovery_passed",False) for t in trials if t["family"]=="matched_pwl"),
        "matched_denominator":3,"cubic_denominator":3,"trials":trials,"source_unchanged":selection["source_sha256"]==_source_hashes(),
        "claim":"information design plus conventional benchmarks; no 1D PINN training or superiority claim"}
    _write(output/"result.json",result)
    _write(output/"completed-manifest.json",{"status":"completed","namespace":namespace,"source_sha256":_source_hashes(),"selection":selection})
    return result

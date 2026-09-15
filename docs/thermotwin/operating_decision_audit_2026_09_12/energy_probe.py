"""Read-only audit of nominal versus per-device diagnostic electrical energy."""
import json
from dataclasses import replace
from statistics import fmean
from thermotwin.studies import operating_decision_realism as r
from thermotwin.studies.operating_decision import default_fixed_policies, VERIFICATION, OperatingRegime, initial_acquisition_regime
from thermotwin.simulation.four_node_experiments import constant_current_contact_reference_experiment
from thermotwin.design.control_comparison import piecewise_electrical_energy

config = r.OperatingDecisionRealismConfig()
reference = constant_current_contact_reference_experiment()
rows = []
for family in r.STAGE3_TRUTH_CONDITIONS:
    for trial in range(config.sensor.trial_count):
        truth = r.realism_truth_for_trial(config, trial)
        effective = replace(reference.thermoelectric_parameters, electrical_resistance=reference.thermoelectric_parameters.electrical_resistance+truth.series_resistance)
        for policy in default_fixed_policies():
            regimes = (initial_acquisition_regime(), *policy.additional_regimes, OperatingRegime('fixed_verification', VERIFICATION, config.verification_current, (r.COLD_EXCHANGER, r.HOT_EXCHANGER)))
            actual = nominal = 0.0
            for regime in regimes:
                loaded = (policy.name == r.FIXED_FACE_TEMPERATURE and regime.name != 'initial_0.8A_20s')
                prediction = r._truth_prediction(family, truth, regime, r.RunInstrumentation(loaded), config)
                actual += piecewise_electrical_energy(prediction.time, prediction.cold_face, prediction.hot_face, effective, regime.current, start_time=prediction.time[0], end_time=prediction.time[-1])
                nominal += r.nominal_realistic_schedule_energy(regime.current, config, temporary_face_sensor=loaded)
            rows.append(dict(family=family, trial_index=trial, policy=policy.name, series_resistance=truth.series_resistance, recorded_nominal_j=nominal, actual_simulated_j=actual, error_percent=100*(nominal-actual)/actual))
summary = [dict(family=family, policy=policy.name, nominal_j=next(x['recorded_nominal_j'] for x in rows if x['family']==family and x['policy']==policy.name), actual_mean_j=fmean(x['actual_simulated_j'] for x in rows if x['family']==family and x['policy']==policy.name), min_error_percent=min(x['error_percent'] for x in rows if x['family']==family and x['policy']==policy.name), max_error_percent=max(x['error_percent'] for x in rows if x['family']==family and x['policy']==policy.name)) for family in r.STAGE3_TRUTH_CONDITIONS for policy in default_fixed_policies()]
print(json.dumps(dict(config_first_seed=config.sensor.first_seed, blocks=config.sensor.trial_count, summary=summary, rows=rows), indent=2))

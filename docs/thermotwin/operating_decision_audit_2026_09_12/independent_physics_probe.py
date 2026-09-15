"""Independent physics/RK5 check. Does not use any project RHS or integrator.

Only public configuration inputs and the production observable generator are
imported. No evaluation artifacts, decisions or labels are accessed.
Run with a Python >=3.10 interpreter from the repository root.
"""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from thermotwin.studies import operating_decision_realism as r
from thermotwin.studies.operating_decision import operating_margin
from thermotwin.inference.sparse_sensors import sparse_training_current

# Dormand-Prince fifth-order tableau; project uses classical RK4.
B = ((), (1/5,), (3/40, 9/40), (44/45, -56/15, 32/9),
     (19372/6561, -25360/2187, 64448/6561, -212/729),
     (9017/3168, -355/33, 46732/5247, 49/176, -5103/18656))
W = (35/384, 0, 500/1113, 125/192, -2187/6784, 11/84)


def independent_simulation(model, current, physical, mass, rs, probe, beta, h=.025):
    rc, cf, lag = physical
    five = model != r.FOUR_STATE_MODEL
    caps = [cf, 100., 50., 100.] + ([mass] if five else [])
    probe_index = len(caps) if probe else None
    if probe:
        caps.append(probe.thermal_capacitance)
    nphys = len(caps)
    # Physical states ordered face-cold, face-hot, exchanger-cold,
    # exchanger-hot, optional interface, optional probe; then sensor lags.
    y = [300.] * (nphys + 2)
    max_energy_residual = 0.
    min_entropy = float('inf')

    def rhs(y, i):
        nonlocal max_energy_residual, min_entropy
        tc, th, xc, xh = y[:4]
        heat = [0.] * nphys
        q_joule = i * i * (2. + rs)
        # Peltier and passive thermoelectric conduction written directly.
        heat[0] += -.05*i*tc + q_joule/2 + .5*(th-tc)
        heat[1] += .05*i*th + q_joule/2 - .5*(th-tc)
        heat[2] += 2.*(300.-xc)
        heat[3] += 4.*(300.-xh)
        edges = [(1, 3, 4.)]
        if five:
            effective_rc = rc*math.exp((beta or 0.)*(y[4]-300.))
            edges += [(2,4,2./effective_rc), (4,0,2./effective_rc)]
        else:
            edges += [(2,0,1./rc)]
        if probe:
            edges += [(0,probe_index,probe.thermal_capacitance/probe.response_time_constant)]
        entropy = q_joule/2*(1/tc+1/th) + .5*(th-tc)**2/(tc*th)
        for a,b,g in edges:
            flux = g*(y[a]-y[b])
            heat[a] -= flux
            heat[b] += flux
            entropy += g*(y[a]-y[b])**2/(y[a]*y[b])
        entropy += 2.*(300.-xc)**2/(300.*xc)+4.*(300.-xh)**2/(300.*xh)
        min_entropy = min(min_entropy, entropy)
        energy_residual = sum(heat)-(.05*i*(th-tc)+q_joule+2.*(300.-xc)+4.*(300.-xh))
        max_energy_residual = max(max_energy_residual, abs(energy_residual))
        return [v/c for v,c in zip(heat,caps)]+[(xc-y[-2])/lag, (xh-y[-1])/lag]

    states = {0.: list(y)}
    switches = [0.] + list(current.transition_times) + [80.]
    for left, right in zip(switches, switches[1:]):
        if right <= left:
            continue
        i = current.value_at((left+right)/2)
        steps = int(round((right-left)/h))
        step = (right-left)/steps
        for j in range(steps):
            k = []
            for coeff in B:
                interim = [v+step*sum(c*prior[d] for c,prior in zip(coeff,k)) for d,v in enumerate(y)]
                k.append(rhs(interim, i))
            y = [v+step*sum(w*kk[d] for w,kk in zip(W,k)) for d,v in enumerate(y)]
            states[round(left+(j+1)*step,9)] = list(y)
    return states, max_energy_residual, min_entropy


def compare(label, model, current, physical, mass, rs, probe, beta, config, h=.025):
    channels = (r.COLD_EXCHANGER,r.HOT_EXCHANGER,r.VOLTAGE) + ((r.COLD_FACE,) if probe else ())
    prod = r._simulate_realistic_observables(model,current,channels,physical,mass,rs,probe,config,contact_beta=beta)
    exact, balance, entropy = independent_simulation(model,current,physical,mass,rs,probe,beta,h)
    face_error = max(abs(tc-exact[round(t,9)][0]) for t,tc in zip(prod.time,prod.cold_face))
    hot_error = max(abs(th-exact[round(t,9)][1]) for t,th in zip(prod.time,prod.hot_face))
    obs_err = {ch:0. for ch in channels}
    for val in prod.values:
        y = exact[round(val.time,9)]
        if val.channel == r.COLD_EXCHANGER:
            ref = y[-2]
        elif val.channel == r.HOT_EXCHANGER:
            ref = y[-1]
        elif val.channel == r.VOLTAGE:
            ref = .05*(y[1]-y[0])+current.value_at(val.time)*(2+rs)
        else:
            ref = y[-3]
        obs_err[val.channel] = max(obs_err[val.channel],abs(ref-val.value))
    pm = operating_margin(prod.cold_face,config.band)
    rm = min(min(y[0]-config.band.lower_temperature,config.band.upper_temperature-y[0]) for y in exact.values())
    coarse_exact = min(min(exact[round(t,9)][0]-config.band.lower_temperature,config.band.upper_temperature-exact[round(t,9)][0]) for t in prod.time)
    return dict(label=label,model=model,instrumented=bool(probe),step=h,
                max_cold_face_error_K=face_error,max_hot_face_error_K=hot_error,
                observation_max_errors=obs_err,margin_production_K=pm,margin_reference_K=rm,
                margin_difference_K=pm-rm,margin_sampling_gap_K=coarse_exact-rm,
                max_energy_balance_residual_W=balance,min_entropy_production_W_K=entropy)


def main():
    config = r.OperatingDecisionRealismConfig()
    result = []
    for model in (r.FOUR_STATE_MODEL,r.FIVE_STATE_MODEL,r.TEMPERATURE_DEPENDENT_CONTACT):
        for loaded in (False,True):
            result.append(compare('nominal_final',model,config.final_current,(.25,50.,1.5),20.,.1,
                                  r.TemporaryFaceSensor() if loaded else None,
                                  .12 if model==r.TEMPERATURE_DEPENDENT_CONTACT else None,config))
    for trial in range(10):
        truth = r.realism_truth_for_trial(config,trial)
        for model in (r.FOUR_STATE_MODEL,r.FIVE_STATE_MODEL,r.TEMPERATURE_DEPENDENT_CONTACT):
            result.append(compare('development_final_'+str(trial),model,config.final_current,
                                  truth.physical_values,truth.interface_mass,truth.series_resistance,None,
                                  truth.contact_beta if model==r.TEMPERATURE_DEPENDENT_CONTACT else None,config))
    # Lowest candidate time scales and maximum probe loading.
    result.append(compare('fast_corner_final',r.TEMPERATURE_DEPENDENT_CONTACT,config.final_current,
                          (.08,20.,.15),config.sensor.interface_mass_bounds[0],.5,
                          r.TemporaryFaceSensor(12.,.5),.16,config))
    # Halve independent reference step to check numerical reference reliability.
    result.append(compare('nominal_final_reference_refinement',r.TEMPERATURE_DEPENDENT_CONTACT,
                          config.final_current,(.25,50.,1.5),20.,.1,r.TemporaryFaceSensor(),.12,config,h=.0125))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

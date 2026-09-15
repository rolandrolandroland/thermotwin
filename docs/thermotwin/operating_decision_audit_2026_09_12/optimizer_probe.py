import json,time
from dataclasses import replace
from thermotwin.studies.operating_decision import default_fixed_policies
from thermotwin.studies.operating_decision_realism import (OperatingDecisionRealismConfig,STAGE3_TRUTH_CONDITIONS,build_realistic_blinded_case,fit_realistic_acquisition_models,fit_realistic_candidate,forecast_realistic_margin_interval,realistic_verification_score)
base=OperatingDecisionRealismConfig()
long=replace(base,sensor=replace(base.sensor,fit_iterations=30))
for ti,pi,idx in [(2,3,0),(1,2,0),(0,0,0),(2,0,1),(2,2,1)]:
    case=build_realistic_blinded_case(STAGE3_TRUTH_CONDITIONS[ti],idx,default_fixed_policies()[pi],base)
    fs=fit_realistic_acquisition_models(case,base)
    for f in fs.fits:
        g=fit_realistic_candidate(f.model_name,case.acquisition_runs,long,initial_log_multipliers=f.log_multipliers)
        def interval(f):
            try: return forecast_realistic_margin_interval(f,case.final_regime,base)
            except Exception as e: return str(e)
        print(json.dumps(dict(truth=ti,policy=pi,trial=idx,model=f.model_name,old_objective=f.objective,new_objective=g.objective,old_params=f.log_multipliers,new_params=g.log_multipliers,old_verification=realistic_verification_score(f,case.verification_run,base)[0],new_verification=realistic_verification_score(g,case.verification_run,base)[0],old_interval=interval(f),new_interval=interval(g))),flush=True)

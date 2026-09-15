import math, random
from thermotwin.studies.operating_decision import default_fixed_policies
from thermotwin.studies.operating_decision_realism import OperatingDecisionRealismConfig,build_realistic_blinded_case,realism_truth_for_trial,_truth_prediction,_run_seed
from thermotwin.studies.sensor_model_discrimination import ALL_CHANNELS,COLD_EXCHANGER,HOT_EXCHANGER,COLD_FACE,VOLTAGE
c=OperatingDecisionRealismConfig(); truth='matched_four_state'; policy=default_fixed_policies()[1]
case=build_realistic_blinded_case(truth,0,policy,c)
t=realism_truth_for_trial(c,0)
def errors(run,ch):
 p=_truth_prediction(truth,t,run.regime,run.instrumentation,c)
 pm={(x.channel,x.time):x.value for x in p.values}
 return [x.value-pm[x.channel,x.time] for x in run.observations.values if x.channel==ch]
a=errors(case.acquisition_runs[0],HOT_EXCHANGER)
b=errors(case.acquisition_runs[1],COLD_EXCHANGER)
print('initial hot vs first thermal cold max residual difference K:',max(abs(x-y) for x,y in zip(a,b)))
seed=_run_seed(c,truth,0,'fixed_verification')+100*ALL_CHANNELS.index(HOT_EXCHANGER)
print('family A trial 0 verification hot seed:',seed)
print('trial 1 physical truth seed:',c.sensor.first_seed+10000)
z_bias=random.Random(seed).gauss(0,1)
t1=realism_truth_for_trial(c,1)
z_contact=math.log(t1.physical_values[0]/c.sensor.fit.nominal_values[0])/c.sensor.truth_log_standard_deviations[0]
print('standard normal driving verification hot bias:',z_bias)
print('standard normal driving next device contact resistance:',z_contact)
for ch,run in [(COLD_FACE,'initial_0.8A_20s'),(VOLTAGE,'thermal_0.6A_15s')]:
 print('trial 0 verification',ch,'seed=',_run_seed(c,truth,0,'fixed_verification')+100*ALL_CHANNELS.index(ch),'trial 1',run,'cold seed=',_run_seed(c,truth,1,run))

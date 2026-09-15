import json, math, random
from pathlib import Path
from collections import Counter
from statistics import fmean, NormalDist

root=Path(__file__).resolve().parents[3]
p=json.loads((root/'thermotwin/OPERATING_DECISION_FINAL_RESULT.json').read_text())
a=json.loads((root/'thermotwin/OPERATING_DECISION_FINAL_ARTIFACT.json').read_text())
assert p['artifact']==a
rows=p['outcomes']; assert len(rows)==900
families=('matched_four_state','extra_interface_mass','temperature_dependent_contact')
procedures=('stop_now','fixed_thermal','fixed_voltage','fixed_face_temperature','decision_directed_selector','mismatch_guarded_selector_v2')
# derive actual selector name from file if existing constant includes suffix
procedures=tuple(dict.fromkeys(x['procedure_name'] for x in rows))
print('Procedures:',procedures)
parent=next(x for x in procedures if 'decision_directed' in x)
revised=next(x for x in procedures if 'mismatch_guarded' in x)
indexed={(x['procedure_name'],x['truth_condition'],x['trial_index']):x for x in rows}
assert len(indexed)==900
assert set(indexed)=={(p,f,i) for p in procedures for f in families for i in range(50)}
for family in families:
 for i in range(50):
  matched=[indexed[proc,family,i] for proc in procedures]
  assert len({r['true_margin'] for r in matched})==1
for r in rows:
 assert r['true_pass']==(r['true_margin']>=0)
 assert r['false_approval']==(r['decision']=='approve' and not r['true_pass'])
 assert r['false_rejection']==(r['decision']=='reject' and r['true_pass'])
 for kind in ('raw','calibrated'):
  lo,hi,cov=(r[f'{kind}_interval_{suffix}'] for suffix in ('lower','upper','covered'))
  if lo is None: assert hi is None and cov is None
  else: assert cov==(lo<=r['true_margin']<=hi)
 if r['raw_interval_lower'] is not None:
  q=r['additive_margin_padding']
  assert math.isclose(r['calibrated_interval_lower'],r['raw_interval_lower']-q,abs_tol=1e-12)
  assert math.isclose(r['calibrated_interval_upper'],r['raw_interval_upper']+q,abs_tol=1e-12)
 if r['calibrated_interval_lower'] is None: expected='insufficient_evidence'
 elif r['calibrated_interval_lower']>=0: expected='approve'
 elif r['calibrated_interval_upper']<0: expected='reject'
 else: expected='insufficient_evidence'
 assert expected==r['decision']
print('PASS: all900 identities, paired true margins, pass/error flags, interval coverage/padding and classifications.')
stopenergy=fmean(x['total_diagnostic_energy'] for x in rows if x['procedure_name']=='stop_now')
scenarios=a['analysis']['cost_scenarios']
def loss(x,s):
 return (s['false_approval_weight']*x['false_approval']+s['false_rejection_weight']*x['false_rejection']+s['abstention_weight']*(x['decision']=='insufficient_evidence')+s['added_run_weight']*max(0,x['diagnostic_run_count']-2)+s['added_sensor_weight']*x['extra_sensor_count']+s['incremental_energy_weight']*max(0,(x['total_diagnostic_energy']-stopenergy)/stopenergy))
def close(x,y):
 assert math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-12),(x,y)
def checkrate(saved,n,d,ci):
 assert saved['numerator']==n and saved['denominator']==d
 if not d:
  assert saved['rate'] is None and saved['lower_95'] is None and saved['upper_95'] is None
  return
 close(saved['rate'],n/d)
 if not ci: assert saved['lower_95'] is None and saved['upper_95'] is None; return
 z=NormalDist().inv_cdf(.975); den=1+z*z/d
 center=(n/d+z*z/(2*d))/den
 radius=z*math.sqrt((n/d)*(1-n/d)/d+z*z/(4*d*d))/den
 close(saved['lower_95'],max(0,center-radius));close(saved['upper_95'],min(1,center+radius))
for s in p['summaries']:
 selected=[x for x in rows if x['procedure_name']==s['procedure_name'] and (s['truth_condition']=='all_families' or x['truth_condition']==s['truth_condition'])]
 assert selected, s['truth_condition']
 n=len(selected); decisions=Counter(x['decision'] for x in selected)
 counts=dict(trial_count=n,approvals=decisions['approve'],rejections=decisions['reject'],insufficient_evidence=decisions['insufficient_evidence'],true_passing=sum(x['true_pass'] for x in selected),true_violating=sum(not x['true_pass'] for x in selected),numerical_failures=sum(x['numerical_failure_count']>0 for x in selected))
 for key,value in counts.items(): assert s[key]==value,(key,s[key],value)
 means={'mean_diagnostic_run_count':'diagnostic_run_count','mean_energized_schedule_time_seconds':'energized_schedule_time_seconds','mean_extra_sensor_count':'extra_sensor_count','mean_total_diagnostic_energy':'total_diagnostic_energy','mean_decision_computation_seconds':'decision_computation_seconds'}
 for key,val in means.items():close(s[key],fmean(x[val] for x in selected))
 raw=[x for x in selected if x['raw_interval_covered'] is not None];cal=[x for x in selected if x['calibrated_interval_covered'] is not None]
 blockgroups=[ [x for x in selected if x['trial_index']==i] for i in range(50)]
 rates={'false_approvals':(sum(x['false_approval'] for x in selected),counts['approvals']),'missed_violations':(sum(x['false_approval'] for x in selected),counts['true_violating']),'false_rejections':(sum(x['false_rejection'] for x in selected),counts['rejections']),'decision_coverage':(counts['approvals']+counts['rejections'],n),'abstentions':(counts['insufficient_evidence'],n),'raw_interval_coverage':(sum(x['raw_interval_covered'] for x in raw),len(raw)),'calibrated_interval_coverage':(sum(x['calibrated_interval_covered'] for x in cal),len(cal)),'simultaneous_block_coverage':(sum(all(x['calibrated_interval_covered'] is not False for x in g) for g in blockgroups),50)}
 for key,(a0,b) in rates.items():checkrate(s[key],a0,b,key=='simultaneous_block_coverage' or s['truth_condition']!='all_families')
 assert Counter(x['selected_policy'] for x in selected)==Counter({d['policy']:d['count'] for d in s['selected_policy_counts']})
 for scenario,stored in zip(scenarios,s['expected_losses']):
  assert scenario['name']==stored['scenario_name'];close(stored['mean_loss'],fmean(loss(x,scenario) for x in selected))
print('PASS: all24 summary cells, counts, means, rate/CI arithmetic, action counts, and expected losses agree with independent recomputation.')
for scenario_index,scenario in enumerate(scenarios):
 diffs=[fmean(loss(indexed[revised,f,i],scenario)-loss(indexed[parent,f,i],scenario) for f in families) for i in range(50)]
 rng=random.Random(a['analysis']['paired_block_bootstrap']['seed']+scenario_index*10000)
 draws=sorted(fmean(diffs[rng.randrange(50)] for _ in range(50)) for _ in range(a['analysis']['paired_block_bootstrap']['draws']))
 ci=(draws[math.ceil(.025*len(draws))-1],draws[math.ceil(.975*len(draws))-1])
 stored=p['paired_loss_comparisons'][scenario_index]
 close(stored['revised_minus_parent_mean'],fmean(diffs));close(stored['lower_95'],ci[0]);close(stored['upper_95'],ci[1])
 print('Bootstrap arithmetic matches:',scenario['name'],fmean(diffs),ci)
print('Bootstrap arithmetic matching is NOT validation of its independence/exchangeability assumptions.')
print('\nAll-family comparison (approval/reject/insufficient; error counts; finite interval; block coverage; runs; energy; sensors; losses bench/balanced/expensive):')
for s in p['summaries']:
 if s['truth_condition']=='all_families':
  print(s['procedure_name'],f"{s['approvals']}/{s['rejections']}/{s['insufficient_evidence']}", f"errors={s['false_approvals']['numerator']}/{s['false_rejections']['numerator']}",f"finite={s['calibrated_interval_coverage']['numerator']}/{s['calibrated_interval_coverage']['denominator']}",f"blocks={s['simultaneous_block_coverage']['numerator']}/50", f"runs={s['mean_diagnostic_run_count']:.8f}",f"J={s['mean_total_diagnostic_energy']:.8f}",f"sensors={s['mean_extra_sensor_count']:.8f}", 'losses='+','.join(f"{x['mean_loss']:.8f}" for x in s['expected_losses']))
print('\nFamily-specific raw decision counts:')
for f in families:
 print(f)
 for proc in procedures:
  s=next(x for x in p['summaries'] if x['truth_condition']==f and x['procedure_name']==proc)
  print(proc,f"{s['approvals']}/{s['rejections']}/{s['insufficient_evidence']}",f"errors={s['false_approvals']['numerator']}/{s['false_rejections']['numerator']}",f"truthpasses={s['true_passing']}/50",f"FA95upper={s['false_approvals']['upper_95']}")
print('\nGuard and decision-removal attribution:')
removed=Counter()
for f in families:
 alarms=[indexed[revised,f,i] for i in range(50) if indexed[revised,f,i]['selected_policy']=='early_mismatch_abstention']
 for r in alarms:print('alarm',f,r['trial_index'],'parent',indexed[parent,f,r['trial_index']]['selected_policy'],indexed[parent,f,r['trial_index']]['decision'])
 gs=next(x for x in p['guard_summaries'] if x['truth_condition']==f)
 assert gs['row_count']==50 and gs['triggered_count']==len(alarms)
 assert gs['selected_voltage_count']==sum(indexed[parent,f,r['trial_index']]['selected_policy']=='fixed_voltage' for r in alarms)
 assert gs['triggered_then_parent_insufficient_count']==sum(indexed[parent,f,r['trial_index']]['selected_policy']=='fixed_voltage' and indexed[parent,f,r['trial_index']]['decision']=='insufficient_evidence' for r in alarms)
 for i in range(50):
  old,new=indexed[parent,f,i],indexed[revised,f,i]
  assert new['decision']=='insufficient_evidence' or new['decision']==old['decision']
  if old['decision']!='insufficient_evidence' and new['decision']=='insufficient_evidence':
   removed['guard' if new['selected_policy']=='early_mismatch_abstention' else 'larger_padding']+=1
print('Lost definite decisions:',dict(removed))
print('PASS: all guard counts and no added/reversed decisions.')

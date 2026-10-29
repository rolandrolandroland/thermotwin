// Run a copy from a private build directory linked to the bundled node_modules.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const [root,build,out,skill,python]=process.argv.slice(2).map(p=>path.resolve(p));
const {finalizePresentation,resolvePresentationFont,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont();
const inputs=JSON.parse(await fs.readFile(path.join(root,'thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json')));
const result=JSON.parse(await fs.readFile(path.join(root,'thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json')));
if(inputs.reserved_scientific_result_digest!==result.scientific_result_digest || result.primary_analysis.primary_success!==false)throw Error('Study identity or interpretation differs');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const ink='#172B3A',muted='#526775',teal='#00836C';
const order=['prospective_selector','stop_now','fixed_thermal','fixed_voltage','fixed_face_temperature'];
const labels=['Selector','Stop','Thermal','Voltage','Face'];
function text(s,value,x,y,w,h,size=28,color=ink,bold=false){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=value;a.text.style={fontSize:size,typeface:font,color,bold,autoFit:'none',wrap:true,insets:0};return a;
}
function slide(title,note,footer='Synthetic evidence. No operating procedure met the frozen qualification criteria.'){
 const s=p.slides.add();s.background.fill='#FFFFFF';text(s,title,72,48,1136,100,46,ink,true);
 text(s,footer,72,655,1136,40,18,muted);s.speakerNotes.text=note;return s;
}
function table(s,values,y=170,height=340,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:72,top:y,width:1136,height,values,columnWidths:widths});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#E9EFF2':'#FFFFFF';cell.text.style={fontSize:r===0?23:25,typeface:font,color:ink,bold:r===0};
 }
 t.borders.assign({fill:'#D5DEE3',width:.6,style:'solid'});return t;
}
function chart(s,categories,values,name,ylabel,max,color=teal){
 const c=s.charts.add('bar',{position:{left:72,top:160,width:1136,height:400},categories,
  series:[{name,values,fill:color}],barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:false,
  xAxis:{textStyle:{fontSize:24,typeface:font,fill:ink},majorGridlines:null},
  yAxis:{min:0,max,title:{text:ylabel,textStyle:{fontSize:22,typeface:font}},textStyle:{fontSize:22,typeface:font,fill:muted},majorGridlines:{fill:'#E4E9EC',width:.8}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:24,typeface:font,fill:ink}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
 applyPresentationChartFont(c,{fontFamily:font});return c;
}

let s=p.slides.add();s.background.fill='#F8F9FB';
text(s,'ThermoTwin operating\ndecision experiment',80,92,1120,185,60,ink,true);
text(s,'Final results',80,312,1120,65,38);
text(s,'Lower simulated diagnostic energy\nPrimary decision quality targets were not established',80,420,1120,110,32,muted);
text(s,'Software only study   Roland Bennett   29 October 2026',80,640,1120,35,20,muted);
s.speakerNotes.text='Sources: docs/thermotwin/OPERATING_DECISION_FINAL_REPORT.md and thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json. Numerical execution completed 21 October 2026. Primary success was not established. All results are synthetic.';

s=slide('The next measurement decision','Sources: docs/thermotwin/OPERATING_DECISION_FINAL_REPORT.md, Methods and information boundary. thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RULE.json. All procedures include common initial acquisition and mandatory verification. The margin is the minimum distance to either edge of the full 285.0 to 301.7 K cold-face band.');
text(s,'Choose the next acquisition from initial measurements, then verify the operating margin.',72,153,1136,82,30);
table(s,[['Action','Added runs','Added instrument'],['Stop','0','None'],['Thermal','3','None'],['Voltage','1','Terminal voltage'],['Face temperature','1','Temporary face probe']],260,280,[420,210,506]);
text(s,'Action value is expected margin interval reduction divided by frozen energy, time and instrument costs.',72,565,1136,65,25,muted);

s=slide('Study design and scientific freeze','Sources: docs/thermotwin/OPERATING_DECISION_PHASE_D6_RESULT.md, OPERATING_DECISION_PHASE_E_RESULT.md and OPERATING_DECISION_PHASE_F_RESULT.md. Calibration used rank 96 of 100 block maxima. Nominal balanced equal-price is the only reserved setting. Family C temperature-dependent contact is absent from the candidate menu.');
text(s,'20 tuning blocks and 10 internal checks\n100 independent calibration blocks\n100 untouched reserved blocks',72,165,1136,185,34,ink,true);
text(s,'Each block pairs three truth families. Two candidate models are fitted.\nThe temperature-dependent contact law is absent from the candidate menu.',72,390,1136,105,28);
text(s,'The complete rule, corrections and source were frozen before reserved generation.\n300 cases per procedure are grouped within 100 independent blocks.',72,545,1136,78,25,muted);

s=slide('The reserved decision comparison','Sources: thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json primary_analysis.procedures. All 300 outcomes per procedure are retained. All five procedures had zero false rejections and no pipeline, verification or missing-interval failures. Approval risk is conditional on approval.');
table(s,[['Procedure','Decisions / 300','Abstentions','False approvals / approvals'],...order.map((k,i)=>{let r=result.primary_analysis.procedures[k].overall;return[labels[i],String(r.definitive_decisions),String(r.abstentions),`${r.false_approvals}/${r.approvals}`];})],170,330,[230,255,220,431]);
text(s,'The selector made 54.3% definitive decisions.\nThe frozen usefulness target was at least 70%.',72,535,1136,83,31,ink,true);

s=slide('Lower energy than each measurement policy','Source: thermotwin/OPERATING_DECISION_PROSPECTIVE_PHASE_F_RESULT.json primary_analysis.procedures and comparisons. Paired energy-saving lower bounds were positive against thermal, voltage and face under the declared approximate Student t method. Stop used less energy. These joules exclude computation electricity, resets and sensor electronics.');
chart(s,labels,order.map(k=>Number(result.primary_analysis.procedures[k].overall.realized_resources.total_diagnostic_energy.mean.toFixed(2))),'Diagnostic energy','Mean simulated diagnostic energy J',115);
text(s,'The selector used 75.03 J per case. Stop used the least energy at 60.65 J.\nEnergy savings did not establish the joint primary claim.',72,572,1136,65,27,ink,true);

s=slide('Why the primary criterion was not met','Sources: docs/thermotwin/OPERATING_DECISION_PHASE_E_PROTOCOL.md and OPERATING_DECISION_PHASE_F_RESULT.md. Bounds use the frozen 73 one-sided directions at alpha 0.05/73. Coverage loss margin was 5 percentage points and approval-risk increase margin 2 points. No fixed comparator qualified. Observed 0.96% error is not the 31.85% confidence upper bound.');
table(s,[['Requirement','Observed selector','Required bound','Obtained bound'],['Definitive decisions','54.33%','Lower at least 70%','Lower 38.10%'],['False approvals / approvals','1/104 = 0.96%','Upper at most 10%','Upper 31.85%']],180,230,[330,260,275,271]);
text(s,'No procedure qualified on both decision quality requirements.\nThe required qualifying fixed comparison set was empty.',72,460,1136,90,31,ink,true);
text(s,'The observed decision rate missed its target. Approval-risk assurance remained inconclusive because of limited approvals and conservative simultaneous bounds.',72,570,1136,66,25,muted);

s=slide('An illustrative unresolved decision','Sources: thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json figure_1 and docs/thermotwin/OPERATING_DECISION_FINAL_FIGURES.md. Post hoc first eligible case, block 1 temperature-dependent contact. Raw interval lies below zero, final interval crosses zero. True saved margin -0.276609859712039 K. One of 27 final-calibration-induced abstentions, not a population efficacy estimate.');
const f=inputs.figure_1;
table(s,[['Stage','Margin lower K','Margin upper K'],['Initial candidate envelope',f.initial_interval.lower.toFixed(4),f.initial_interval.upper.toFixed(4)],['Verified after thermal',f.raw_after_interval.lower.toFixed(4),f.raw_after_interval.upper.toFixed(4)],['Final calibrated interval',f.final_interval.lower.toFixed(4),f.final_interval.upper.toFixed(4)]],180,265,[560,288,288]);
text(s,'Thermal acquisition narrowed the raw interval from 0.486 to 0.354 K.\nCalibration added 0.02544 K to each endpoint and left an abstention.',72,480,1136,105,29,ink,true);
text(s,'Saved true margin: -0.277 K. This is an illustrative case from the closed cohort.',72,599,1136,40,24,muted);

s=slide('The development measurement map','Sources: thermotwin/OPERATING_DECISION_PROSPECTIVE_FINAL_FIGURE_INPUTS.json figure_3. All 48 cells retain 60 cases, 20 paired development blocks. No final q is applied. The table below shows nominal sensing with balanced weights and three face instrumentation prices. All physical stress rows were actually computed. No selection, verification or pipeline failures. One face-action ineligibility per high-probe-loading cell, with all 60 cases retained.');
const cells=inputs.figure_3.cells.filter(c=>c.sensor_scenario==='nominal'&&c.cost_scenario.startsWith('balanced_'));
table(s,[['Face price / voltage','Stop','Thermal','Voltage','Face','Definitive / 60'],...cells.map((c,i)=>{const a=Object.fromEntries(c.actions.map(a=>[a.action,a.selected]));return[['0.25','1.00 primary','4.00'][i],a.stop_now,a.fixed_thermal,a.fixed_voltage,a.fixed_face_temperature,c.definitive].map(String);})],175,240,[280,145,165,165,145,236]);
text(s,'48 cells cross sensor quality and loading with energy, time and instrument costs.\nAcross cells, definitive decisions ranged from 28 to 32 of 60.',72,470,1136,90,29,ink,true);
text(s,'These are correlated development sensitivities before final calibration.\nReserved confirmation covers only nominal sensors with balanced equal-price costs.',72,575,1136,65,25,muted);

s=slide('Narrower intervals often still led to abstention','Source: thermotwin/OPERATING_DECISION_PROSPECTIVE_ABSTENTION_DIAGNOSIS.json summary. This diagnosis is post hoc. First abstaining layer partitions all 137 selector abstentions. Of 125 additional measurements, 116 narrowed the interval, 85 of those still abstained. Only three of 98 raw-crossing cases were solely due to opposing definitive candidate intervals.');
const c=chart(s,['Raw interval','Development padding','Final calibration'],[98,12,27],'First abstaining layer','Cases among 137 abstentions',115,'#607C90');
c.dataLabels.numberFormatCode='0';
text(s,'116 of 125 selected measurements narrowed uncertainty.\n85 of those still ended in abstention.',72,575,1136,65,29,ink,true);

s=slide('Recommended next work','Sources: docs/thermotwin/OPERATING_DECISION_FINAL_REPORT.md Recommendation within the tested scope and OPERATING_DECISION_ABSTENTION_DIAGNOSIS.md. A new decision-resolution objective is a hypothesis only. It needs fresh development, calibration and unseen reserved partitions and a predeclared precision/compute budget. Current exposed cases cannot confirm a redesign. Hardware tests and all broader thermoelectric claims remain separate.');
text(s,'Close this version with its primary conclusion intact',72,170,1136,75,35,ink,true);
text(s,'Test a new action objective based on the chance of resolving a reliable final decision per cost.\n\nUse fresh development, calibration and reserved evidence, with a precision and compute budget fixed in advance.',72,290,1136,220,30);
text(s,'No operating policy is promoted. Hardware sensing, probe loading and deployment latency remain to be tested.',72,565,1136,65,26,muted);

if(p.slides.items.length!==10)throw Error('Unexpected deck length');
await fs.mkdir(build,{recursive:true});await fs.mkdir(out,{recursive:true});
for(const [i,s] of p.slides.items.entries()){
 const image=await p.export({slide:s,format:'png',scale:1.25});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));
}
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const receipt=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(out,'operating-decision-final-presentation.pptx'),pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry',...([2,4,6,7,8].flatMap(i=>['--require-native-table-slide',String(i)]))],
 requiredNativeTableOwnerSlides:[2,4,6,7,8],requiredNativeChartOwnerSlides:[5,9],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'presentation-validation.json')});
console.log(JSON.stringify({slides:10,font,output:path.join(out,'operating-decision-final-presentation.pptx'),receipt},null,2));

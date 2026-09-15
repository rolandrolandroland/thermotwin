import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from thermotwin.studies import operating_decision_final_evaluation as f

root = Path(__file__).resolve().parents[3]
text = (Path(__file__).resolve().parent / 'stage2_reproduction.txt').read_text()
expected = [(2,6,2,4,6),(3,6,1,4,6),(4,6,0,4,6),(4,6,0,4,6),(4,0,6,8,2),(7,0,3,8,2),(7,1,2,8,2),(7,2,1,8,2)]
actual = [tuple(map(int, x)) for x in re.findall(r'decisions: approve=(\d+); reject=(\d+); insufficient=(\d+); true pass/violate=(\d+)/(\d+)', text)]
assert actual == expected, actual
assert 'four_state: -0.1579 K' in text and 'five_state: 0.1173 K' in text
assert re.findall(r'interval coverage where an interval exists: .*?\((\d+/\d+);', text) == ['9/10','9/10','10/10','10/10','10/10','10/10','10/10','9/10']
assert re.findall(r'diagnostic effort: (\d+) runs, .*? seconds .*?, ([\d.]+) J;', text) == [('2','57.99'),('5','94.86'),('3','85.53'),('3','85.53'),('2','57.99'),('5','94.86'),('3','85.53'),('3','85.53')]
assert text.count('numerical failures=0') == 8
print('Stage2: all eight report rows, pass prevalence, interval coverage, nominal margins, run counts, energy, and zero failures reproduce exactly at documented precision.')

config=f.OperatingDecisionFinalConfig()
parent=f.load_stage4_calibration_artifact(root/'thermotwin/OPERATING_DECISION_CALIBRATION_ARTIFACT.json')
actual_artifact=f.load_revised_artifact(root/'thermotwin/OPERATING_DECISION_FINAL_ARTIFACT.json')
f._validate_revised_artifact(actual_artifact,parent,config)
print('Saved final artifact passes current loader and frozen-component validation; no evaluation generated.')

for revision in [actual_artifact.implementation_revision, 'NOT_A_REAL_REVISION']:
    artifact=f._build_artifact(source_revision=revision,parent=parent,config=config,guard=actual_artifact.guard,calibration=actual_artifact.procedure_calibration)
    with TemporaryDirectory(prefix='thermotwin-audit-provenance-') as directory:
        path=Path(directory)/'artifact.json'
        f.save_revised_artifact(artifact,path)
        with patch.object(f,'_run_truths',side_effect=RuntimeError('MOCK_GENERATION_BOUNDARY')) as generation:
            try:
                f.evaluate_reserved_stage5(path,parent,config=config)
            except RuntimeError as error:
                assert str(error)=='MOCK_GENERATION_BOUNDARY'
            else:
                raise AssertionError('did not reach generation boundary')
            assert generation.call_count==1
    print(f'Frozen evaluator accepts claimed revision {revision!r} and reaches MOCK generation boundary without inspecting current code revision.')
print('No Stage5 seed was instantiated; _run_truths was mocked in every evaluator invocation.')

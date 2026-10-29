import hashlib
import json
from pathlib import Path
import stat
import tempfile
import unittest
import warnings
import zipfile

from tools import operating_decision_closeout_package as package
from tools import operating_decision_report_diagnostics as diagnostics
from tools import verify_prospective_phase_f_result as audit
from tools.build_operating_decision_final_report import sections


class CloseoutIntegrityTests(unittest.TestCase):
    def make_archive(self,path,content=b'accepted evidence',stored=None,extra=None,link=False):
        stored=content if stored is None else stored
        row={'path':'repository/evidence.json','bytes':len(stored),'sha256':hashlib.sha256(stored).hexdigest()}
        manifest={'files':[row]};manifest['artifact_digest']=audit.digest(package.DOMAIN,manifest)
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest))
            info=zipfile.ZipInfo(row['path']);info.external_attr=((stat.S_IFLNK if link else stat.S_IFREG)|0o644)<<16
            z.writestr(info,content)
            if extra:z.writestr(*extra)

    def test_archive_round_trip_retains_exact_bytes_and_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'data.zip';self.make_archive(p)
            result=package.verify(p,root/'unpacked')
            self.assertEqual((root/'unpacked/repository/evidence.json').read_bytes(),b'accepted evidence')
            self.assertTrue(result['round_trip_extraction_verified'])
            self.assertFalse(result['immutable_external_deposit'])
            with self.assertRaises(ValueError):package.verify(p,root/'unpacked')

    def test_changed_member_is_rejected_even_with_a_valid_zip_crc(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.zip';self.make_archive(p,content=b'altered evidence!',stored=b'accepted evidence')
            with self.assertRaises(ValueError):package.verify(p)

    def test_unlisted_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.zip';self.make_archive(p,extra=('unlisted.txt',b'unknown'))
            with self.assertRaises(ValueError):package.verify(p)

    def test_traversal_and_absolute_paths_are_rejected_before_extraction(self):
        for name in ('../escape.txt','/absolute.txt','repository/../escape.txt','repository\\escape.txt'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                p=Path(d)/'data.zip';self.make_archive(p,extra=(name,b'unknown'))
                with self.assertRaises(ValueError):package.verify(p,Path(d)/'out')
                self.assertFalse((Path(d)/'out').exists())

    def test_symlink_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.zip';self.make_archive(p,link=True)
            with self.assertRaises(ValueError):package.verify(p)

    def test_duplicate_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.zip'
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');self.make_archive(p,extra=('repository/evidence.json',b'replacement'))
            with self.assertRaises(ValueError):package.verify(p)

    def test_resealed_manifest_cannot_omit_a_member(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.zip';manifest={'files':[]};manifest['artifact_digest']=audit.digest(package.DOMAIN,manifest)
            with zipfile.ZipFile(p,'w') as z:
                z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest));z.writestr('evidence.json','orphan')
            with self.assertRaises(ValueError):package.verify(p)

    def seed_entries(self):
        key={'block':0,'campaign':'accepted_campaign','channel':None,'family':None,'partition':'reserved',
             'protocol_version':'thermotwin-operating-decision-rng-v2','purpose':'physical_parameter_0',
             'run':None,'stream_kind':'device_truth'}
        seed=int.from_bytes(hashlib.sha256(audit.canonical({**key,'domain':'thermotwin.operating_decision.random_stream'})).digest(),'big')
        return [{'key':dict(key),'seed':seed,'pairing_member':f} for f in audit.FAMILIES]

    def test_scalar_parameter_replay_authenticates_the_registered_seed(self):
        entries=self.seed_entries();expected=entries[0]['seed']
        self.assertEqual(diagnostics.original_seed(entries,'physical_parameter_0',0),expected)
        for row in entries:row['seed']+=1
        with self.assertRaises(ValueError):diagnostics.original_seed(entries,'physical_parameter_0',0)

    def test_scalar_parameter_replay_rejects_incomplete_or_inconsistent_sharing(self):
        for mode in ('missing','duplicate_family','different_seed'):
            entries=self.seed_entries()
            if mode=='missing':entries.pop()
            elif mode=='duplicate_family':entries[-1]['pairing_member']=entries[0]['pairing_member']
            else:entries[-1]['seed']+=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):diagnostics.original_seed(entries,'physical_parameter_0',0)

    def test_final_report_retains_all_policy_clearance_rows_and_the_non_success(self):
        pages=sections(Path(__file__).resolve().parents[1])
        text=' '.join(value for _,items in pages for kind,value in items if kind=='p')
        self.assertIn('primary success was not established',text)
        self.assertIn('No procedure qualified',text)
        secondary=[v for _,items in pages for kind,v in items if kind=='table' and v[0][0]=='Clearance K']
        self.assertEqual(len(secondary),1)
        rows=secondary[0][1]
        self.assertEqual(len(rows),15)
        self.assertEqual({(row[0],row[1]) for row in rows},{(c,p) for c in ('0.05','0.1','0.2') for p in ('Selector','Stop','Thermal','Voltage','Face')})


if __name__=='__main__':unittest.main()

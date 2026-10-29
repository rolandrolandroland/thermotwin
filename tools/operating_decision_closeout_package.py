"""Create or independently verify a local evidence ZIP without rerunning science.

The package has a per-file SHA-256 manifest. Its whole-file SHA is published
separately to avoid a self-reference. It is hash-verified, not an external
immutable deposit. No Git metadata, virtual environment or unrelated worktree
edits are packaged.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import zipfile

from tools import verify_prospective_phase_f_result as audit

DOMAIN='thermotwin.operating_decision.closeout_package_v1'


def safe_name(name):
    path=PurePosixPath(name)
    audit.require(bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name and name==path.as_posix(), 'unsafe archive path')
    return path


def stream_hash(stream):
    h=hashlib.sha256();n=0
    while chunk:=stream.read(1024*1024):h.update(chunk);n+=len(chunk)
    return {'sha256':h.hexdigest(),'bytes':n}


def file_record(path):
    with Path(path).open('rb') as f:return stream_hash(f)


def verify(archive,extract_to=None):
    if extract_to is not None:
        audit.require(not extract_to.exists(),'refuse to reuse extraction directory')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        audit.require(len(names)==len(set(names)),'duplicate archive paths')
        for entry in z.infolist():
            safe_name(entry.filename)
            audit.require(not entry.is_dir() and not stat.S_ISLNK(entry.external_attr>>16),'archive directories or symbolic links are not allowed')
        manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
        audit.require(manifest['artifact_digest']==audit.digest(DOMAIN,{k:v for k,v in manifest.items() if k!='artifact_digest'}),'package manifest seal differs')
        records=manifest['files']
        audit.require(len(records)==len({v['path'] for v in records}),'duplicate manifest entries')
        audit.require(set(names)=={'PACKAGE_MANIFEST.json',*(v['path'] for v in records)},'archive and manifest membership differ')
        for row in records:
            info=z.getinfo(row['path'])
            audit.require(info.file_size==row['bytes'],'archive member length differs')
            with z.open(info) as source:
                audit.require(stream_hash(source)=={k:row[k] for k in ('sha256','bytes')},'archive member hash differs: '+row['path'])
        if extract_to is not None:
            extract_to.mkdir(parents=True)
            for name in names:
                target=extract_to/safe_name(name)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as source,target.open('xb') as dest:
                    while chunk:=source.read(1024*1024):dest.write(chunk)
            for row in records:
                audit.require(file_record(extract_to/row['path'])=={k:row[k] for k in ('sha256','bytes')},'extracted file differs')
    return {'package_file':file_record(archive),'manifest_digest':manifest['artifact_digest'],
            'files_verified':len(records),'uncompressed_bytes':sum(v['bytes'] for v in records),
            'safe_membership_verified':True,'round_trip_extraction_verified':extract_to is not None,
            'immutable_external_deposit':False}


def pack(args):
    root=args.repository_root.resolve();plan=json.loads(args.plan.read_bytes())
    audit.require(not args.output.exists(),'refuse to overwrite an evidence package')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    audit.require(head==plan['repository_base_revision'],'repository base advanced; review the package plan')
    raw=subprocess.check_output(['git','archive','--format=tar',head],cwd=root)
    sources={}
    with tarfile.open(fileobj=io.BytesIO(raw)) as snapshot:
        for entry in snapshot:
            if entry.isfile():sources['repository/'+entry.name]=snapshot.extractfile(entry).read()
    for name in plan['source_overlays']:
        safe_name(name);sources['repository/'+name]=(root/name).read_bytes()
    mapping={}
    for name in plan['evidence_directories']:
        directory=root/name
        audit.require(directory.is_dir(),'missing required evidence directory: '+name)
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                audit.require(not path.is_symlink(),'evidence symlink is not archival data')
                mapping['repository/'+path.relative_to(root).as_posix()]=path
    for name in plan['evidence_files']:
        audit.require((root/name).is_file(),'missing evidence file: '+name)
        mapping['repository/'+name]=root/name
    evaluation=root/plan['evaluation_path']
    audit.require(file_record(evaluation)['sha256']=='299f75ad492e5bde0acc724ed4e72ae029c86f2c05538677c2a856c23734c564','wrong reserved evaluation')
    saved=json.loads(evaluation.read_bytes())
    manifest=saved['context']['source_manifest']
    frozen=root/'.artifacts/operating-decision/source-7e3b9a3'
    for entry in manifest['files']:
        expected={'sha256':entry['sha256'],'bytes':entry['byte_count']}
        audit.require(file_record(frozen/entry['path'])==file_record(root/entry['path'])==expected,'frozen source differs')
        mapping['frozen-source/'+entry['path']]=frozen/entry['path']
    sources['frozen-source/SOURCE_MANIFEST.json']=json.dumps(manifest,sort_keys=True,indent=2).encode()+b'\n'
    for directory in (root/plan['figures_directory'],root/plan['deliverables_directory']):
        for path in sorted(directory.iterdir()):
            if path.is_file():mapping['repository/'+path.relative_to(root).as_posix()]=path
    sources['README.md']=(root/'docs/thermotwin/OPERATING_DECISION_REPRODUCIBILITY.md').read_bytes()
    audit.require(not(set(sources)&set(mapping)),'duplicate source/data destination')
    records=[]
    for name in sorted(set(sources)|set(mapping)):
        safe_name(name)
        value={'sha256':hashlib.sha256(sources[name]).hexdigest(),'bytes':len(sources[name])} if name in sources else file_record(mapping[name])
        records.append({'path':name,**value})
    value={'schema_version':1,'package_version':'operating_decision_closeout_v1','repository_base_revision':head,
           'source_overlay_paths':plan['source_overlays'],'reserved_source_revision':audit.SOURCE_REVISION,
           'scientific_result_digest':saved['scientific_result_digest'],'files':records,
           'source_manifest_files':len(manifest['files']),'external_immutable_deposit':False,
           'environment_binaries_included':False,'new_fits_observations_or_partitions':False}
    value['artifact_digest']=audit.digest(DOMAIN,value)
    encoded=json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        def info(name):
            i=zipfile.ZipInfo(name,date_time=(2026,10,29,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=(stat.S_IFREG|0o644)<<16;return i
        z.writestr(info('PACKAGE_MANIFEST.json'),encoded)
        for row in records:
            name=row['path']
            with z.open(info(name),'w',force_zip64=True) as dest:
                if name in sources:dest.write(sources[name])
                else:
                    with mapping[name].open('rb') as source:
                        while chunk:=source.read(1024*1024):dest.write(chunk)
    audit.require(not args.manifest_output.exists(),'refuse to replace a package manifest')
    with args.manifest_output.open('xb') as f:f.write(encoded)
    print(json.dumps({'archive':str(args.output),'manifest_digest':value['artifact_digest'],'files':len(records),'uncompressed_bytes':sum(r['bytes'] for r in records)},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('pack');a.add_argument('--repository-root',type=Path,default=Path('.'));a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--manifest-output',type=Path,required=True)
    a=sub.add_parser('verify');a.add_argument('--archive',type=Path,required=True);a.add_argument('--extract-to',type=Path);a.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.command=='pack':pack(args)
    else:
        value=verify(args.archive,args.extract_to)
        with args.output.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
        print(json.dumps(value,indent=2))


if __name__=='__main__':main()

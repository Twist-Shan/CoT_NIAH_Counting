from pathlib import Path
import tarfile,hashlib,json,os
r=Path(__file__).resolve().parents[1]/'runs/topk_completion_20260907_v1'
expected='857ccc8b5841f0c8c11f7280290d71f0468980be97f6f5087a4768424aaeef19'
assert hashlib.sha256((r/'results.tgz').read_bytes()).hexdigest()==expected
dest=Path('\\\\?\\'+str((r/'downloaded').resolve())) if os.name=='nt' else r/'downloaded'
with tarfile.open(r/'results.tgz') as f:f.extractall(dest,filter='data')
audit=json.loads((r/'analysis/audit.json').read_text())
for rel,h in audit['file_hashes'].items():assert hashlib.sha256((dest/rel).read_bytes()).hexdigest()==h,rel
assert hashlib.sha256((dest/'protocol.json').read_bytes()).hexdigest()==audit['protocol_sha256']
(r/'analysis/download_audit.json').write_text(json.dumps(dict(status='PASS',archive_sha256=expected,points_verified=len(audit['file_hashes']))))
print('DOWNLOAD_AUDIT_PASS',len(audit['file_hashes']))

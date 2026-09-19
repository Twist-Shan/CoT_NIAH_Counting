import json, pathlib, hashlib, tarfile
b = pathlib.Path(__file__).resolve().parents[1]
r = b/'runs/aligned_transfer_20260906_v3'
h = hashlib.sha256((r/'results.tgz').read_bytes()).hexdigest()
assert h == '42494c8c72704c0ef25768539a51ead71e345eb022412046763b111d5e4b3f6b'
tarfile.open(r/'results.tgz').extractall(r/'downloaded', filter='data')
p = b/'STATUS.json'
s = json.loads(p.read_text(encoding='utf-8'))
s['aligned_transfer_v3'].update(status='COMPLETE_AUDIT_PENDING', archive_sha256=h, last_monitor=json.loads((r/'monitor_20260906T160151Z.json').read_text()))
p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
src = (b/'deployment/audit_aligned_results.py').read_text(encoding='utf-8')
src = src.replace('runs/aligned_transfer_20260906_v2/downloaded','runs/aligned_transfer_20260906_v3/downloaded').replace("['Qwen3-8B','Gemma4-E4B']", "['Gemma4-E4B']")
(b/'deployment/audit_aligned_gemma_v3_results.py').write_text(src,encoding='utf-8')
print(h)

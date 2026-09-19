from pathlib import Path
import hashlib,tarfile
b=Path(__file__).resolve().parents[1];d=b/'deployment';r=b/'runs/aligned_gemma_kth_sdpa_20260907_v1'
assert hashlib.sha256((r/'results.tgz').read_bytes()).hexdigest()=='c059252656d7d72908a64c62b0fb9822e9902b9f054fee77f58d8039273f39d9'
tarfile.open(r/'results.tgz').extractall(r/'downloaded',filter='data')
s=(d/'audit_aligned_gemma_v3_results.py').read_text(encoding='utf-8').replace('aligned_transfer_20260906_v3','aligned_gemma_kth_sdpa_20260907_v1').replace("['kth','category']","['kth']")
(d/'audit_regenerated_aligned.py').write_text(s,encoding='utf-8')
s=(d/'verify_aligned_v3_code.py').read_text().replace('aligned_transfer_20260906_v3','aligned_gemma_kth_sdpa_20260907_v1')
(d/'verify_regenerated_code.py').write_text(s)
s=(d/'audit_original_target_scores.py').read_text()
s=s.replace("[('Qwen3-8B','v2','models--Qwen--Qwen3-8B'),('Gemma4-E4B','v3','models--google--gemma-4-E4B-it')]","[('Gemma4-E4B','v1','models--google--gemma-4-E4B-it')]")
s=s.replace("root=b/f'aligned_transfer_20260906_{version}'","root=b/'aligned_gemma_kth_sdpa_20260907_v1'").replace("['kth','category']","['kth']").replace('aligned_original_target_score_audit.json','regenerated_target_score_audit.json')
(d/'audit_regenerated_target_scores.py').write_text(s)

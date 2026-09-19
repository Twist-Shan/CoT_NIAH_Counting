from pathlib import Path
b=Path(__file__).resolve().parent
s=(b/'diagnose_gemma_v3_kth.py').read_text()
s=s.replace('from protocol import encode_ids','from protocol import encode_ids\nfrom realistic_niah_v4.modeling import _temporary_attention_backend')
s=s.replace('gens.append(generate_answer_completion(model,tok,enc,max_new_tokens=64))','with _temporary_attention_backend(model, "eager"):\n   gens.append(generate_answer_completion(model,tok,enc,max_new_tokens=64))')
s=s.replace('aligned_gemma_v3_kth_diagnostic.json','aligned_gemma_v3_kth_eager_diagnostic.json')
(b/'diagnose_gemma_v3_kth_eager.py').write_text(s)

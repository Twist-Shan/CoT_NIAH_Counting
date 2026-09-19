"""Read-only inventory of canonical Enumeration artifacts on the shared volume."""
import json
from pathlib import Path

mount = Path('outputs/external/lambda_nfs_CoT-Native-thinking-v5')
old = mount/'supplements/enumeration_alignment_20260916'
for model in ('Qwen3-8B', 'Gemma4-E4B'):
    folder = mount/'runs/v6_enumeration_replication_20260828/enumeration_bullet'/model/'generation'
    with (folder/'generations.jsonl').open() as f:
        rows = [json.loads(line) for line in f]
    canonical = [r for r in rows if 1234 <= r['seed'] <= 1263]
    print(json.dumps(dict(model=model, total=len(rows), canonical=len(canonical),
        keys=sorted(rows[0]), sample={k:rows[0].get(k) for k in ('seed','split','gold_count','model_label','prompt_mode','passage_sha256')},
        manifests={p.name:json.loads(p.read_text()) for p in folder.glob('manifest*.json')})))
print('N10_MANIFEST', (old/'fresh_n10_update_v1/manifest.json').read_text()[:3500])

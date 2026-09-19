from pathlib import Path
b=Path(__file__).resolve().parent
s=(b/'diagnose_aligned_clean.py').read_text(encoding='utf-8')
s=s.replace('Gemma4-E4B','Qwen3-8B').replace('category_count_seed1254_city3_targetcity','category_count_seed1261_city7_targetcity')
s=s.replace('aligned_clean_diagnostic_fixed.json','aligned_qwen_clean_diagnostic_fixed.json').replace('aligned_clean_diagnostic.json','aligned_qwen_clean_diagnostic.json')
(b/'diagnose_aligned_qwen_clean.py').write_text(s,encoding='utf-8')

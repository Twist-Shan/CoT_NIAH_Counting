"""Use the existing CoT styles with NCC-selected plotting data."""
from pathlib import Path
import importlib.util,json
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
p=ROOT/'figures/cot_appendix_20260912/representations/build_figures.py'
s=importlib.util.spec_from_file_location('cot_plots',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.OUT=OUT;m.DATA=OUT/'data'
m.main()
p=OUT/'figure_manifest.json';v=json.loads(p.read_text());v['selection_changed']=True;v['pca_refit']='Gemma domain panels at NCC-selected layers; canonical coordinates reused';v['selection_file']='selection.json';p.write_text(json.dumps(v,indent=2))

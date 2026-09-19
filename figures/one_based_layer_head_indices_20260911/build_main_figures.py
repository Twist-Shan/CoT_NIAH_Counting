"""Run established top-level builders with outputs isolated in this directory."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=['nonthinking_main', 'native_main'])
    target = parser.parse_args().target
    out = HERE / target
    out.mkdir(exist_ok=True)
    if target == 'nonthinking_main':
        source = WORK / 'figures/non-thinking'
        shutil.copytree(source / 'data', out / 'data', dirs_exist_ok=True)
        shutil.copy2(source / 'transformer_mechanism_polished.drawio', out)
        path = source / 'build_nonthinking_section.py'
        code = path.read_text(encoding='utf-8')
        assert code.count('OUT=Path(__file__).resolve().parent') == 1
        code = code.replace('OUT=Path(__file__).resolve().parent', 'OUT=STAGING')
        code = code.replace('WORK=OUT.parents[1]', 'WORK=WORKSPACE')
        exec(compile(code, str(path), 'exec'),
             {'__file__': str(path), '__name__': '__main__', 'STAGING': out, 'WORKSPACE': WORK})
        pdf_python = Path(sys.executable)
        subprocess.run([str(pdf_python), '-s', str(source / 'compose_selectable_pdf.py'),
                        '--composite', str(source / 'nonthinking_section_v11_outlined.pdf'),
                        '--panels', str(out / 'result_panels_v11.pdf'),
                        '--output', str(out / 'nonthinking_form_retrieve_consolidate.pdf'),
                        '--audit', str(out / 'composition_audit.json')], check=True)
    else:
        source = WORK / 'figures/cot-reasoning'
        sys.path.insert(0, str(source))
        path = source / 'prepare_data.py'
        code = path.read_text(encoding='utf-8')
        code = code.replace('OUT = Path(__file__).resolve().parent', 'OUT = STAGING')
        code = code.replace('ROOT = OUT.parents[1]', 'ROOT = WORKSPACE')
        code = code.replace('FINAL = ROOT / "output/pdf/cot_reasoning_main.pdf"',
                            'FINAL = OUT / "native_retrieve_encode_count_loop.pdf"')
        module = types.ModuleType('prepare_data')
        module.__dict__.update(__file__=str(path), STAGING=out, WORKSPACE=WORK)
        sys.modules['prepare_data'] = module
        exec(compile(code, str(path), 'exec'), module.__dict__)
        path = source / 'build_figure.py'
        code = path.read_text(encoding='utf-8').replace(
            "CONTROL=d.ROOT/'output/pdf/cot_reasoning_controls.pdf'",
            "CONTROL=d.OUT/'cot_reasoning_controls.pdf'")
        exec(compile(code, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})
    print(json.dumps({'target': target, 'output': str(out), 'raw_data_modified': False}))


if __name__ == '__main__':
    main()

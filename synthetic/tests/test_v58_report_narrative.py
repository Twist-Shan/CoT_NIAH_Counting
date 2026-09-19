"""Lightweight structure checks for the self-contained reader-facing report."""
import importlib.util
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_caption_numbering_does_not_mutate_embedded_script():
    spec = importlib.util.spec_from_file_location('v58_report_narrative', ROOT/'scripts/v58_report_narrative.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = '<script>const x="图4a";</script><figcaption>图4a｜first</figcaption><figcaption>图｜second</figcaption>'
    result = module.finalize_structure(source)
    assert '<script>const x="图4a";</script>' in result
    assert '<figcaption>图 1｜first</figcaption>' in result
    assert '<figcaption>图 2｜second</figcaption>' in result


def test_generated_report_has_complete_navigation_and_unique_captions():
    report = (ROOT/'reports/NiaH_Synthetic_report.html').read_text(encoding='utf-8')
    anchors = re.findall(r'<a href="#([^"]+)">', report)
    assert len(anchors) == 13
    for anchor in anchors:
        assert report.count(f'id="{anchor}"') == 1
    captions = re.findall(r'<figcaption>图 (\d+)｜', report)
    assert list(map(int,captions)) == list(range(1,len(captions)+1))
    assert len(captions) >= 10
    assert report.count('说明性示例') >= 12


def test_generated_report_preserves_critical_scope_and_loss_definitions():
    report = (ROOT/'reports/NiaH_Synthetic_report.html').read_text(encoding='utf-8')
    for term in ['代码没有再除以系数总和', '无联合mode训练', '全部正文needles',
                 'Thinking最多26个新token', '不是10个独立训练seed', '不能称为全新未见',
                 '尚未补算同一trace query', 'expected_abs_error', 'geometry-layer-3d']:
        assert term in report
    assert '<script src="http' not in report
    assert '<img src="http' not in report

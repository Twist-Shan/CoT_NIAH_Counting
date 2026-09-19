"""Deliver the explicitly approved two-score exception without rewriting audits."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

B = Path(__file__).resolve().parents[1]
VERSION = 'native_broad_full_span_20260909_v1'
RUN = B / 'runs' / VERSION


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    authorization = read(RUN / 'numeric_recovery/authorization.json')
    assert authorization['approved'] is True and authorization['version'] == VERSION
    assert authorization['user_reply'] == '接受该验收例外，完成 HTML'
    root_plain = RUN / 'downloaded'
    root = Path('\\\\?\\' + str(root_plain)) if os.name == 'nt' else root_plain
    cfg = read(root / 'protocol.json')
    assert sha(root / 'protocol.json') == 'eb29c8afa0180e6a0af265a471565fcc37f6cd31d15fc5b4d3013b1e984f692b'
    assert sha(RUN / 'package.tgz') == 'b3585bb4370b9235e6ba2bd9811ad8889754d13197b355a55203466f0549592c'
    assert sha(RUN / 'results.tgz') == 'e5bf11772b16981c4052dd257b6a7a070d76ec8a1a81820af738f7edc1b1cee8'
    for relative, expected in cfg['files'].items():
        assert sha(root / relative) == sha(RUN / 'package' / relative) == expected
    remote = read(RUN / 'remote_audit.json')
    local = read(RUN / 'numeric_recovery/local_analysis/audit.json')
    comparison = read(RUN / 'numeric_recovery/comparison.json')
    assert remote['status'] == 'PASS' and remote['numeric_comparison']['mode'] == 'exact'
    assert remote['numeric_comparison']['max_ulp_distance'] == 0
    assert local['status'] == 'NUMERIC_HOLD' and local['diagnostic_only'] is True
    assert local['numeric_comparison']['allowed_ulp_distance'] == 8
    assert local['numeric_comparison']['max_ulp_distance'] == 9
    assert local['checks'] == remote['checks'] == dict(frozen_files=241, geometry_inputs=1200,
        literal_record_spans=4831, discovery_rows=800, discovery_rows_rescored=610,
        complete_rankings_reproduced=4, frozen_doses=26, arms_rescored=8340, points=1668)
    expected_violations = {('Qwen3-8B', 'kth', 'kth_needle_seed1242_level10', 18, 13),
        ('Qwen3-8B', 'category', 'category_count_seed1252_city3_targetflower', 9, 16)}
    assert len(local['numeric_violations']) == 2
    assert {(v['model'], v['task'], v['case_id'], v['layer'], v['head']) for v in local['numeric_violations']} == expected_violations
    assert all(v['ulp_distance'] == 9 for v in local['numeric_violations'])
    assert local['file_hashes'] == remote['file_hashes']
    for relative, expected in remote['file_hashes'].items():
        assert sha(root / relative) == expected
    assert len(comparison['comparisons']) == 6
    for item in comparison['comparisons']:
        assert item['max_numeric_difference'] == 0
        assert sha(root / 'analysis' / item['file']) == sha(RUN / 'numeric_recovery/local_analysis' / item['file']) == item['remote_sha256'] == item['local_sha256']
    evidence = ['remote_audit.json', 'downloaded/analysis/audit.json', 'numeric_recovery/local_analysis/audit.json',
        'numeric_recovery/comparison.json', 'numeric_recovery/authorization.json',
        'numeric_recovery/avx512_reproduction.json', 'numeric_recovery/intervention_observation.json']
    evidence += ['downloaded/analysis/' + item['file'] for item in comparison['comparisons']]
    accepted = dict(status='ACCEPTED_NUMERIC_EXCEPTION', version=VERSION,
        authorization=authorization, accepted_at_utc=datetime.now(timezone.utc).isoformat(),
        local_numeric_comparison=local['numeric_comparison'], violations=local['numeric_violations'],
        original_local_audit_status=local['status'], remote_audit_status=remote['status'],
        checks=local['checks'], evidence_hashes={name: sha(RUN / name) for name in evidence})
    save(RUN / 'numeric_recovery/accepted_verification.json', accepted)
    html = B.parent.parent / 'NiaH_Additional-tasks_report.html'
    previous = RUN / 'previous_report'
    previous.mkdir(exist_ok=True)
    if not (previous / html.name).exists():
        shutil.copyfile(html, previous / html.name)
    def run(name):
        result = subprocess.run([sys.executable, str(B / 'deployment' / name)], cwd=B.parent,
            text=True, encoding='utf-8', capture_output=True, timeout=180, check=True)
        return json.loads(result.stdout.strip())
    built = run('build_task_local_html.py')
    check = run('check_additional_html.py')
    assert built['all_requested_components_complete'] and built['figures'] == 12
    assert check['status'] == 'PASS' and check['native_full_span_rows_verified'] == 4 and check['target_category_rows_verified'] == 6
    delivery = dict(status='DELIVERED', version=VERSION, validation='ACCEPTED_NUMERIC_EXCEPTION',
        points_verified=1668, arms_rescored=8340, archive_sha256=sha(RUN / 'results.tgz'),
        protocol_sha256=sha(root / 'protocol.json'), comparisons=comparison['comparisons'],
        numeric_comparison=local['numeric_comparison'], accepted_verification_sha256=sha(RUN / 'numeric_recovery/accepted_verification.json'),
        report=built, report_check=check, delivered_at_utc=datetime.now(timezone.utc).isoformat())
    save(RUN / 'delivery.json', delivery)
    save(RUN / 'numeric_recovery/failure_resolution.json', dict(status='RESOLVED_BY_USER_APPROVED_EXCEPTION',
        original_failure_sha256=sha(RUN / 'delivery_failure.json'), delivery_sha256=sha(RUN / 'delivery.json'),
        original_failure_preserved=True))
    print(json.dumps(delivery, ensure_ascii=False))


if __name__ == '__main__':
    main()

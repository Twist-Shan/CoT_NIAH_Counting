"""Replace draw.io's outlined B/C/D panel group with its native Matplotlib PDF.

The original drawing supplies the mechanism and the exact placement transform.
Only the outlined results group is removed. Native PDF composition preserves
embedded fonts, searchable text, paths, and transparency without rasterization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream


def concat(parent, local):
    a, b, c, d, e, f = parent
    u, v, w, x, y, z = local
    return (a*u+c*v, b*u+d*v, a*w+c*x, b*w+d*x,
            a*y+c*z+e, b*y+d*z+f)


def close_enough(actual, expected):
    return len(actual) == len(expected) and all(
        abs(float(a)-float(b)) < 1e-4 for a, b in zip(actual, expected)
    )


def find_panel_group(operations, width, height):
    """Locate the outlined SVG's full-page white background and enclosing clip."""
    matrix = (1., 0., 0., 1., 0., 0.)
    stack = []
    candidates = []
    pattern = [
        (b'm', [0, height]), (b'l', [width, height]),
        (b'l', [width, 0]), (b'l', [0, 0]), (b'l', [0, height]),
    ]
    for index, (args, op) in enumerate(operations):
        if op == b'q':
            stack.append((matrix, index))
        elif op == b'Q':
            matrix, _ = stack.pop()
        elif op == b'cm':
            matrix = concat(matrix, tuple(float(v) for v in args))
        elif op == b'm' and len(stack) >= 2:
            chunk = operations[index:index+len(pattern)]
            if len(chunk) == len(pattern) and all(
                got_op == want_op and close_enough(got_args, want_args)
                for (got_args, got_op), (want_op, want_args) in zip(chunk, pattern)
            ):
                start = stack[-2][1]
                # The group must start with the rectangular SVG viewport clip.
                assert [op for _, op in operations[start:start+5]] == [
                    b'q', b're', b'W*', b'n', b'q'
                ], 'Unexpected SVG clipping structure; preserve the original PDF.'
                candidates.append((start, matrix))
    assert len(candidates) == 1, (
        f'Expected one outlined results group, found {len(candidates)}. '
        'Supply a fresh draw.io export, not an already composed PDF.'
    )
    start, matrix = candidates[0]
    depth = 0
    for end in range(start, len(operations)):
        op = operations[end][1]
        depth += (op == b'q') - (op == b'Q')
        if depth == 0:
            return start, end+1, matrix
    raise ValueError('Unbalanced PDF graphics state in results group.')


def compose(composite: Path, panels: Path, output: Path):
    source_hash = hashlib.sha256(composite.read_bytes()).hexdigest()
    panel_hash = hashlib.sha256(panels.read_bytes()).hexdigest()
    original, native = PdfReader(composite), PdfReader(panels)
    assert len(original.pages) == len(native.pages) == 1
    writer = PdfWriter()
    page, panel = writer.add_page(original.pages[0]), native.pages[0]
    assert page.rotation == panel.rotation == 0
    assert list(panel.mediabox.lower_left) == [0, 0]
    width, height = float(panel.mediabox.width), float(panel.mediabox.height)
    content = ContentStream(page.get_contents(), writer)
    start, end, matrix = find_panel_group(content.operations, width, height)
    original_text = page.extract_text() or ''

    # draw.io splits the last SVG into sibling clipping groups. Everything
    # after its first background belongs to B/C/D, so remove all those groups.
    # In this composite A precedes the SVG and is the only source of PDF text.
    assert not any(op in (b'BT', b'Tj', b'TJ')
                   for _, op in content.operations[start:]), (
        'Text follows the outlined SVG; review drawing order before replacement.'
    )
    end = len(content.operations)
    prefix = content.operations[:start]
    depth = sum((op == b'q') - (op == b'Q') for _, op in prefix)
    assert depth >= 0
    # Retain A's original drawing operators exactly, including its arrows/math.
    content.operations = prefix + [([], b'Q')] * depth
    page.replace_contents(content)
    a, b, c, d, e, f = matrix
    # The outlined SVG points downward; the native PDF points upward.
    placement = (a, b, -c, -d, c*height+e, d*height+f)
    assert a > 0 and d < 0 and abs(b)+abs(c) < 1e-8
    page.merge_transformed_page(panel, Transformation(placement), expand=False)
    writer.add_metadata({
        '/Title': 'Non-thinking: form, retrieve, and consolidate',
        '/Producer': 'Native PDF composition with embedded text (pypdf)',
        '/NonthinkingPanelComposition': panel_hash,
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.pending.pdf')
    with temporary.open('wb') as stream:
        writer.write(stream)
    check = PdfReader(temporary)
    extracted = check.pages[0].extract_text() or ''
    required = ['Record-span patching', 'Head ablation', 'Answer-state patching',
                'Intervention layer', 'Qwen', 'Gemma', 'Restore', 'Corrupt']
    assert all(label in extracted for label in required), 'Missing result text.'
    assert all(label in extracted for label in ['High layer', 'Form', 'Retrieve',
                                               'Consolidate', 'Total: 8'])
    assert len(extracted) > len(original_text)
    temporary.replace(output)
    audit = {
        'status': 'PASS', 'source': str(composite), 'panels': str(panels),
        'source_sha256': source_hash, 'panel_sha256': panel_hash,
        'output': str(output),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
        'page_box': [float(v) for v in page.mediabox],
        'removed_operator_range': [start, end],
        'native_panel_transform': placement,
        'panel_box_pdf_coordinates': [e, d*height+f, a*width+e, f],
        'original_text_characters': len(original_text),
        'composed_text_characters': len(extracted),
        'verified_text': required,
        'mechanism_drawing_operators_preserved': True,
    }
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composite', required=True, type=Path)
    parser.add_argument('--panels', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    audit = compose(args.composite, args.panels, args.output)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))

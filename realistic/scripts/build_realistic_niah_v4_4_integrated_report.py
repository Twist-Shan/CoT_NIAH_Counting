from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_rows_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_viewport_meta(document: str) -> str:
    """Make responsive report CSS use the physical device viewport."""

    if re.search(r'<meta\s+name=["\']viewport["\']', document, flags=re.I):
        return document
    document, count = re.subn(
        r"<head>",
        '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',
        document,
        count=1,
        flags=re.I,
    )
    if count != 1:
        raise RuntimeError("Base report has no head element for viewport metadata")
    return document


def ensure_first_locator_appendix(document: str, repo_root: Path) -> str:
    """Keep first-locator evidence in one standalone appendix, never in the body."""

    legacy_pattern = re.compile(
        r'<div class="figure-block first-span-profile">.*?'
        r'<div class="conclusion">.*?</div>\s*</div>',
        re.S,
    )
    document, legacy_count = legacy_pattern.subn("", document)
    if legacy_count not in (0, 1):
        raise RuntimeError("Unexpected legacy first-span body-section count")
    appendix_pattern = re.compile(
        r'<section id="appendix-first-locator".*?</section>', re.S
    )
    document, appendix_count = appendix_pattern.subn("", document)
    if appendix_count not in (0, 1):
        raise RuntimeError("Unexpected first-locator appendix count")
    if "</main>" not in document:
        raise RuntimeError("Base report has no main terminator for Appendix A")

    from build_realistic_niah_v4_4_report import _first_locator_appendix

    fragment = _first_locator_appendix(repo_root).strip()
    return document.replace("</main>", fragment + "\n</main>", 1)


def remove_first_locator_body_claims(document: str) -> str:
    """Remove the legacy endpoint-locator interpretation from the main narrative."""

    return document.replace(
        "first-needle locator 比稳定的任意单一 targeted-occurrence head 更清楚。",
        "",
    )


def remove_legacy_endpoint_phenotype_table(document: str) -> str:
    """Drop the endpoint-only locator table superseded by complete-span evidence."""

    pattern = re.compile(
        r'<details class="data-table"><summary>'
        r'V4\.4 endpoint phenotype counts and representatives.*?</details>',
        re.S,
    )
    document, count = pattern.subn("", document)
    if count not in (0, 1):
        raise RuntimeError("Unexpected legacy endpoint phenotype table count")
    return document


def fmt(value: float | int | None, digits: int = 4, *, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    prefix = "+" if signed and float(value) > 0 else ""
    return f"{prefix}{float(value):.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def holm_adjusted_pvalues(pvalues: list[float]) -> list[float]:
    """Return monotone Holm family-wise adjusted p-values in input order."""
    order = sorted(range(len(pvalues)), key=lambda index: float(pvalues[index]))
    adjusted = [1.0] * len(pvalues)
    running_max = 0.0
    family_size = len(pvalues)
    for rank, index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(pvalues[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def ci_text(
    row: dict[str, Any],
    *,
    mean: str = "mean",
    low: str = "ci95_low",
    high: str = "ci95_high",
    digits: int = 4,
) -> str:
    return (
        f"{fmt(row[mean], digits)} [{fmt(row[low], digits)}, {fmt(row[high], digits)}]"
    )


def table(
    headers: list[str], rows: Iterable[Iterable[str]], *, classes: str = "paper-table"
) -> str:
    """Render a table body; the final report pass makes it collapsible."""
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-scroll"><table class="{classes}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details_table(
    title: str,
    headers: list[str],
    rows: Iterable[Iterable[str]],
    *,
    opened: bool = False,
) -> str:
    opened_attr = " open" if opened else ""
    rendered = list(rows)
    return (
        f'<details class="data-table"{opened_attr}>'
        f"<summary>{html.escape(title)} · {len(rendered)} rows · 点击展开</summary>"
        f"{table(headers, rendered)}"
        "</details>"
    )


def make_all_tables_collapsible(document: str) -> str:
    """Wrap every table not already inside a native ``details`` disclosure.

    Some inherited V4.4 tables are literal HTML rather than calls to ``table``.
    This final pass covers both sources while avoiding nested disclosures.
    """

    pattern = re.compile(
        r'<div class="table-scroll"><table\b.*?</table></div>', re.S
    )
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(document):
        pieces.append(document[cursor : match.start()])
        prefix = document[: match.start()]
        inside_details = len(re.findall(r"<details\b", prefix)) > len(
            re.findall(r"</details>", prefix)
        )
        block = match.group(0)
        if inside_details:
            pieces.append(block)
        else:
            row_count = max(0, len(re.findall(r"<tr\b", block)) - 1)
            pieces.append(
                '<details class="data-table">'
                f'<summary>数据表 · {row_count} rows · 点击展开</summary>'
                f"{block}</details>"
            )
        cursor = match.end()
    pieces.append(document[cursor:])
    return "".join(pieces)


def make_secondary_content_collapsible(document: str) -> str:
    """Collapse secondary analyses while leaving each heading and claim discoverable."""

    secondary_subsections = {
        "2.1 ": "展开：Transformer对象与操作词典",
        "4.5 ": "展开：cue-removal 表征敏感性",
        "5.2 ": "展开：outside-context mask 的负面/范围受限结果",
    }
    for prefix, summary in secondary_subsections.items():
        pattern = re.compile(
            rf'(<h3[^>]*>{re.escape(prefix)}.*?</h3>)(.*?)(?=<h3\b|</section>)',
            re.S,
        )
        document, count = pattern.subn(
            lambda match: (
                match.group(1)
                + f'<details class="secondary-analysis"><summary>{summary}</summary>'
                + '<div class="secondary-analysis-body">'
                + match.group(2)
                + "</div></details>"
            ),
            document,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"Could not collapse secondary subsection {prefix}")

    limits_pattern = re.compile(
        r'(<section id="limits">\s*<h2>12\s*·.*?</h2>)(.*?)(</section>)', re.S
    )
    document, count = limits_pattern.subn(
        lambda match: (
            match.group(1)
            + '<details class="secondary-section"><summary>展开：复现路径、审计文件与 source ledger</summary>'
            + '<div class="secondary-section-body">'
            + match.group(2)
            + "</div></details>"
            + match.group(3)
        ),
        document,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not collapse reproduction section body")
    return document


def apply_publication_editorial_pass(document: str) -> str:
    """Turn the evidence-complete report into a paper-facing main narrative.

    Earlier revisions intentionally exposed every diagnostic view.  The final
    report instead follows the inferential order mechanism -> representation ->
    causal identification, while preserving machine-readable source artifacts.
    This pass removes duplicate/exploratory views, moves all residual-subspace
    interventions into the causal chapter, and adds explicit claim boundaries.
    """

    def section(section_id: str) -> str:
        match = re.search(
            rf'<section id="{re.escape(section_id)}".*?</section>',
            document,
            flags=re.S,
        )
        if match is None:
            raise RuntimeError(f"Publication pass could not find section {section_id}")
        return match.group(0)

    def replace_fragment(old: str, new: str) -> None:
        nonlocal document
        if document.count(old) != 1:
            raise RuntimeError("Publication pass expected one section fragment")
        document = document.replace(old, new, 1)

    # The animated centroid trajectory is the paper-facing prompt figure.  The
    # second projector repeats the same evidence with more UI controls and the
    # answer projector repeats the layerwise rank/decoder evidence in Section 4.
    for label, pattern in (
        (
            "prompt seed-level projector",
            r'\s*<div class="figure-block"><h3>3\.2 .*?</figure></div>',
        ),
        (
            "answer seed-level projector",
            r'\s*<div class="figure-block"><h3>6\.3 .*?</figure></div>',
        ),
    ):
        document, count = re.subn(pattern, "", document, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"Could not remove duplicate {label}")
    document = document.replace(
        "makeMechanismWalkthrough();makeRunningIndex();makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();",
        "makeMechanismWalkthrough();makeRunningIndex();makeJoint();",
        1,
    )

    # The all-head heatmap is a discovery artifact, not an inferential result.
    # Keep the score definition and frozen banks in the main text; the causal
    # validation remains the matched-control nested-K curve in Section 8.
    attention = section("attention")
    atlas_marker = "\n<p>Atlas 的每一个格子"
    atlas_start = attention.find(atlas_marker)
    if atlas_start < 0:
        raise RuntimeError("Could not locate the exploratory all-head atlas")
    attention = attention[:atlas_start]
    attention = attention.replace(
        "<strong>可视化。</strong><span>下方只保留 full-span atlas：横轴是 layer，纵轴是 head，颜色编码 seed-aggregated broad score。颜色只用于发现候选，功能性必须由第8节的 layer-matched ablation 验证。</span>",
        "<strong>呈现范围。</strong><span>正文保留分数定义、冻结规则与进入确认实验的 head sets；逐 head atlas 属于 discovery 审计，不作为结果图重复展示。功能性只由第8节的 layer-matched ablation 判定。</span>",
    )
    attention += """
<div class="compact-evidence-strip"><strong>冻结候选。</strong>全层行为排序的 Qwen top-4 为 L27H18、L28H19、L23H29、L23H13；为避免把 L28 writer 同时计作 upstream source，Qwen 路径实验在 L&lt;28 内重新冻结 L27H18、L23H29、L23H13、L23H28。Gemma 冻结 top-2 为 L29H4、L35H2。完整逐头 discovery atlas 保留在机器可读 attention artifact 中，不进入确认性主图。</div>
<div class="conclusion"><strong>第7节结论。</strong>full-span attention 只承担 outcome-blind 候选发现：它表明哪些 heads 广泛读取 needle spans，但不提供因果结论。后续所有“功能 bank”主张都以独立 seeds 上的 layer-matched ablation、patching 或 mediation 为依据。</div>
</section>"""
    replace_fragment(section("attention"), attention)

    # State formation ends with input-level evidence.  Residual removal,
    # transport, and map-stability tests are interventions and therefore belong
    # in the causal-identification chapter, after the common estimand definitions.
    formation = section("formation-tests")
    subspace_start = formation.find("<h3>5.4 ")
    if subspace_start < 0:
        raise RuntimeError("Could not locate the subspace causal block")
    subspace = formation[subspace_start : -len("</section>")]
    formation = formation[:subspace_start] + """
<div class="conclusion"><strong>第5节结论。</strong>两模型都在 needle endpoint 聚合更早目标记录，且替换真实 needle 内容会造成远大于等预算普通文本替换的行为损伤。由此可得“needle evidence 在 prompt 中被整合并且是必要输入”；尚不能仅凭 attention 或输入 corruption 指定唯一内部 carrier。</div>
</section>"""
    replace_fragment(section("formation-tests"), formation)

    for caption in (
        "图 5C · Set-wide prompt count-subspace removal",
        "Figure · Adjacent-layer transport-aligned dose response",
        "Figure · Absolute answer-query damage",
        "Figure · Predictive map stability",
    ):
        pattern = re.compile(
            r'<figure\b(?:(?!<figure\b).)*?' + re.escape(caption) + r'.*?</figure>',
            re.S,
        )
        subspace, count = pattern.subn("", subspace, count=1)
        if count != 1:
            raise RuntimeError(f"Could not remove redundant causal figure: {caption}")
    subspace = re.sub(
        r'<details class="data-table"[^>]*>.*?</details>', "", subspace, flags=re.S
    )
    subspace = re.sub(r'<div class="figure-grid[^>]*>\s*</div>', "", subspace)
    subspace = re.sub(r'(<h[345][^>]*>)5\.4', r'\g<1>8.3', subspace)
    subspace = subspace.replace("§5.4", "§8.3").replace("5.4D 的", "8.3D 的")

    causal = section("causal")
    causal = causal.replace(
        "<h2>8 · 因果实验：先定位功能 bank，再验证可执行 answer state</h2>",
        "<h2>8 · 因果识别：必要性、充分性与路径中介</h2>",
        1,
    )
    causal = causal.replace(
        "<h3>8.1 这部分只回答两个问题</h3>",
        """<h3>8.1 因果证据链与判定顺序</h3>
<p>本章依次检验四类主张：固定 count plane 是否必要、跨层 transport direction 是否足以搬运 count、discovery-ranked retrieval bank 是否具有 matched-control-specific 功能，以及 answer-query state 是否足以改变数字。相似性、可解码性或 attention 排名均不单独计作因果证据。</p>""",
        1,
    )
    causal = causal.replace(
        "<h3>8.2 Effect 的逐样本定义与跨 seed 聚合</h3>",
        """<div class="conclusion"><strong>8.1 结论。</strong>本章采用递进而非互换的因果语言：removal 检验测试 component 的必要性，transport/patch 检验方向或 state 的充分性，mediation 检验 donor effect 是否经过指定路径；任一单项 positive 都不推出唯一机制。</div>
<h3>8.2 统一效应定义、统计单位与多重比较</h3>""",
        1,
    )
    causal = causal.replace(
        "<h3>8.3 Full-span Top-K ablation：检索 bank 是否具有因果功能？</h3>",
        """<div class="conclusion"><strong>因果判定口径。</strong>所有主效应都先在 seed 内聚合，再以 seed 为推断单位；candidate 必须超过 outcome-blind matched control，并在预先定义的比较 family 内通过校正。正结果支持测试方向的必要性或充分性，不能自动升级为唯一通道。</div>

"""
        + subspace
        + "\n<h3>8.4 Full-span Top-K ablation：检索 bank 是否具有因果功能？</h3>",
        1,
    )
    causal = causal.replace(
        "<h3>8.4 Answer-query patching：all samples 与 correct-only</h3>",
        "<h3>8.5 Answer-query patching：all samples 与 correct-only</h3>",
        1,
    )
    causal = causal.replace(
        "<h4>8.3B · 完整结果图、统计量与边界对照</h4>",
        "<h5>8.3B-1 · 确认性效应与边界</h5>",
        1,
    )
    replace_fragment(section("causal"), causal)

    # The extension memo audit duplicated the same claims and numbers after the
    # synthesis.  The reproducibility ledger remains, but the paper main text
    # now ends its argument once, in the synthesis and limitations sections.
    audit_match = re.search(
        r'<section id="question-audit".*?</section>', document, flags=re.S
    )
    if audit_match is not None:
        document = document[: audit_match.start()] + document[audit_match.end() :]

    document = document.replace("<h2>12 · 复现信息</h2>", "<h2>11 · 复现与证据边界</h2>", 1)
    document = document.replace(
        "<h3>6.4 Prompt 与 answer counter 的共同坐标</h3>",
        "<h3>6.3 Prompt 与 answer counter 的共同坐标</h3>",
        1,
    )
    document = document.replace(
        "<figcaption>Role-centered joint geometry. 圆点/实线是 prompt needle-end centroid trajectory，方块/虚线是 answer-query trajectory；同颜色细线只连接相同 count。三个轴是 shared PC scores。图下的 linear CKA、centroid-distance correlation 和 adjacent-step cosine 均在 full residual space 计算，不依赖 PCA 旋转。</figcaption>",
        "<figcaption><strong>图 6B · Role-centered joint geometry。</strong>横轴、纵轴与深度轴是 pooled role-centered PCA 的 shared PC scores；圆点/实线为 prompt needle-end centroid trajectory，方块/虚线为 answer-query trajectory，同颜色细线只连接相同 count。图下的 linear CKA、centroid-distance correlation 和 adjacent-step cosine 均在 full residual space 计算，不依赖 PCA 旋转；因此三维视图只展示全空间关系的投影。</figcaption>",
        1,
    )

    mechanism = section("mechanism-overview")
    mechanism = mechanism.replace(
        '<article class="model-mechanism qwen">',
        """<div class="conclusion"><strong>定义结论。</strong>本文把 state 定义为指定 token 与 post-block layer 的 residual vector，把 count direction 定义为只用 discovery 数据拟合并冻结的统计方向。后续结论因此指向“状态中的可读/可干预分量”，不把一个 token 或一个 PCA 轴写成离散整数寄存器。</div>
<article class="model-mechanism qwen">""",
        1,
    )
    mechanism = mechanism.replace(
        '<article class="model-mechanism gemma">',
        """<div class="conclusion"><strong>Qwen 局部结论。</strong>Qwen 的证据链把 full-span-ranked early source bank 接到 L28 H16/H19，并进一步确认这组 heads 通过自身 <em>W</em><sub>O</sub> 写入、在 L29–L35 传播并影响最终数字；单个 head 的独立充分性不由 set-level 结果保证。</div>
<article class="model-mechanism gemma">""",
        1,
    )
    mechanism = mechanism[:-len("</section>")] + """
<div class="conclusion"><strong>机制总述。</strong>两模型共享“prompt-side running state → distributed retrieval → residual-space rewrite → executable answer state”的计算链；Qwen 的写入定位到 L28 H16/H19，Gemma 的写入定位到 L29H4/L35H2→L37→L41。该总述是后文 representation、necessity/sufficiency 与 mediation 证据的合取，而不是由机制示意图本身推出。</div>
</section>"""
    replace_fragment(section("mechanism-overview"), mechanism)

    # Paper-facing navigation follows the argument rather than every diagnostic.
    nav = """<nav><a href="#mechanism-overview">Mechanism claim</a><a href="#scope">Conclusions</a><a href="#methods">Design &amp; estimands</a><a href="#prompt">Prompt representation</a><a href="#representation-extension">Representation audit</a><a href="#formation-tests">State formation</a><a href="#answer">Answer state</a><a href="#attention">Retrieval definition</a><a href="#causal">Causal identification</a><a href="#natural-ov">Write &amp; mediation</a><a href="#synthesis">Synthesis</a><a href="#limits">Limits &amp; reproducibility</a><a href="#appendix-first-locator">Appendix A</a></nav>"""
    document, nav_count = re.subn(r"<nav>.*?</nav>", nav, document, count=1, flags=re.S)
    if nav_count != 1:
        raise RuntimeError("Could not install paper-facing navigation")
    return document


def apply_paper_math_typesetting(document: str) -> str:
    """Mark mathematical code spans so prose code and equations render differently.

    The legacy report used ``<code>`` for both implementation identifiers and
    mathematical notation.  That made equations monospaced and gave every
    variable a colored code-chip background.  This pass keeps literal tokens
    such as ``Total:`` and artifact names as code, while converting expressions,
    indexed variables, Greek symbols, and one-letter mathematical variables to
    semantic inline-math spans.  Display equations already carry the dedicated
    ``equation`` or ``formula-line`` class and are styled separately in CSS.
    """

    mathematical_symbols = set(
        "=Σ∑∈≈≠≤≥←→±−·Δδβαℓεμρ√∥⟨⟩"
    )
    single_symbol = re.compile(r"[A-Za-zα-ωΑ-Ωℓ]")

    def replace_code(match: re.Match[str]) -> str:
        contents = match.group(1)
        plain = html.unescape(re.sub(r"<[^>]+>", "", contents)).strip()
        is_math = (
            "<sub>" in contents
            or "<sup>" in contents
            or any(symbol in plain for symbol in mathematical_symbols)
            or single_symbol.fullmatch(plain) is not None
        )
        if not is_math:
            return match.group(0)
        return f'<span class="math-inline">{contents}</span>'

    head, marker, tail = document.partition("<script>")
    if not marker:
        raise RuntimeError("Paper math pass could not locate the report script")
    head = re.sub(r"<code>(.*?)</code>", replace_code, head, flags=re.S)
    return head + marker + tail


def model_coverage_matrix() -> str:
    """Machine-readable editorial audit: every core question covers both models."""

    rows = [
        ("Prompt/answer geometry", "rank、PCA、ridge、固定分类器", "Qwen全层", "Gemma全层", "同定义"),
        ("Prompt noise", "balanced count×prompt decomposition", "Qwen全层", "Gemma全层", "同定义"),
        ("Token-role gate", "冻结endpoint basis投影全部token", "Qwen L8", "Gemma L9", "同定义"),
        ("Cue removal", "present/absent共享basis", "prompt+answer", "prompt+answer", "同定义"),
        ("Earlier-span", "top-10 full-span minus matched ordinary", "10 heads×10 seeds", "10 heads×10 seeds", "同estimand；Gemma仅改为单query-row重建"),
        ("Endpoint key restriction", "clean/needle-only/matched ordinary", "Qwen L8", "Gemma L9", "同定义"),
        ("Needle corruption", "等token-budget输入替换", "all+correct-only", "all+correct-only", "同定义"),
        ("Fixed prompt plane", "rank-3 removal vs orthogonal", "Qwen L8", "Gemma L9", "同定义"),
        ("Rotating transport", "aligned vs orthogonal；1×/2×", "L28→L29", "L36→L37", "同因果问题，层位模型特异"),
        ("Broad retrieval", "full-span frozen top-K ablation", "K=1–32", "K=1–32", "同定义"),
        ("Executable answer state", "donor→receiver residual patch", "all+correct-only", "all+correct-only", "同定义"),
        ("Write/mediation", "pre-O/residual write与下游采用", "L28 H16/H19", "K2→L37→L41", "同机制问题，边界随架构改变"),
        ("Answer error", "|generated count−gold count|", "全部样本", "全部样本", "同定义"),
    ]
    rendered = table(
        ["核心问题", "实验/estimand", "Qwen", "Gemma", "可比性"],
        rows,
        classes="paper-table compact-result-table model-coverage-table",
    )
    return f"""
<details class="secondary-analysis model-coverage" open>
<summary>双模型覆盖审计：核心问题是否同时有 Qwen 与 Gemma 结果？</summary>
<div class="secondary-analysis-body">
<p>下表逐项检查正文的核心实验。<strong>“同定义”</strong>表示样本单位、effect与推断口径一致；<strong>“同机制问题”</strong>表示科学问题相同，但由于Gemma的sliding/full-attention交替结构，具体层或hook边界不能强制与Qwen相同。没有用一个模型的结果替代另一个模型。</p>
{rendered}
<div class="conclusion"><strong>覆盖审计结论。</strong>所有核心representation、formation、retrieval、causal execution与error实验均同时报告Qwen和Gemma。唯一不能逐head镜像的是最终写入路径：Qwen定位为局部H16/H19 pre-O写入，Gemma定位为distributed K2→L37→L41 residual path；这是一项明确报告的架构差异，不是缺失数据。</div>
</div></details>
"""


def merge_transport_into_section_5_4(document: str) -> str:
    """Remove the standalone transport chapter and close the numbering gap."""

    document, nav_count = re.subn(
        r'<a href="#transport-subspace">.*?</a>', "", document, count=1
    )
    if nav_count != 1:
        raise RuntimeError("Could not remove the standalone transport nav entry")

    # The former chapter 7 now lives inside 5.4B.  Later visible chapter and
    # subsection numbers therefore shift down by one, while stable HTML ids do
    # not change.
    def heading_number(match: re.Match[str]) -> str:
        return match.group(1) + str(int(match.group(2)) - 1)

    document = re.sub(
        r'(<h[23][^>]*>)(13|12|11|10|9|8)(?=(?:\.\d+)?(?:\s|·))',
        heading_number,
        document,
    )

    document = document.replace("第7节", "本小节下方").replace(
        "第 7 节", "本小节下方"
    )

    def section_reference(match: re.Match[str]) -> str:
        major = int(match.group(1))
        suffix = match.group(2) or ""
        if major == 7:
            return "§5.4B" + suffix
        return "§" + str(major - 1) + suffix

    document = re.sub(
        r'§(13|12|11|10|9|8|7)(\.\d+)?', section_reference, document
    )

    def chinese_section_reference(match: re.Match[str]) -> str:
        major = int(match.group(1))
        suffix = match.group(2) or ""
        return f"第{major - 1}{suffix}节"

    document = re.sub(
        r'第\s*(13|12|11|10|9|8)\s*(\.\d+)?\s*节',
        chinese_section_reference,
        document,
    )
    return document


def insert_concrete_examples(document: str) -> str:
    """Place a worked example immediately after every study rationale."""

    examples = [
        "一条 N=10 prompt 读到第4条目标记录时，我们取第4条记录最后一个 token 的 residual；若它比第3条更靠近 count=4 的冻结中心、比第5条更远，就把这一步视为 running-index 表征的一个具体观测。",
        "即使 count=1…10 的十个中心几乎落在一条三维曲线上，同一个 count=6 在30个不同 prompt 中仍可能沿几十个其他方向散开；前者是低维信号，后者是上下文背景。",
        "例如分类器只在 seeds 1–15 的 state 上学习，随后面对完全未见过文本的 seed 16；若把其中 gold count=7 的 answer state 判为7，才算 held-out 可解码，而不是记住某个 prompt。",
        "设 seed A 的所有 endpoint states 都整体向方向 u 平移，这是 seed/context 主效应；若 seed A 的 n=1→10 曲线比 seed B 更陡或更弯，则额外部分属于 count×seed interaction。",
        "在句子“Reno has a score of 7.”中，我们分别投影内部 token“score”和记录末端 token“.”；若只有末端随它是第几条目标记录而移动，就说明 count state 是 endpoint-gated。",
        "对同一段 passage 做成一对输入：A 保留开头的计数说明，B 只删除这两句；比较两者 count=6 中心相对 count=5、7 的位置是否保持。",
        "当 query 位于第6个 needle 的末端时，候选 head 若主要回看前5个 needle 的完整句子，而不是同长度、同深度的普通文本，就符合 earlier-span aggregation。",
        "在第6个 endpoint，我们让它只能读取已出现的6段 needle；另一个等预算条件只能读取6段位置匹配的普通文本。若 needle-only 反而更差，ordinary context 就不是可直接删掉的纯噪声。",
        "原始输入中的“Reno has a score of 7.”会被等 token 数的普通 passage 片段替换，而不是删除；控制条件则只替换另一段普通文本，因此两者的长度、query 位置和改动 token 数完全相同。",
        "以 gold count=6 为例，模型已有6个 active endpoint states。实验同时从这6个向量中删除冻结 rank-3 count component，再与删除等范数正交 component 比较最终答案，而不是只改最后一个 endpoint。",
        "gold=8 时，预测7的 absolute deviation 是1，预测3则是5；accuracy 都记为错误，但后者说明内部计数偏离更严重。",
        "例如 prompt endpoint 的 count=5 方向不必与 answer query 的 count=5 方向平行；只要把 count=3 donor 的 answer state patch 给 count=8 receiver 后，receiver 沿可执行输出方向转向3，就说明 answer state 已完成重新编码。",
        "如果 all-fit PCA 把 count=8 与9分开，而只用答对 discovery rows 拟合后，同一批 V4.4 states 的8/9中心距离仍几乎不变，就说明几何不是由错误样本人为制造。",
        "对 receiver=1、donor=2，我们只注入 source transport basis 中的1-count差；若下一层沿1→2 centroid chord 前进约1单位，而等范数正交注入接近0，就说明该局部 subspace 能传递 count。",
        "若一个 answer-query head 对每条 active needle 的整句都分配质量，它的 full-span score 会高；只盯每句最后一个 token 的 endpoint-key score则可能漏掉这种分布式读取。",
        "例如 K=4 时只将冻结排名前4的 heads 在 Total: query 的 pre-O 输出置零；若 absolute error 比删除同层4个随机 heads 增加更多，这才是检索 bank 的特异因果效应。",
        "receiver 的正确答案为3、donor 的正确答案为8；把 donor 的 answer-query residual patch 到 receiver 后，若输出概率从3移向8，就是 donor adoption，而不只是 hidden state 变近。",
        "receiver 当前倾向输出3。若在某个 head 的 pre-O z 加入一个自然的+1 count step，经过这个 head 自己的 W_O 后，答案期望值向4移动；这才说明该 head 的 OV path 能把读到的 count 写进 residual。",
        "对 Qwen receiver=3，我们在 L28 H16/H19 的 pre-O z 中加入 +1 个自然 count step；只有经过这两个 heads 自己的 W_O 后，答案期望值也向4移动，才算 OV 写入充分性。",
        "对 Gemma receiver=3、donor=8，我们只替换 L29H4/L35H2 的 pre-O z；若 L37 出现 donor-aligned residual，删除这段 induced residual 又让输出退回3附近，就构成 source→mediator→answer 的具体路径。",
    ]
    pattern = re.compile(r'<div class="study-preface">.*?</div>', re.S)
    blocks = list(pattern.finditer(document))
    if len(blocks) != len(examples):
        raise RuntimeError(
            f"Study-preface/example mismatch: {len(blocks)} blocks vs {len(examples)} examples"
        )
    pieces: list[str] = []
    cursor = 0
    for match, example in zip(blocks, examples):
        block = match.group(0)
        if "具体例子。" not in block:
            first_span_end = block.find("</span>")
            if first_span_end < 0:
                raise RuntimeError("Study preface has no first explanatory span")
            insertion = (
                "<strong>具体例子。</strong><span>" + example + "</span>"
            )
            block = (
                block[: first_span_end + len("</span>")]
                + insertion
                + block[first_span_end + len("</span>") :]
            )
        pieces.append(document[cursor : match.start()])
        pieces.append(block)
        cursor = match.end()
    pieces.append(document[cursor:])
    return "".join(pieces)


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not math.isfinite(low) or not math.isfinite(high):
        return [0.0]
    if low == high:
        return [low]
    span = high - low
    raw = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(abs(raw)))
    normalized = raw / power
    if normalized <= 1:
        step = 1 * power
    elif normalized <= 2:
        step = 2 * power
    elif normalized <= 5:
        step = 5 * power
    else:
        step = 10 * power
    start = math.floor(low / step) * step
    stop = math.ceil(high / step) * step
    ticks: list[float] = []
    value = start
    while value <= stop + step * 0.25 and len(ticks) < 20:
        ticks.append(value)
        value += step
    return ticks


def forest_svg(
    rows: list[dict[str, Any]],
    *,
    title: str,
    description: str,
    x_label: str,
    width: int = 1180,
    left: int = 310,
    right: int = 300,
    zero: float = 0.0,
    colors: tuple[str, ...] = ("#6750E8", "#00D4B4", "#FF5FA2", "#F6E36A"),
) -> str:
    top, bottom = 44, 74
    row_h = 54
    height = top + bottom + row_h * len(rows)
    svg_slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "effect"
    title_id = f"forest-{svg_slug}-title"
    desc_id = f"forest-{svg_slug}-desc"
    values = [zero]
    for row in rows:
        values.extend([float(row["low"]), float(row["high"]), float(row["mean"])])
    low, high = min(values), max(values)
    pad = max((high - low) * 0.12, 0.01)
    low, high = low - pad, high + pad
    plot_w = width - left - right

    def x(value: float) -> float:
        return left + (value - low) / (high - low) * plot_w

    parts = [
        f'<svg class="stat-svg integrated-forest" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id} {desc_id}">',
        f'<title id="{title_id}">{html.escape(title)}</title>',
        f'<desc id="{desc_id}">{html.escape(description)}</desc>',
    ]
    ticks = _nice_ticks(low, high, 6)
    y_axis = top + row_h * len(rows)
    for tick in ticks:
        if tick < low - 1e-12 or tick > high + 1e-12:
            continue
        tx = x(tick)
        parts.append(
            f'<line class="grid" x1="{tx:.1f}" y1="{top - 12}" x2="{tx:.1f}" y2="{y_axis}"/>'
        )
        parts.append(
            f'<text class="tick" x="{tx:.1f}" y="{y_axis + 22}" text-anchor="middle">{fmt(tick, 3)}</text>'
        )
    zx = x(zero)
    parts.append(
        f'<line class="zero" x1="{zx:.1f}" y1="{top - 14}" x2="{zx:.1f}" y2="{y_axis}"/>'
    )
    for idx, row in enumerate(rows):
        cy = top + idx * row_h + row_h / 2
        lo, hi, mean = float(row["low"]), float(row["high"]), float(row["mean"])
        color = str(row.get("color") or colors[idx % len(colors)])
        parts.append(
            f'<text class="row-label" x="{left - 18}" y="{cy + 4:.1f}" text-anchor="end">{html.escape(str(row["label"]))}</text>'
        )
        parts.append(
            f'<line class="ci" x1="{x(lo):.1f}" y1="{cy:.1f}" x2="{x(hi):.1f}" y2="{cy:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<line class="cap" x1="{x(lo):.1f}" y1="{cy - 6:.1f}" x2="{x(lo):.1f}" y2="{cy + 6:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<line class="cap" x1="{x(hi):.1f}" y1="{cy - 6:.1f}" x2="{x(hi):.1f}" y2="{cy + 6:.1f}" style="stroke:{color}"/>'
        )
        parts.append(
            f'<circle class="dot" cx="{x(mean):.1f}" cy="{cy:.1f}" r="6" style="fill:{color}"/>'
        )
        parts.append(
            f'<text class="value-label" x="{x(hi) + 10:.1f}" y="{cy + 4:.1f}">{html.escape(str(row.get("value", fmt(mean, 4))))}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 14}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def write_trace_svg(
    rows: list[dict[str, Any]],
    *,
    id_prefix: str = "write",
    title: str = "L28 natural OV write propagates through L35",
    description: str = "Layer is on the horizontal axis. Natural-minus-orthogonal count-axis coefficient is on the vertical axis. Points are seed means and bars are 95 percent bootstrap confidence intervals.",
) -> str:
    width, height = 1040, 500
    left, right, top, bottom = 90, 38, 36, 78
    plot_w, plot_h = width - left - right, height - top - bottom
    layers = [int(row["layer"]) for row in rows]
    raw_low = min(0.0, min(float(row["ci95_low"]) for row in rows))
    raw_high = max(0.0, max(float(row["ci95_high"]) for row in rows))
    span = max(raw_high - raw_low, 1e-6)
    ymin = raw_low - (0.10 * span if raw_low < 0 else 0.0)
    ymax = raw_high + (0.18 * span if raw_high > 0 else 0.10 * span)

    def x(layer: int) -> float:
        return left + (layer - min(layers)) / max(max(layers) - min(layers), 1) * plot_w

    def y(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * plot_h

    parts = [
        f'<svg class="stat-svg write-trace" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
        f'<title id="{id_prefix}-title">{html.escape(title)}</title>',
        f'<desc id="{id_prefix}-desc">{html.escape(description)}</desc>',
    ]
    for tick in _nice_ticks(ymin, ymax, 6):
        if tick < ymin or tick > ymax:
            continue
        yy = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end">{fmt(tick, 3)}</text>'
        )
    if ymin < 0 < ymax:
        zero_y = y(0.0)
        parts.append(
            f'<line class="zero" x1="{left}" y1="{zero_y:.1f}" x2="{width - right}" y2="{zero_y:.1f}"/>'
        )
    path = " ".join(
        ("M" if idx == 0 else "L")
        + f" {x(int(row['layer'])):.1f} {y(float(row['mean'])):.1f}"
        for idx, row in enumerate(rows)
    )
    parts.append(f'<path class="trace-line" d="{path}"/>')
    for row in rows:
        xx = x(int(row["layer"]))
        yy = y(float(row["mean"]))
        lo_y, hi_y = y(float(row["ci95_low"])), y(float(row["ci95_high"]))
        parts.append(
            f'<line class="trace-ci" x1="{xx:.1f}" y1="{hi_y:.1f}" x2="{xx:.1f}" y2="{lo_y:.1f}"/>'
        )
        parts.append(f'<circle class="trace-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="6"/>')
        parts.append(
            f'<text class="value-label" x="{xx:.1f}" y="{yy - 12:.1f}" text-anchor="middle">{fmt(float(row["mean"]), 3)}</text>'
        )
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 24}" text-anchor="middle">L{int(row["layer"])}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle">decoder layer</text>'
    )
    parts.append(
        f'<text class="axis-label" transform="translate(22 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">natural − orthogonal count-axis coefficient</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def evidence_gate_svg(
    families: list[dict[str, Any]],
    *,
    id_prefix: str = "gate",
    title: str = "Four preregistered natural-OV evidence gates",
    description: str = (
        "Four boxes summarize natural signal, true pre-O sufficiency, centered "
        "z-space necessity, and path mediation. A check or cross marks the "
        "family-level decision; the global intersection-union p value is the "
        "largest family p value."
    ),
) -> str:
    width, height = 1040, 470
    positions = [(28, 38), (530, 38), (28, 246), (530, 246)]
    parts = [
        f'<svg class="stat-svg gate-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{id_prefix}-title {id_prefix}-desc">',
        f'<title id="{id_prefix}-title">{html.escape(title)}</title>',
        f'<desc id="{id_prefix}-desc">{html.escape(description)}</desc>',
    ]
    for idx, family in enumerate(families):
        x, y = positions[idx]
        passed = bool(family.get("passed", True))
        status_class = "gate-pass" if passed else "gate-fail"
        parts.append(
            f'<rect class="gate-box {status_class}" x="{x}" y="{y}" width="482" height="174" rx="8"/>'
        )
        parts.append(
            f'<circle class="gate-check {status_class}" cx="{x + 34}" cy="{y + 34}" r="15"/>'
        )
        parts.append(
            f'<text class="gate-check-text" x="{x + 34}" y="{y + 39}" text-anchor="middle">{"✓" if passed else "×"}</text>'
        )
        parts.append(
            f'<text class="gate-heading" x="{x + 60}" y="{y + 40}">{html.escape(family["title"])}</text>'
        )
        parts.append(
            f'<text class="gate-main" x="{x + 24}" y="{y + 83}">{html.escape(family["main"])}</text>'
        )
        parts.append(
            f'<text class="gate-sub" x="{x + 24}" y="{y + 112}">{html.escape(family["sub"])}</text>'
        )
        parts.append(
            f'<text class="gate-p" x="{x + 24}" y="{y + 145}">{html.escape(family["p"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def transport_dose_svg(condition_rows: list[dict[str, str]]) -> str:
    """Two-panel dose plot for the discovery-frozen adjacent-layer relay test."""
    width, height = 1120, 390
    left, panel_w, gap, top, bottom = 76, 430, 105, 52, 76
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00D4B4"}
    labels = {
        "matched_orthogonal": "orthogonal\ncontrol",
        "aligned_dose_1": "aligned\ndose 1",
        "aligned_dose_2": "aligned\ndose 2",
    }
    order = ["matched_orthogonal", "aligned_dose_1", "aligned_dose_2"]
    rows = [row for row in condition_rows if row["support"] == "answer_query_relay"]
    by_model = {
        model: {row["condition"]: float(row["mean_target_donor_fraction"]) for row in rows if row["model_label"] == model}
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    ymax = 2.05
    parts = [
        f'<svg class="stat-svg transport-dose-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="transport-dose-title transport-dose-desc">',
        '<title id="transport-dose-title">Direction-specific adjacent-layer count transport</title>',
        '<desc id="transport-dose-desc">For Qwen and Gemma, aligned dose one and dose two move the next-layer state toward the donor count, while an equal-norm orthogonal control does not.</desc>',
    ]
    for panel, model in enumerate(("Qwen3-8B", "Gemma4-E4B")):
        x0 = left + panel * (panel_w + gap)
        plot_h = height - top - bottom
        y = lambda value: top + plot_h * (1.0 - value / ymax)
        for tick in (0.0, 0.5, 1.0, 1.5, 2.0):
            yy = y(tick)
            parts.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0 + panel_w}" y2="{yy:.1f}"/>')
            if panel == 0:
                parts.append(f'<text class="tick" x="{x0 - 12}" y="{yy + 4:.1f}" text-anchor="end">{tick:.1f}</text>')
        parts.append(f'<text class="panel-title" x="{x0 + panel_w/2:.1f}" y="24" text-anchor="middle">{model}</text>')
        points = []
        for index, condition in enumerate(order):
            xx = x0 + 55 + index * (panel_w - 110) / 2
            yy = y(by_model[model][condition])
            points.append((xx, yy))
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="7" fill="{colors[model]}"/>')
            parts.append(f'<text class="bar-label" x="{xx:.1f}" y="{yy - 14:.1f}" text-anchor="middle">{by_model[model][condition]:.3f}</text>')
            for line_index, line in enumerate(labels[condition].split("\n")):
                parts.append(f'<text class="tick" x="{xx:.1f}" y="{height-bottom+24+line_index*15}" text-anchor="middle">{line}</text>')
        parts.append('<polyline points="' + " ".join(f"{x:.1f},{y_: .1f}" for x, y_ in points) + f'" fill="none" stroke="{colors[model]}" stroke-width="3"/>')
    parts.append(f'<text class="axis-label" transform="translate(18 {top + (height-top-bottom)/2:.1f}) rotate(-90)" text-anchor="middle">target-layer donor fraction</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_transport_aligned_section(
    condition_rows: list[dict[str, str]],
    contrast_rows: list[dict[str, str]],
) -> str:
    def contrast(model: str, name: str) -> dict[str, str]:
        return next(
            row for row in contrast_rows
            if row["model_label"] == model
            and row["support"] == "answer_query_relay"
            and row["contrast"] == name
            and row["metric"] == "target_donor_fraction"
        )

    rows = []
    for model, edge in (("Qwen3-8B", "L28 → L29"), ("Gemma4-E4B", "L36 → L37")):
        specificity = contrast(model, "aligned_dose_1_minus_orthogonal")
        dose = contrast(model, "dose_2_minus_dose_1")
        rows.append(
            [
                model,
                edge,
                f'{float(specificity["mean_contrast"]):.3f} '
                f'[{float(specificity["bootstrap_95ci_low"]):.3f}, {float(specificity["bootstrap_95ci_high"]):.3f}]',
                fmt_p(float(specificity["exact_seed_signflip_p_two_sided"])),
                f'{float(dose["mean_contrast"]):.3f} '
                f'[{float(dose["bootstrap_95ci_low"]):.3f}, {float(dose["bootstrap_95ci_high"]):.3f}]',
                fmt_p(float(dose["exact_seed_signflip_p_two_sided"])),
            ]
        )
    return f"""
<div class="transport-subspace-embedded" id="transport-subspace">
<h4>5.4B · 完整结果图、统计量与边界对照</h4>
<div class="study-preface"><strong>为什么做。</strong><span>跨层 PCA 轴不必保持同一个欧氏方向，因此直接把上一层的 PCA vector 塞进下一层可能失败。这里检验更精确的问题：answer-query residual 中是否存在一个从上游 count coordinates 映射到下一层 count coordinates 的方向特异通道。</span><strong>如何定义。</strong><span>只在 discovery count centroids 上，把 source residual 回归到 target 的 rank-3 count coordinates，再对回归权重的 row-space 做 QR，得到冻结的 source transport basis <code>B</code>。对 confirmation receiver count R 与 donor count D，注入 <code>Proj<sub>B</sub>(μ<sub>D</sub>−μ<sub>R</sub>)</code> 的 1×、2× dose；控制是在 <code>B</code> 正交补中、与 1× dose 等范数的方向。</span><strong>如何评估。</strong><span>target donor fraction 为干预后 target state 沿 frozen receiver→donor centroid chord 移动的比例；1 表示到达 donor centroid，0 表示未移动。每 seed 先平均 1→2、2→1、5→6、6→5 四对，再用10个 confirmation seeds做50,000次 bootstrap CI与双侧 exact sign-flip。</span></div>
<figure>{transport_dose_svg(condition_rows)}<figcaption><strong>Figure · Adjacent-layer transport-aligned dose response.</strong> 左、右面板分别是 Qwen L28→L29 与 Gemma L36→L37。横轴依次是等范数正交控制、transport-aligned 1×、2× dose；纵轴是下一层 target donor fraction。正交控制接近0，而 aligned dose 接近1并在2×时接近1.8；这同时检验方向特异性与近似剂量响应。</figcaption></figure>
{table(["model", "tested edge", "dose1 − orthogonal", "exact p", "dose2 − dose1", "exact p"], rows, classes="paper-table compact-result-table")}
<p><strong>边界对照。</strong>同一分析也把旧的“最后一个 prompt needle endpoint”当 source support；Gemma 的 target movement 为0，Qwen 仅约2×10<sup>−4</sup>。因此正结果定位的是<strong>同一 answer-query position 的局部 relay</strong>，不是“最后一个 prompt endpoint 会直接跳到 answer”。</p>
<div class="conclusion"><strong>本节结论</strong>Qwen 与 Gemma 的 answer-query residual 中都存在一个 discovery-frozen、方向特异且近似剂量响应的 count transport subspace，并能跨各自测试的相邻层传播（两模型的 specificity 与 dose-increment 均 exact p=0.001953）。这证明局部 residual relay 的可用性；它不声称所有相邻层共享同一静态 PCA basis，也不把 prompt endpoint 误写成直接 source。</div>
</div>
"""


SERIES_COLORS = (
    "#23165C", "#6750E8", "#00C2FF", "#00D4B4", "#39E58C", "#C04DFF", "#FF5FA2"
)


def layer_curve_svg(
    rows: list[dict[str, Any]],
    *,
    panels: Sequence[tuple[str, str, str]],
    series_key: str,
    value_key: str,
    series_order: Sequence[str],
    y_min: float,
    y_max: float,
    y_label: str,
    title: str,
    description: str,
) -> str:
    width, panel_h = 1160, 300
    panel_w, left, gap, top, bottom = 480, 76, 90, 44, 60
    height = panel_h + 72
    title_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    parts = [
        f'<svg class="stat-svg layer-curve-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}-title {title_id}-desc">',
        f'<title id="{title_id}-title">{html.escape(title)}</title>',
        f'<desc id="{title_id}-desc">{html.escape(description)}</desc>',
    ]
    for panel_index, (model, role, panel_title) in enumerate(panels):
        x0 = left + panel_index * (panel_w + gap)
        subset = [row for row in rows if str(row.get("model_label")) == model and str(row.get("role")) == role]
        layers = sorted({int(row["layer"]) for row in subset})
        if not layers:
            continue
        x = lambda layer: x0 + (int(layer) - layers[0]) / max(layers[-1] - layers[0], 1) * panel_w
        y = lambda value: top + (y_max - float(value)) / max(y_max - y_min, 1e-12) * (panel_h - top - bottom)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = y_min + fraction * (y_max - y_min)
            yy = y(value)
            parts.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0+panel_w}" y2="{yy:.1f}"/>')
            if panel_index == 0:
                parts.append(f'<text class="tick" x="{x0-10}" y="{yy+4:.1f}" text-anchor="end">{value:.2f}</text>')
        for tick in (layers[0], layers[len(layers)//2], layers[-1]):
            xx = x(tick)
            parts.append(f'<line class="axis" x1="{xx:.1f}" y1="{panel_h-bottom}" x2="{xx:.1f}" y2="{panel_h-bottom+5}"/>')
            parts.append(f'<text class="tick" x="{xx:.1f}" y="{panel_h-bottom+22}" text-anchor="middle">L{tick}</text>')
        parts.append(f'<text class="panel-title" x="{x0+panel_w/2:.1f}" y="24" text-anchor="middle">{html.escape(panel_title)}</text>')
        for series_index, series in enumerate(series_order):
            points = sorted(
                ((int(row["layer"]), float(row[value_key])) for row in subset if str(row.get(series_key)) == series and str(row.get(value_key, "")) not in {"", "nan", "NaN"}),
                key=lambda item: item[0],
            )
            if not points:
                continue
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            parts.append('<polyline points="' + " ".join(f"{x(layer):.1f},{y(value):.1f}" for layer, value in points) + f'" fill="none" stroke="{color}" stroke-width="2.3"/>')
        parts.append(f'<text class="axis-label" x="{x0+panel_w/2:.1f}" y="{panel_h+5}" text-anchor="middle">decoder layer</text>')
    legend_y = height - 25
    legend_x = 130
    for index, series in enumerate(series_order):
        x0 = legend_x + index * 155
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        parts.append(f'<line x1="{x0}" y1="{legend_y}" x2="{x0+25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{x0+32}" y="{legend_y+4}">{html.escape(series)}</text>')
    parts.append(f'<text class="axis-label" transform="translate(18 {top+(panel_h-top-bottom)/2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')
    parts.append('</svg>')
    return "".join(parts)


def layerwise_ci_curve_svg(
    rows: list[dict[str, Any]],
    *,
    panels: Sequence[tuple[str, str]],
    series_key: str,
    series_order: Sequence[str],
    series_labels: dict[str, str],
    x_key: str,
    layer_label_key: str,
    value_key: str,
    low_key: str,
    high_key: str,
    significance_key: str | None,
    y_label: str,
    title: str,
    description: str,
    include_zero: bool = True,
) -> str:
    """Draw two model panels with pointwise seed-bootstrap intervals.

    The x coordinate is normally normalized decoder depth, while the displayed
    ticks retain the actual registered layer or boundary.  A filled outer ring
    denotes a layer surviving the preregistered Holm family.
    """

    def finite(row: dict[str, Any], key: str) -> bool:
        try:
            return math.isfinite(float(row[key]))
        except (KeyError, TypeError, ValueError):
            return False

    def significant(row: dict[str, Any]) -> bool:
        if significance_key is None:
            return False
        return str(row.get(significance_key, "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }

    plotted = [
        row
        for row in rows
        if str(row.get("model_label")) in {panel[0] for panel in panels}
        and str(row.get(series_key)) in set(series_order)
        and finite(row, value_key)
        and finite(row, low_key)
        and finite(row, high_key)
        and finite(row, x_key)
    ]
    if not plotted:
        raise RuntimeError(f"No finite rows for layerwise curve: {title}")
    bounds = [float(row[key]) for row in plotted for key in (low_key, high_key)]
    if include_zero:
        bounds.append(0.0)
    raw_min, raw_max = min(bounds), max(bounds)
    span = max(raw_max - raw_min, max(abs(raw_min), abs(raw_max), 1.0) * 0.08)
    y_min = raw_min - 0.09 * span
    y_max = raw_max + 0.09 * span

    width, panel_h = 1160, 330
    panel_w, left, gap, top, bottom = 480, 78, 90, 42, 70
    height = panel_h + 76
    title_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "layerwise-ci"
    parts = [
        f'<svg class="stat-svg layer-curve-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="{title_id}-title {title_id}-desc">',
        f'<title id="{title_id}-title">{html.escape(title)}</title>',
        f'<desc id="{title_id}-desc">{html.escape(description)}</desc>',
    ]
    for panel_index, (model, panel_title) in enumerate(panels):
        x0 = left + panel_index * (panel_w + gap)
        subset = [row for row in plotted if str(row.get("model_label")) == model]
        x_values = [float(row[x_key]) for row in subset]
        x_min, x_max = min(x_values), max(x_values)
        x = lambda value: x0 + (float(value) - x_min) / max(x_max - x_min, 1e-12) * panel_w
        y = lambda value: top + (y_max - float(value)) / max(y_max - y_min, 1e-12) * (panel_h - top - bottom)
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = y_min + fraction * (y_max - y_min)
            yy = y(value)
            parts.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0+panel_w}" y2="{yy:.1f}"/>')
            if panel_index == 0:
                parts.append(f'<text class="tick" x="{x0-10}" y="{yy+4:.1f}" text-anchor="end">{value:.2f}</text>')
        if include_zero and y_min <= 0.0 <= y_max:
            yy = y(0.0)
            parts.append(f'<line class="axis" x1="{x0}" y1="{yy:.1f}" x2="{x0+panel_w}" y2="{yy:.1f}" stroke-dasharray="5 4"/>')
        representatives = sorted(
            subset,
            key=lambda row: (float(row[x_key]), str(row[layer_label_key])),
        )
        tick_rows = [
            representatives[0],
            representatives[len(representatives) // 2],
            representatives[-1],
        ]
        seen_ticks: set[tuple[float, str]] = set()
        for row in tick_rows:
            tick = (float(row[x_key]), str(row[layer_label_key]))
            if tick in seen_ticks:
                continue
            seen_ticks.add(tick)
            xx = x(tick[0])
            parts.append(f'<line class="axis" x1="{xx:.1f}" y1="{panel_h-bottom}" x2="{xx:.1f}" y2="{panel_h-bottom+5}"/>')
            parts.append(f'<text class="tick" x="{xx:.1f}" y="{panel_h-bottom+23}" text-anchor="middle">{html.escape(tick[1])}</text>')
        parts.append(f'<text class="panel-title" x="{x0+panel_w/2:.1f}" y="24" text-anchor="middle">{html.escape(panel_title)}</text>')
        for series_index, series in enumerate(series_order):
            series_rows = sorted(
                [row for row in subset if str(row.get(series_key)) == series],
                key=lambda row: float(row[x_key]),
            )
            if not series_rows:
                continue
            color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
            parts.append(
                '<polyline points="'
                + " ".join(
                    f'{x(row[x_key]):.1f},{y(row[value_key]):.1f}'
                    for row in series_rows
                )
                + f'" fill="none" stroke="{color}" stroke-width="2.4"/>'
            )
            for row in series_rows:
                xx = x(row[x_key])
                yy, y_low, y_high = y(row[value_key]), y(row[low_key]), y(row[high_key])
                parts.append(f'<line x1="{xx:.1f}" y1="{y_low:.1f}" x2="{xx:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="1.4" opacity="0.82"/>')
                parts.append(f'<line x1="{xx-3:.1f}" y1="{y_low:.1f}" x2="{xx+3:.1f}" y2="{y_low:.1f}" stroke="{color}"/>')
                parts.append(f'<line x1="{xx-3:.1f}" y1="{y_high:.1f}" x2="{xx+3:.1f}" y2="{y_high:.1f}" stroke="{color}"/>')
                if significant(row):
                    parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="6.2" fill="none" stroke="{color}" stroke-width="2"/>')
                parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.5" fill="{color}"/>')
        parts.append(f'<text class="axis-label" x="{x0+panel_w/2:.1f}" y="{panel_h+5}" text-anchor="middle">normalized decoder depth (ticks show registered layer)</text>')
    legend_y = height - 25
    legend_x = 105
    spacing = min(360, max(180, 880 // max(len(series_order), 1)))
    for index, series in enumerate(series_order):
        xx = legend_x + index * spacing
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        parts.append(f'<line x1="{xx}" y1="{legend_y}" x2="{xx+25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="tick" x="{xx+32}" y="{legend_y+4}">{html.escape(series_labels.get(series, series))}</text>')
    if significance_key is not None:
        parts.append(f'<circle cx="{width-202}" cy="{legend_y}" r="6" fill="none" stroke="#C04DFF" stroke-width="2"/>')
        parts.append(f'<text class="tick" x="{width-188}" y="{legend_y+4}">Holm p ≤ .05</text>')
    parts.append(f'<text class="axis-label" transform="translate(18 {top+(panel_h-top-bottom)/2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_layerwise_subspace_section(
    prompt_statistics: list[dict[str, str]],
    prompt_trends: list[dict[str, str]],
    prompt_audit: dict[str, Any],
    answer_statistics: list[dict[str, str]],
    answer_damage_statistics: list[dict[str, str]],
    answer_trends: list[dict[str, str]],
    answer_audit: dict[str, Any],
    transport_statistics: list[dict[str, str]],
    transport_trends: list[dict[str, str]],
    transport_audit: dict[str, Any],
    map_summary: list[dict[str, str]],
    map_audit: dict[str, Any],
    registered_stability: list[dict[str, str]],
    map_causal_tests: list[dict[str, str]],
    map_causal_correlations: list[dict[str, str]],
    map_causal_audit: dict[str, Any],
) -> str:
    """Build the outcome-complete layer sweep that extends report section 5.4."""

    models = ("Qwen3-8B", "Gemma4-E4B")
    panel_titles = (("Qwen3-8B", "Qwen3-8B"), ("Gemma4-E4B", "Gemma4-E4B"))

    def median(values: Iterable[float]) -> float:
        ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
        if not ordered:
            return math.nan
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    def ci(row: dict[str, str], mean_key: str = "mean_effect") -> str:
        return (
            f'{float(row[mean_key]):.3f} '
            f'[{float(row["bootstrap_95ci_low"]):.3f}, '
            f'{float(row["bootstrap_95ci_high"]):.3f}]'
        )

    def is_holm_positive(row: dict[str, str], p_key: str, effect_key: str) -> bool:
        return float(row[p_key]) <= 0.05 and float(row[effect_key]) > 0.0

    prompt_primary = [
        dict(row, layer_tick=f'L{int(row["layer"])}')
        for row in prompt_statistics
        if row["population"] == "all"
        and row["endpoint"] == "absolute_error_specificity"
    ]
    prompt_primary_trends = [
        row
        for row in prompt_trends
        if row["population"] == "all"
        and row["endpoint"] == "absolute_error_specificity"
    ]
    prompt_summary_rows: list[list[str]] = []
    prompt_detail_rows: list[list[str]] = []
    prompt_sentences: list[str] = []
    for model in models:
        model_rows = sorted(
            [row for row in prompt_primary if row["model_label"] == model],
            key=lambda row: int(row["layer"]),
        )
        trend = next(row for row in prompt_primary_trends if row["model_label"] == model)
        positive = [
            row
            for row in model_rows
            if is_holm_positive(
                row,
                "holm_p_within_model_population_endpoint",
                "mean_effect",
            )
        ]
        negative = [
            row
            for row in model_rows
            if float(row["holm_p_within_model_population_endpoint"]) <= 0.05
            and float(row["mean_effect"]) < 0.0
        ]
        peak = max(model_rows, key=lambda row: float(row["mean_effect"]))
        prompt_summary_rows.append(
            [
                model,
                f"{len(positive)}/{len(model_rows)}",
                f"{len(negative)}/{len(model_rows)}",
                f'L{int(peak["layer"])}: {ci(peak)}',
                f'{float(trend["mean_slope_per_unit_depth"]):.3f} '
                f'[{float(trend["bootstrap_95ci_low"]):.3f}, '
                f'{float(trend["bootstrap_95ci_high"]):.3f}]',
                fmt_p(float(trend["exact_seed_signflip_p_two_sided"])),
            ]
        )
        prompt_sentences.append(
            f'{model} 有 {len(positive)}/{len(model_rows)} 个注册层呈正向且 Holm 显著的方向特异 damage'
        )
        for row in model_rows:
            prompt_detail_rows.append(
                [
                    model,
                    f'L{int(row["layer"])}',
                    ci(row),
                    fmt_p(float(row["exact_seed_signflip_p_two_sided"])),
                    fmt_p(float(row["holm_p_within_model_population_endpoint"])),
                ]
            )

    prompt_svg = layerwise_ci_curve_svg(
        prompt_primary,
        panels=panel_titles,
        series_key="endpoint",
        series_order=("absolute_error_specificity",),
        series_labels={"absolute_error_specificity": "rank-3 removal − orthogonal control"},
        x_key="normalized_depth",
        layer_label_key="layer_tick",
        value_key="mean_effect",
        low_key="bootstrap_95ci_low",
        high_key="bootstrap_95ci_high",
        significance_key="significant_holm_0_05",
        y_label="absolute-error specificity (count units)",
        title="Layerwise rank-three prompt removal",
        description=(
            "Mean confirmation-seed difference between rank-three removal damage and "
            "the realized-norm-matched orthogonal control, with 95 percent seed bootstrap intervals."
        ),
    )

    answer_primary = [
        dict(row, layer_tick=f'L{int(row["layer"])}')
        for row in answer_statistics
        if row["population"] == "all"
        and row["endpoint"] == "absolute_error_specificity"
    ]
    answer_damage = [
        dict(row, layer_tick=f'L{int(row["layer"])}')
        for row in answer_damage_statistics
        if row["population"] == "all"
        and row["endpoint"]
        in {
            "candidate_absolute_error_damage",
            "control_absolute_error_damage",
        }
    ]
    answer_primary_trends = [
        row
        for row in answer_trends
        if row["population"] == "all"
        and row["endpoint"] == "absolute_error_specificity"
    ]

    def damage_ci(row: dict[str, str]) -> str:
        return (
            f'{float(row["mean_damage"]):.3f} '
            f'[{float(row["bootstrap_95ci_low"]):.3f}, '
            f'{float(row["bootstrap_95ci_high"]):.3f}]'
        )

    answer_summary_rows: list[list[str]] = []
    answer_detail_rows: list[list[str]] = []
    answer_sentences: list[str] = []
    for model in models:
        model_rows = sorted(
            [row for row in answer_primary if row["model_label"] == model],
            key=lambda row: int(row["layer"]),
        )
        trend = next(
            row for row in answer_primary_trends if row["model_label"] == model
        )
        positive = [
            row
            for row in model_rows
            if is_holm_positive(
                row,
                "holm_p_within_model_population_endpoint",
                "mean_effect",
            )
        ]
        specificity_peak = max(model_rows, key=lambda row: float(row["mean_effect"]))
        candidate_by_layer = {
            int(row["layer"]): row
            for row in answer_damage
            if row["model_label"] == model
            and row["endpoint"] == "candidate_absolute_error_damage"
        }
        control_by_layer = {
            int(row["layer"]): row
            for row in answer_damage
            if row["model_label"] == model
            and row["endpoint"] == "control_absolute_error_damage"
        }
        candidate_peak = max(
            candidate_by_layer.values(), key=lambda row: float(row["mean_damage"])
        )
        control_peak = max(
            control_by_layer.values(), key=lambda row: float(row["mean_damage"])
        )
        answer_summary_rows.append(
            [
                model,
                f"{len(positive)}/{len(model_rows)}",
                f'L{int(specificity_peak["layer"])}: {ci(specificity_peak)}',
                f'L{int(candidate_peak["layer"])}: {damage_ci(candidate_peak)}',
                f'L{int(control_peak["layer"])}: {damage_ci(control_peak)}',
                f'{float(trend["mean_slope_per_unit_depth"]):.3f} '
                f'[{float(trend["bootstrap_95ci_low"]):.3f}, '
                f'{float(trend["bootstrap_95ci_high"]):.3f}]',
                fmt_p(float(trend["exact_seed_signflip_p_two_sided"])),
            ]
        )
        answer_sentences.append(
            f'{model} 有 {len(positive)}/{len(model_rows)} 个注册层呈正向且 Holm 显著的 answer-query specificity'
        )
        for row in model_rows:
            layer = int(row["layer"])
            answer_detail_rows.append(
                [
                    model,
                    f"L{layer}",
                    damage_ci(candidate_by_layer[layer]),
                    damage_ci(control_by_layer[layer]),
                    ci(row),
                    fmt_p(float(row["exact_seed_signflip_p_two_sided"])),
                    fmt_p(float(row["holm_p_within_model_population_endpoint"])),
                ]
            )

    answer_damage_svg = layerwise_ci_curve_svg(
        answer_damage,
        panels=panel_titles,
        series_key="endpoint",
        series_order=(
            "candidate_absolute_error_damage",
            "control_absolute_error_damage",
        ),
        series_labels={
            "candidate_absolute_error_damage": "candidate damage vs clean",
            "control_absolute_error_damage": "orthogonal-control damage vs clean",
        },
        x_key="normalized_depth",
        layer_label_key="layer_tick",
        value_key="mean_damage",
        low_key="bootstrap_95ci_low",
        high_key="bootstrap_95ci_high",
        significance_key=None,
        y_label="absolute-error damage from clean (count units)",
        title="Answer-query removal: absolute candidate and control damage",
        description=(
            "Absolute-error damage from clean generation after rank-three answer-query removal "
            "and after its fixed-direction realized-norm-matched orthogonal control."
        ),
    )
    answer_specificity_svg = layerwise_ci_curve_svg(
        answer_primary,
        panels=panel_titles,
        series_key="endpoint",
        series_order=("absolute_error_specificity",),
        series_labels={
            "absolute_error_specificity": "candidate damage − orthogonal-control damage"
        },
        x_key="normalized_depth",
        layer_label_key="layer_tick",
        value_key="mean_effect",
        low_key="bootstrap_95ci_low",
        high_key="bootstrap_95ci_high",
        significance_key="significant_holm_0_05",
        y_label="absolute-error specificity (count units)",
        title="Answer-query removal: direction-specific damage",
        description=(
            "Candidate absolute-error damage minus fixed-direction orthogonal-control damage "
            "across registered decoder layers, with seed-bootstrap intervals."
        ),
    )

    requested_contrasts = (
        "aligned_dose_1_minus_orthogonal",
        "dose_2_minus_dose_1",
    )
    transport_primary = [
        dict(
            row,
            boundary_tick=f'L{int(row["source_layer"])}→L{int(row["target_layer"])}',
        )
        for row in transport_statistics
        if row["metric"] == "target_donor_fraction"
        and row["contrast"] in requested_contrasts
    ]
    transport_primary_trends = [
        row
        for row in transport_trends
        if row["metric"] == "target_donor_fraction"
        and row["contrast"] in requested_contrasts
    ]
    transport_summary_rows: list[list[str]] = []
    transport_detail_rows: list[list[str]] = []
    transport_sentences: list[str] = []
    for model in models:
        for contrast_name in requested_contrasts:
            model_rows = sorted(
                [
                    row
                    for row in transport_primary
                    if row["model_label"] == model
                    and row["contrast"] == contrast_name
                ],
                key=lambda row: int(row["source_layer"]),
            )
            trend = next(
                row
                for row in transport_primary_trends
                if row["model_label"] == model
                and row["contrast"] == contrast_name
            )
            positive = [
                row
                for row in model_rows
                if is_holm_positive(
                    row,
                    "holm_p_within_model_metric_contrast",
                    "mean_contrast",
                )
            ]
            negative = [
                row
                for row in model_rows
                if float(row["holm_p_within_model_metric_contrast"]) <= 0.05
                and float(row["mean_contrast"]) < 0.0
            ]
            peak = max(model_rows, key=lambda row: float(row["mean_contrast"]))
            short_label = (
                "aligned 1× − orthogonal"
                if contrast_name == "aligned_dose_1_minus_orthogonal"
                else "aligned 2× − aligned 1×"
            )
            transport_summary_rows.append(
                [
                    model,
                    short_label,
                    f"{len(positive)}/{len(model_rows)}",
                    f"{len(negative)}/{len(model_rows)}",
                    f'L{int(peak["source_layer"])}→L{int(peak["target_layer"])}: '
                    f'{ci(peak, "mean_contrast")}',
                    f'{float(trend["mean_slope_per_unit_depth"]):.3f} '
                    f'[{float(trend["bootstrap_95ci_low"]):.3f}, '
                    f'{float(trend["bootstrap_95ci_high"]):.3f}]',
                    fmt_p(float(trend["exact_seed_signflip_p_two_sided"])),
                ]
            )
            if contrast_name == "aligned_dose_1_minus_orthogonal":
                transport_sentences.append(
                    f'{model} 有 {len(positive)}/{len(model_rows)} 个注册边界通过方向特异性的 Holm 检验'
                )
            for row in model_rows:
                transport_detail_rows.append(
                    [
                        model,
                        str(row["boundary_tick"]),
                        short_label,
                        ci(row, "mean_contrast"),
                        fmt_p(float(row["exact_seed_signflip_p_two_sided"])),
                        fmt_p(float(row["holm_p_within_model_metric_contrast"])),
                    ]
                )

    transport_svg = layerwise_ci_curve_svg(
        transport_primary,
        panels=panel_titles,
        series_key="contrast",
        series_order=requested_contrasts,
        series_labels={
            "aligned_dose_1_minus_orthogonal": "aligned 1× − orthogonal",
            "dose_2_minus_dose_1": "aligned 2× − aligned 1×",
        },
        x_key="normalized_depth",
        layer_label_key="boundary_tick",
        value_key="mean_contrast",
        low_key="bootstrap_95ci_low",
        high_key="bootstrap_95ci_high",
        significance_key="significant_holm_0_05",
        y_label="target donor-fraction contrast",
        title="Layerwise transport-aligned causal contrasts",
        description=(
            "Direction-specific aligned transport and dose increment across preregistered adjacent "
            "decoder boundaries, with 95 percent confirmation-seed bootstrap intervals."
        ),
    )

    rank_three_maps: list[dict[str, Any]] = []
    for source in map_summary:
        if int(source["rank"]) != 3:
            continue
        row: dict[str, Any] = dict(source)
        row["layer"] = int(source["source_layer"])
        row["map_role"] = source["role"]
        row["role"] = "all"
        row["cv_centroid_r2_clipped"] = max(-1.0, min(1.0, float(source["cv_centroid_r2"])))
        rank_three_maps.append(row)
    map_cv_svg = layer_curve_svg(
        rank_three_maps,
        panels=(("Qwen3-8B", "all", "Qwen"), ("Gemma4-E4B", "all", "Gemma")),
        series_key="map_role",
        value_key="cv_centroid_r2_clipped",
        series_order=("prompt_running", "answer_query"),
        y_min=-1.0,
        y_max=1.0,
        y_label="grouped-seed centroid CV R² (clipped below −1)",
        title="Cross-layer linear-map generalization",
        description=(
            "Rank-three adjacent-layer centroid cross-validation for prompt-running and answer-query "
            "states. Values below minus one are displayed at minus one and remain available in the table."
        ),
    )
    operator_rows = [
        row
        for row in rank_three_maps
        if str(row.get("full_operator_cosine_to_next", "")) not in {"", "nan", "NaN"}
    ]
    map_operator_svg = layer_curve_svg(
        operator_rows,
        panels=(("Qwen3-8B", "all", "Qwen"), ("Gemma4-E4B", "all", "Gemma")),
        series_key="map_role",
        value_key="full_operator_cosine_to_next",
        series_order=("prompt_running", "answer_query"),
        y_min=-1.0,
        y_max=1.0,
        y_label="cosine(Tℓ, Tℓ₊₁)",
        title="Gauge-invariant full-space operator stability",
        description=(
            "Frobenius cosine between consecutive rank-three full-space operators; unlike raw PCA "
            "coordinate rotations, this quantity is invariant to independent basis rotations."
        ),
    )
    map_detail_rows: list[list[str]] = []
    for row in sorted(
        rank_three_maps,
        key=lambda item: (
            models.index(str(item["model_label"])),
            str(item["map_role"]),
            int(item["source_layer"]),
        ),
    ):
        def optional_metric(key: str, digits: int = 3) -> str:
            value = str(row.get(key, ""))
            if value in {"", "nan", "NaN"}:
                return "—"
            return f"{float(value):.{digits}f}"

        stable_label = "—"
        if row["map_role"] == "answer_query":
            stable_label = (
                "stable"
                if float(row["cv_centroid_r2"]) >= 0.9
                and float(row["bootstrap_map_relative_frobenius_median"]) <= 0.1
                else "unstable"
            )
        map_detail_rows.append(
            [
                str(row["model_label"]),
                "prompt" if row["map_role"] == "prompt_running" else "answer",
                f'L{int(row["source_layer"])}→L{int(row["target_layer"])}',
                f'{float(row["cv_centroid_r2"]):.3f}',
                f'{float(row["bootstrap_map_relative_frobenius_median"]):.3f}',
                f'{float(row["bootstrap_rotation_geodesic_degrees_median"]):.2f}°',
                f'{float(row["subspace_principal_angle_max_degrees"]):.2f}°',
                optional_metric("full_operator_cosine_to_next"),
                optional_metric("full_operator_relative_drift_to_next"),
                stable_label,
            ]
        )

    map_phase_rows: list[list[str]] = []
    stretch_rows: list[list[str]] = []
    for model in models:
        for role in ("prompt_running", "answer_query"):
            subset = [
                row
                for row in rank_three_maps
                if row["model_label"] == model and row["map_role"] == role
            ]
            early = [row for row in subset if float(row["normalized_depth"]) <= 1.0 / 3.0]
            late = [row for row in subset if float(row["normalized_depth"]) >= 2.0 / 3.0]
            map_phase_rows.append(
                [
                    model,
                    "prompt running" if role == "prompt_running" else "answer query",
                    f'{median(float(row["cv_centroid_r2"]) for row in early):.3f} → '
                    f'{median(float(row["cv_centroid_r2"]) for row in late):.3f}',
                    f'{median(float(row["bootstrap_map_relative_frobenius_median"]) for row in early):.3f} → '
                    f'{median(float(row["bootstrap_map_relative_frobenius_median"]) for row in late):.3f}',
                    f'{median(float(row["bootstrap_rotation_geodesic_degrees_median"]) for row in early):.2f}° → '
                    f'{median(float(row["bootstrap_rotation_geodesic_degrees_median"]) for row in late):.2f}°',
                    f'{median(float(row["subspace_principal_angle_max_degrees"]) for row in early):.2f}° → '
                    f'{median(float(row["subspace_principal_angle_max_degrees"]) for row in late):.2f}°',
                    f'{median(float(row["full_operator_cosine_to_next"]) for row in early if str(row.get("full_operator_cosine_to_next", "")) not in {"", "nan", "NaN"}):.3f} → '
                    f'{median(float(row["full_operator_cosine_to_next"]) for row in late if str(row.get("full_operator_cosine_to_next", "")) not in {"", "nan", "NaN"}):.3f}',
                    f'{median(float(row["full_operator_relative_drift_to_next"]) for row in early if str(row.get("full_operator_relative_drift_to_next", "")) not in {"", "nan", "NaN"}):.3f} → '
                    f'{median(float(row["full_operator_relative_drift_to_next"]) for row in late if str(row.get("full_operator_relative_drift_to_next", "")) not in {"", "nan", "NaN"}):.3f}',
                ]
            )
            stretch_rows.append(
                [
                    model,
                    "prompt running" if role == "prompt_running" else "answer query",
                    f'[{median(float(row["map_singular_last"]) for row in early):.3f}, '
                    f'{median(float(row["map_singular_1"]) for row in early):.3f}]',
                    f'[{median(float(row["map_singular_last"]) for row in late):.3f}, '
                    f'{median(float(row["map_singular_1"]) for row in late):.3f}]',
                ]
            )

    stable_rows: list[list[str]] = []
    for model in models:
        subset = [row for row in registered_stability if row["model_label"] == model]
        stable = [
            row
            for row in subset
            if str(row["locally_stable"]).strip().lower() in {"1", "true", "yes"}
        ]
        all_answer_boundaries = sorted(
            [
                row
                for row in rank_three_maps
                if row["model_label"] == model and row["map_role"] == "answer_query"
            ],
            key=lambda row: int(row["source_layer"]),
        )
        persistent = "none"
        for index, row in enumerate(all_answer_boundaries):
            suffix = all_answer_boundaries[index:]
            if all(
                float(item["cv_centroid_r2"]) >= 0.9
                and float(item["bootstrap_map_relative_frobenius_median"]) <= 0.1
                for item in suffix
            ):
                persistent = (
                    f'L{int(row["source_layer"])}→L{int(row["target_layer"])} onward'
                )
                break
        stable_rows.append(
            [
                model,
                f"{len(stable)}/{len(subset)}",
                ", ".join(
                    f'L{int(row["source_layer"])}→L{int(row["target_layer"])}'
                    for row in stable
                ),
                persistent,
            ]
        )

    primary_link_rows = [
        row
        for row in map_causal_tests
        if str(row["is_primary"]).strip().lower() in {"1", "true", "yes"}
    ]
    link_table_rows: list[list[str]] = []
    supported_models: list[str] = []
    inverse_models: list[str] = []
    for model in models:
        row = next(item for item in primary_link_rows if item["model_label"] == model)
        supported = (
            float(row["bootstrap_95ci_low"]) > 0.0
            and float(row["holm_p_within_model_six_tests"]) <= 0.05
        )
        inverse = (
            float(row["bootstrap_95ci_high"]) < 0.0
            and float(row["holm_p_within_model_six_tests"]) <= 0.05
        )
        if supported:
            supported_models.append(model)
        if inverse:
            inverse_models.append(model)
        link_table_rows.append(
            [
                model,
                f'{int(row["stable_boundaries"])}/{int(row["unstable_boundaries"])}',
                ci(row, "mean_stable_minus_unstable"),
                fmt_p(float(row["exact_seed_signflip_p_two_sided"])),
                fmt_p(float(row["holm_p_within_model_six_tests"])),
                "supports" if supported else ("significant inverse" if inverse else "not established"),
            ]
        )
    if inverse_models:
        positive_text = ", ".join(supported_models) if supported_models else "none"
        link_conclusion = (
            f'正向支持模型：{positive_text}；显著反向模型：{", ".join(inverse_models)}。'
        )
    elif len(supported_models) == len(models):
        link_conclusion = "两模型都支持：局部 map 稳定区的 held-out causal transport 更强。"
    elif supported_models:
        link_conclusion = f'只有 {", ".join(supported_models)} 支持稳定区的 held-out causal transport 更强；另一模型未建立该联系。'
    else:
        link_conclusion = "两模型都未建立“局部 map 越稳定，held-out causal transport 越强”的注册联系。"

    predictor_labels = {
        "cv_centroid_r2": "centroid CV R²",
        "bootstrap_map_relative_frobenius_median": "bootstrap map dispersion",
        "bootstrap_rotation_geodesic_degrees_median": "rotation dispersion",
        "subspace_principal_angle_max_degrees": "max principal angle",
        "full_operator_cosine_to_next": "operator cosine to next",
        "full_operator_relative_drift_to_next": "operator drift to next",
    }
    correlation_rows: list[list[str]] = []
    for row in map_causal_correlations:
        if (
            row["contrast"] != "aligned_dose_1_minus_orthogonal"
            or row["metric"] != "target_donor_fraction"
        ):
            continue
        raw_rho = str(row.get("spearman_rho_descriptive", ""))
        rho = "—" if raw_rho in {"", "nan", "NaN"} else f"{float(raw_rho):.3f}"
        correlation_rows.append(
            [
                row["model_label"],
                predictor_labels.get(row["predictor"], row["predictor"]),
                str(int(row["boundaries"])),
                rho,
            ]
        )

    rank_sensitivity_rows: list[list[str]] = []
    for model in models:
        for role in ("prompt_running", "answer_query"):
            values = []
            for rank in (1, 2, 3):
                subset = [
                    float(row["cv_centroid_r2"])
                    for row in map_summary
                    if row["model_label"] == model
                    and row["role"] == role
                    and int(row["rank"]) == rank
                ]
                values.append(f"{median(subset):.3f}")
            rank_sensitivity_rows.append(
                [
                    model,
                    "prompt running" if role == "prompt_running" else "answer query",
                    *values,
                ]
            )

    audit_rows = [
        [
            "needle-end prompt removal",
            str(prompt_audit["status"]),
            f'{int(prompt_audit["rows"]):,} raw rows; '
            f'max candidate/control target-norm relative difference '
            f'{float(prompt_audit["max_target_norm_relative_difference"]):.4g}; '
            f'max realized control-ratio error '
                f'{float(prompt_audit["max_control_norm_ratio_error"]):.4g}',
        ],
        [
            "answer-query removal",
            str(answer_audit["status"]),
            f'{int(answer_audit["rows"]):,} raw rows; '
            f'max candidate/control target-norm relative difference '
            f'{float(answer_audit["max_target_norm_relative_difference"]):.4g}; '
            f'max realized control-ratio error '
            f'{float(answer_audit["max_control_norm_ratio_error"]):.4g} '
            f'(tolerance {float(answer_audit["realized_norm_relative_tolerance"]):.1%})',
        ],
        [
            "transport",
            str(transport_audit["status"]),
            f'{int(transport_audit["rows"]):,} raw rows; '
            f'max realized control-ratio error '
            f'{float(transport_audit["max_control_norm_ratio_error"]):.4g}; '
            f'max realized 2×/1× ratio error '
            f'{float(transport_audit["max_dose2_norm_ratio_error"]):.4g}',
        ],
        [
            "linear maps",
            str(map_audit["status"]),
            f'{int(map_audit["boundaries"])} adjacent boundaries; '
            f'{int(map_audit["summary_rows"]):,} rank-summary rows; '
            f'{int(map_audit["bootstrap_rows"]):,} bootstrap rows',
        ],
        [
            "map–causal link",
            str(map_causal_audit["status"]),
            f'{int(map_causal_audit["bootstrap_draws"]):,} seed bootstraps; '
            "outcome-blind stability rule; exact seed sign-flip",
        ],
    ]

    return f"""
<div class="transport-subspace-embedded layerwise-subspace-sweep" id="layerwise-subspace-sweep">
<h4>5.4C · 实验 A 的跨层扫描：三维 removal 在哪些层具有方向特异性？</h4>
<p><strong>两个位置、两个 estimand。</strong>原实验改动每条 active needle 的 <em>endpoint token</em>，现在明确称为 <strong>needle-end prompt removal</strong>；新增实验只改 final <code>Total:</code> 的 <strong>answer-query token</strong>。两者使用相同 discovery/confirmation seeds、counts 2–10、rank 3 与注册层网格，但绝不能把位置不同的结果合并成一条曲线。</p>
<p><strong>共同主估计量。</strong>每个注册层都只用 discovery seeds 冻结该位置的 rank-3 count plane。candidate 删除当前 state 相对 discovery global centroid 在 count plane 上的投影；control 删除它在 discovery-frozen orthogonal within-count rank-3 basis 上的投影，并且只沿该原方向缩放。纵轴 specificity 为 <code>candidate absolute-error damage from clean − control absolute-error damage from clean</code>。answer-query 的 BF16 realized-norm 相对容差为 5%；不搜索或优化 control 方向。</p>
<h5>5.4C-1 · Needle-end prompt removal（保留的原实验）</h5>
<figure>{prompt_svg}<figcaption><strong>图 8A · Needle-end prompt rank-3 removal 的逐层特异效应。</strong>横轴为实施 intervention 的 decoder layer，纵轴为 candidate 相对等范数正交 control 多造成的 absolute-error damage；0 表示两种方向的破坏相同。candidate 与 control 都在每条 needle 的 endpoint token 上实施；点是10个 confirmation seeds 的等权均值，误差棒为 seed-bootstrap 95% CI，外圈表示在“模型 × population × endpoint”family 内跨层 Holm 后仍显著。</figcaption></figure>
{table(["model", "positive Holm layers", "negative Holm layers", "largest observed effect", "linear depth slope [95% CI]", "trend exact p"], prompt_summary_rows, classes="paper-table compact-result-table")}
{details_table("All registered needle-end prompt-removal layers", ["model", "layer", "specificity [95% CI]", "raw exact p", "Holm p"], prompt_detail_rows)}
<p><strong>Needle-end 读法。</strong>{'；'.join(prompt_sentences)}。它只说明“在单层同时删除各 needle endpoint 的三维 prompt-side 分量”没有建立方向特异必要性，不能据此推断 answer-query state 也不必要。</p>

<h5>5.4C-2 · Answer-query removal（新增的对称位置实验）</h5>
<figure>{answer_damage_svg}<figcaption><strong>Figure · Absolute answer-query damage.</strong> 两条曲线分别是 candidate 与固定方向 orthogonal control 相对 clean generation 的绝对误差损伤；这里只展示 intervention dose 的行为后果，不把各自相对零的 CI 当作 count-direction specificity 检验。</figcaption></figure>
<figure>{answer_specificity_svg}<figcaption><strong>图 8B · Answer-query removal 的逐层方向特异效应。</strong>横轴为 answer-query intervention layer，纵轴为逐 seed 配对的 candidate−control absolute-error damage；0 表示 count plane 与固定正交方向的破坏相同。点与误差棒分别是等权 seed mean 和 seed-bootstrap 95% CI；外圈表示跨注册层 Holm 后通过。所有 counts 先在 seed 内平均。</figcaption></figure>
{table(["model", "positive Holm layers", "peak specificity", "peak candidate damage", "peak control damage", "specificity depth slope [95% CI]", "trend exact p"], answer_summary_rows, classes="paper-table compact-result-table")}
{details_table("All registered answer-query removal layers", ["model", "layer", "candidate damage [95% CI]", "control damage [95% CI]", "specificity [95% CI]", "raw exact p", "Holm p"], answer_detail_rows)}
<p><strong>Answer-query 读法。</strong>{'；'.join(answer_sentences)}。两个模型的 specificity 都随深度显著上升，且显著层集中在中后段；这与 needle-end 的全层 null 形成位置特异的对照。结论是 late answer-query count subspace 对输出具有方向特异的必要性证据，而不是“任意位置删除同样三维方向都会破坏计数”。线性 depth slope 是注册的全局趋势摘要，非单调细节仍以逐层估计为准。</p>
<div class="conclusion"><strong>8.3C 结论。</strong>needle-end prompt removal 在注册层上没有建立方向特异必要性，而 answer-query removal 的 specificity 随深度增加并在中后层出现确认性正效应。因而“同一个三维 count plane 在任何 token 位置都同样必要”不成立；当前必要性证据定位于 late answer-query state。</div>

<h4>5.4D · 实验 B 的跨层扫描：transport causal effect 与剂量响应如何随层变化？</h4>
<p>对每个注册相邻边界，把 source ambient centroids ridge-regress 到 target rank-3 coordinates，并对回归权重做 QR 得到 discovery-frozen transport basis <code>B<sub>ℓ</sub></code>；confirmation 只使用 1→2、2→1、5→6、6→5 四个预注册 donor/receiver 对。紫色曲线检验 aligned 1× 相对等范数正交方向的 specificity；蓝色曲线检验 2× 相对 1× 的 dose increment。</p>
<figure>{transport_svg}<figcaption><strong>图 8C · Layerwise transport specificity 与 dose response。</strong>横轴为相邻 source→target decoder-layer boundary，纵轴为 target donor fraction 的 seed-level contrast；紫色为 aligned 1×−orthogonal，蓝色为 aligned 2×−aligned 1×，0 表示没有方向特异性或剂量增量。误差棒为 seed-bootstrap 95% CI，外圈表示在“模型 × metric × contrast”family 内跨注册边界 Holm 后仍显著。</figcaption></figure>
{table(["model", "contrast", "positive Holm boundaries", "negative Holm boundaries", "largest observed effect", "linear depth slope [95% CI]", "trend exact p"], transport_summary_rows, classes="paper-table compact-result-table")}
{details_table("All registered transport boundaries", ["model", "boundary", "contrast", "effect [95% CI]", "raw exact p", "Holm p"], transport_detail_rows)}
<p><strong>跨层复现。</strong>{'；'.join(transport_sentences)}。这一区分“某个锚点可 transport”与“transport 通道在整个 decoder 中均匀存在”；后者必须由曲线而不是单点结果判断。</p>
<div class="conclusion"><strong>8.3D 结论。</strong>方向特异 transport 与 2×−1× 剂量增量主要在 decoder 中后段增强，并非从早层到晚层均匀存在。该结果支持局部成熟的 count relay，而不支持全网络共享恒定强度的单一 transport channel。</div>

<h4>5.4E · 如何定义跨层 linear map、rotation 与稳定性？</h4>
<p><strong>形式化定义。</strong>令 discovery-fitted source/target bases 为 <code>U<sub>ℓ</sub></code> 与 <code>U<sub>ℓ+1</sub></code>，中心化 rank-3 coordinates 为 <code>z<sub>ℓ</sub></code>。以 seed-grouped ridge 拟合 <code>z<sub>ℓ+1</sub> ≈ z<sub>ℓ</sub>A<sub>ℓ</sub></code>，再做 polar decomposition <code>A<sub>ℓ</sub>=R<sub>ℓ</sub>S<sub>ℓ</sub></code>：<code>R</code> 是 coordinate rotation/reflection，<code>S</code> 是 stretch。PCA basis 可各自右乘任意正交矩阵，因此裸的 <code>R<sub>ℓ</sub></code> 角度依赖 gauge；它只在 bootstrap basis 先向 full-discovery reference 做 Procrustes 对齐后用于<strong>同一边界内</strong>的不确定性。</p>
<p><strong>与 causal patch 的关系。</strong><code>A<sub>ℓ</sub></code> 是两端都先截断为 rank 3 的坐标 map；5.4D 的 <code>B<sub>ℓ</sub></code> 则是从 source ambient centroid curve 直接预测 target rank-3 coordinates 后取得的三维 row-space。二者共享 discovery centroids 与 target rank，但不是同一个参数化；实验没有把 <code>R<sub>ℓ</sub></code> 矩阵本身写入模型。最后的 map–causal test 问的是“<code>A<sub>ℓ</sub></code> 稳定的边界上，<code>B<sub>ℓ</sub></code> 定义的 held-out causal transport 是否更强”，不是把相关性误写成对 rotation matrix 的直接干预。</p>
<p><strong>跨边界主稳定量。</strong>把 map 提升回 residual space：<code>T<sub>ℓ</sub>=U<sub>ℓ</sub>A<sub>ℓ</sub>U<sub>ℓ+1</sub><sup>T</sup></code>。若独立换 gauge 为 <code>U′<sub>ℓ</sub>=U<sub>ℓ</sub>Q<sub>ℓ</sub></code>，则 <code>A′<sub>ℓ</sub>=Q<sub>ℓ</sub><sup>T</sup>A<sub>ℓ</sub>Q<sub>ℓ+1</sub></code>，代回即有 <code>T′<sub>ℓ</sub>=T<sub>ℓ</sub></code>。因此可比较 <code>cos(T<sub>ℓ</sub>,T<sub>ℓ+1</sub>)</code> 与相对 Frobenius drift，而不把 PCA 轴的任意换号/旋转当成机制变化。另报告 grouped-seed centroid CV R²（跨 seed 泛化）和 seed bootstrap map relative-Frobenius dispersion（估计稳定性）。</p>
<figure>{map_cv_svg}<figcaption><strong>Figure · Predictive map stability.</strong> rank-3 grouped-seed centroid CV R²；为保留后层分辨率，小于 −1 的早层失败值画在 −1 下限，未截断数值保留在机器可读 CSV。</figcaption></figure>
<figure>{map_operator_svg}<figcaption><strong>图 8D · Gauge-invariant operator continuity。</strong>横轴为 map 起始 decoder layer，纵轴为相邻两个 full-space operators <code>T<sub>ℓ</sub></code> 与 <code>T<sub>ℓ+1</sub></code> 的 Frobenius cosine。1 表示两条相邻边界实施近乎相同的 ambient-space linear transport，0 表示正交；它不等于“PCA 轴标签相同”，也不是对 rotation matrix 的直接因果干预。</figcaption></figure>
{table(["model", "role", "early→late median CV R²", "early→late map dispersion", "early→late rotation dispersion", "early→late max principal angle", "early→late operator cosine", "early→late operator drift"], map_phase_rows, classes="paper-table compact-result-table")}
{details_table("All adjacent rank-3 map boundaries (unclipped)", ["model", "role", "boundary", "CV R²", "map dispersion", "rotation dispersion", "max principal angle", "operator cosine", "operator drift", "answer stability rule"], map_detail_rows)}
{details_table("Polar stretch summary: median [smallest, largest] singular value", ["model", "role", "early third", "late third"], stretch_rows)}
{details_table("Rank sensitivity: median grouped-seed centroid CV R²", ["model", "role", "rank 1", "rank 2", "rank 3"], rank_sensitivity_rows)}
<p><strong>Outcome-blind stability rule。</strong>该阈值是在 discovery-only map sweep 已完成之后、任何 layerwise transport confirmation outcome 产生之前冻结：answer-query rank-3 boundary 同时满足 CV R²≥0.9 与 bootstrap relative-Frobenius median≤0.1 才标为 locally stable。因此它不是完全事前注册的 map 阈值，但也没有用 causal outcome 调参。</p>
{table(["model", "stable / registered", "stable registered boundaries", "first persistently stable full-sweep boundary"], stable_rows, classes="paper-table compact-result-table")}
{table(["model", "stable/unstable boundaries", "stable − unstable causal effect [95% CI]", "raw exact p", "Holm p (6 tests/model)", "decision"], link_table_rows, classes="paper-table compact-result-table")}
{details_table("Descriptive boundary-wise map–causal Spearman correlations", ["model", "map predictor", "complete boundaries", "Spearman ρ"], correlation_rows)}
{details_table("Layerwise experiment machine audits", ["component", "status", "completeness / numerical audit"], audit_rows)}
<div class="conclusion"><strong>跨层结论</strong>{link_conclusion} 该检验使用 held-out confirmation causal effects，不把 discovery map fit 与 causal outcome 循环定义；boundary-wise Spearman 仅作描述，不把固定层位当成随机总体。由于 locally stable 与 decoder depth 高度共线，这个 regime contrast 仍不能把“map 稳定性本身”与一般性的 late-layer maturation 分离；需要 fresh-seed 的密集过渡区或 depth-matched 边界才能进一步识别。</div>
</div>
"""


def all_token_scatter_svg(
    rows: list[dict[str, str]], *, model: str, layer: int
) -> str:
    subset = [row for row in rows if row["model_label"] == model and int(row["layer"]) == layer]
    # Keep every endpoint and a deterministic thin sample of the much larger controls.
    controls = [row for row in subset if row["category"] != "needle_endpoint"]
    stride = max(1, len(controls) // 900)
    plotted = [row for row in subset if row["category"] == "needle_endpoint"] + controls[::stride]
    xs = [float(row["pc1"]) for row in plotted]
    ys = [float(row["pc2"]) for row in plotted]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    width, height, left, right, top, bottom = 530, 440, 58, 22, 36, 56
    x = lambda value: left + (float(value) - xmin) / max(xmax - xmin, 1e-12) * (width-left-right)
    y = lambda value: top + (ymax - float(value)) / max(ymax - ymin, 1e-12) * (height-top-bottom)
    category_color = {
        "needle_endpoint": "#F6E36A",
        "needle_interior": "#6750E8",
        "hard_negative": "#FF5FA2",
        "ordinary_passage": "#8190A5",
    }
    parts = [f'<svg class="stat-svg all-token-scatter" viewBox="0 0 {width} {height}" role="img" aria-label="{model} layer {layer} frozen endpoint PCA projection">']
    parts.append(f'<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="#15142A"/>')
    for row in plotted:
        category = row["category"]
        radius = 3.2 if category == "needle_endpoint" else 1.5
        opacity = 0.85 if category == "needle_endpoint" else 0.25
        parts.append(f'<circle cx="{x(row["pc1"]):.2f}" cy="{y(row["pc2"]):.2f}" r="{radius}" fill="{category_color[category]}" opacity="{opacity}"/>')
    parts.append(f'<text class="panel-title" x="{width/2}" y="22" text-anchor="middle">{model} · L{layer}</text>')
    parts.append(f'<text class="axis-label" x="{left+(width-left-right)/2}" y="{height-12}" text-anchor="middle">endpoint-fitted PC1</text>')
    parts.append(f'<text class="axis-label" transform="translate(16 {top+(height-top-bottom)/2}) rotate(-90)" text-anchor="middle">endpoint-fitted PC2</text>')
    parts.append('</svg>')
    return "".join(parts)


def _row(
    rows: Sequence[dict[str, str]], **criteria: Any
) -> dict[str, str]:
    hits = [
        item
        for item in rows
        if all(str(item.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one row for {criteria}; found {len(hits)}")
    return hits[0]


def _global_classifier_ranking(
    rows: Sequence[dict[str, str]], algorithms: Sequence[str]
) -> list[tuple[str, float]]:
    ranking = []
    for algorithm in algorithms:
        values = [
            float(item["accuracy"])
            for item in rows
            if item["algorithm"] == algorithm
        ]
        if not values:
            raise RuntimeError(f"No classifier rows for {algorithm}")
        ranking.append((algorithm, sum(values) / len(values)))
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def build_extension_representation_section(
    rank_rows: list[dict[str, str]],
    regression_rows: list[dict[str, str]],
    clustering_rows: list[dict[str, str]],
    noise_rows: list[dict[str, str]],
    all_token_metrics: list[dict[str, str]],
    formula_rows: list[dict[str, str]],
    projection_rows: list[dict[str, str]],
    classifier_rows: list[dict[str, str]],
    classifier_correct_rows: list[dict[str, str]],
    cue_doc: str,
) -> str:
    """Paper-facing answer to the representation questions in the extension memo."""

    landmarks = (
        ("Qwen3-8B", "prompt_running", 8, "Prompt running index"),
        ("Gemma4-E4B", "prompt_running", 9, "Prompt running index"),
        ("Qwen3-8B", "answer_query", 29, "Answer query"),
        ("Gemma4-E4B", "answer_query", 37, "Answer query"),
    )
    algorithms = (
        "logistic_l2",
        "ridge_classifier",
        "linear_svm",
        "nearest_centroid",
        "shrinkage_lda",
        "knn_k5_cosine",
    )
    algorithm_labels = {
        "logistic_l2": "L2 logistic",
        "ridge_classifier": "ridge",
        "linear_svm": "linear SVM",
        "nearest_centroid": "centroid",
        "shrinkage_lda": "shrinkage LDA",
        "knn_k5_cosine": "cosine kNN-5",
    }

    rank_curve_rows: list[dict[str, Any]] = []
    for item in rank_rows:
        for series, field in (
            ("all-state rank-3", "total_variance_capture_k3"),
            ("count-centroid rank-3", "centroid_curve_capture_k3"),
        ):
            rank_curve_rows.append(
                {
                    "model_label": item["model_label"],
                    "role": item["role"],
                    "layer": item["layer"],
                    "series": series,
                    "value": item[field],
                }
            )
    prompt_rank_svg = layer_curve_svg(
        rank_curve_rows,
        panels=(
            ("Qwen3-8B", "prompt_running", "Qwen prompt endpoints"),
            ("Gemma4-E4B", "prompt_running", "Gemma prompt endpoints"),
        ),
        series_key="series",
        value_key="value",
        series_order=("all-state rank-3", "count-centroid rank-3"),
        y_min=0.0,
        y_max=1.0,
        y_label="variance fraction",
        title="Prompt endpoint rank three variance",
        description=(
            "Layerwise rank-three variance of every endpoint state and of the ten "
            "count centroids, fit on discovery seeds."
        ),
    )
    answer_rank_svg = layer_curve_svg(
        rank_curve_rows,
        panels=(
            ("Qwen3-8B", "answer_query", "Qwen answer query"),
            ("Gemma4-E4B", "answer_query", "Gemma answer query"),
        ),
        series_key="series",
        value_key="value",
        series_order=("all-state rank-3", "count-centroid rank-3"),
        y_min=0.0,
        y_max=1.0,
        y_label="variance fraction",
        title="Answer query rank three variance",
        description=(
            "Layerwise rank-three variance of answer-query states and of the ten "
            "count centroids."
        ),
    )

    regression_plot_rows = []
    for item in regression_rows:
        regression_plot_rows.append(
            {
                **item,
                "display_r2": max(-0.25, min(1.0, float(item["r2_mean"]))),
            }
        )
    prompt_regression_svg = layer_curve_svg(
        regression_plot_rows,
        panels=(
            ("Qwen3-8B", "prompt_running", "Qwen prompt endpoints"),
            ("Gemma4-E4B", "prompt_running", "Gemma prompt endpoints"),
        ),
        series_key="algorithm",
        value_key="display_r2",
        series_order=("ridge", "knn5_distance"),
        y_min=-0.25,
        y_max=1.0,
        y_label="held-out R² (floor at −0.25)",
        title="Prompt count regression",
        description=(
            "Held-out count regression across layers. Values below minus 0.25 are "
            "display-clipped but retained exactly in the tables."
        ),
    )
    answer_regression_svg = layer_curve_svg(
        regression_plot_rows,
        panels=(
            ("Qwen3-8B", "answer_query", "Qwen answer query"),
            ("Gemma4-E4B", "answer_query", "Gemma answer query"),
        ),
        series_key="algorithm",
        value_key="display_r2",
        series_order=("ridge", "knn5_distance"),
        y_min=-0.25,
        y_max=1.0,
        y_label="held-out R² (floor at −0.25)",
        title="Answer count regression",
        description="Seed-grouped held-out count regression at the answer query.",
    )

    classifier_display_rows = [
        {**item, "algorithm_label": algorithm_labels[item["algorithm"]]}
        for item in classifier_rows
    ]
    classifier_order = tuple(algorithm_labels[item] for item in algorithms)
    prompt_classifier_svg = layer_curve_svg(
        classifier_display_rows,
        panels=(
            ("Qwen3-8B", "prompt_running", "Qwen prompt endpoints"),
            ("Gemma4-E4B", "prompt_running", "Gemma prompt endpoints"),
        ),
        series_key="algorithm_label",
        value_key="accuracy",
        series_order=classifier_order,
        y_min=0.0,
        y_max=1.0,
        y_label="10-class held-out accuracy",
        title="Prompt classifier comparison",
        description="All six fixed classifier families are shown at every layer.",
    )
    answer_classifier_svg = layer_curve_svg(
        classifier_display_rows,
        panels=(
            ("Qwen3-8B", "answer_query", "Qwen answer query"),
            ("Gemma4-E4B", "answer_query", "Gemma answer query"),
        ),
        series_key="algorithm_label",
        value_key="accuracy",
        series_order=classifier_order,
        y_min=0.0,
        y_max=1.0,
        y_label="10-class held-out accuracy",
        title="Answer classifier comparison",
        description="All-sample answer-query classification with seed-grouped folds.",
    )
    classifier_correct_display_rows = [
        {**item, "algorithm_label": algorithm_labels[item["algorithm"]]}
        for item in classifier_correct_rows
    ]
    answer_classifier_correct_svg = layer_curve_svg(
        classifier_correct_display_rows,
        panels=(
            ("Qwen3-8B", "answer_query", "Qwen correct-only answer"),
            ("Gemma4-E4B", "answer_query", "Gemma correct-only answer"),
        ),
        series_key="algorithm_label",
        value_key="balanced_accuracy",
        series_order=classifier_order,
        y_min=0.0,
        y_max=1.0,
        y_label="correct-only balanced accuracy",
        title="Correct-only answer classifier comparison",
        description=(
            "All six fixed classifier families on clean-correct answer-query rows. "
            "Balanced accuracy gives each surviving count class equal weight."
        ),
    )

    classifier_ranking = _global_classifier_ranking(classifier_rows, algorithms)
    global_primary = classifier_ranking[0][0]
    classifier_rank_table = [
        [algorithm_labels[name], fmt(value, 3)]
        for name, value in classifier_ranking
    ]

    representative_rows = []
    correct_rows = []
    cluster_rows = []
    for model, role, layer, label in landmarks:
        rank = _row(rank_rows, model_label=model, role=role, layer=layer)
        ridge = _row(
            regression_rows,
            model_label=model,
            role=role,
            layer=layer,
            algorithm="ridge",
        )
        classifier = _row(
            classifier_rows,
            model_label=model,
            role=role,
            layer=layer,
            algorithm=global_primary,
        )
        cluster = _row(clustering_rows, model_label=model, role=role, layer=layer)
        representative_rows.append(
            [
                model,
                label,
                f"L{layer}",
                fmt(float(rank["stable_rank"]), 2),
                str(rank["numeric_rank_90"]),
                fmt(float(rank["total_variance_capture_k3"]), 3),
                fmt(float(rank["centroid_curve_capture_k3"]), 3),
                fmt(float(rank["count_eta_squared"]), 3),
                fmt(float(ridge["r2_mean"]), 3),
                fmt(float(classifier["accuracy"]), 3),
            ]
        )
        cluster_rows.append(
            [
                model,
                label,
                f"L{layer}",
                fmt(float(cluster["silhouette_cosine_mean"]), 3),
                fmt(float(cluster["calinski_harabasz_mean"]), 2),
                fmt(float(cluster["davies_bouldin_mean"]), 2),
            ]
        )
        if role == "answer_query":
            sensitivity = _row(
                classifier_correct_rows,
                model_label=model,
                role=role,
                layer=layer,
                algorithm=global_primary,
            )
            correct_rows.append(
                [
                    model,
                    f"L{layer}",
                    f'{classifier["rows"]} / {sensitivity["rows"]}',
                    fmt(float(classifier["accuracy"]), 3),
                    fmt(float(sensitivity["accuracy"]), 3),
                    fmt(float(sensitivity["balanced_accuracy"]), 3),
                    str(sensitivity["count_class_count"]),
                ]
            )

    noise_plot_rows: list[dict[str, Any]] = []
    for item in noise_rows:
        if item["population"] != "confirmation":
            continue
        for series, field in (
            ("count main effect", "fraction_count"),
            ("seed/context main effect", "fraction_seed_context"),
            ("count×seed deformation", "fraction_count_by_seed_interaction"),
        ):
            noise_plot_rows.append(
                {
                    "model_label": item["model_label"],
                    "role": item["role"],
                    "layer": item["layer"],
                    "series": series,
                    "value": item[field],
                }
            )
    noise_svg = layer_curve_svg(
        noise_plot_rows,
        panels=(
            ("Qwen3-8B", "prompt_running", "Qwen confirmation endpoints"),
            ("Gemma4-E4B", "prompt_running", "Gemma confirmation endpoints"),
        ),
        series_key="series",
        value_key="value",
        series_order=(
            "count main effect",
            "seed/context main effect",
            "count×seed deformation",
        ),
        y_min=0.0,
        y_max=1.0,
        y_label="fraction of balanced Frobenius SS",
        title="Prompt endpoint variance decomposition",
        description=(
            "Balanced two-way decomposition of prompt endpoint states into count, "
            "seed-context, and count-by-seed deformation."
        ),
    )
    noise_landmark_rows = []
    for model, layer in (("Qwen3-8B", 8), ("Gemma4-E4B", 9)):
        item = _row(
            noise_rows,
            model_label=model,
            role="prompt_running",
            layer=layer,
            population="confirmation",
        )
        noise_landmark_rows.append(
            [
                model,
                f"L{layer}",
                fmt(float(item["fraction_count"]), 3),
                fmt(float(item["fraction_seed_context"]), 3),
                fmt(float(item["fraction_count_by_seed_interaction"]), 3),
            ]
        )

    token_rows = []
    for model, layer in (("Qwen3-8B", 8), ("Gemma4-E4B", 9)):
        endpoint = _row(
            all_token_metrics,
            model_label=model,
            layer=layer,
            category="needle_endpoint",
        )
        interior = _row(
            all_token_metrics,
            model_label=model,
            layer=layer,
            category="needle_interior",
        )
        hard = _row(
            all_token_metrics,
            model_label=model,
            layer=layer,
            category="hard_negative",
        )
        ordinary = _row(
            all_token_metrics,
            model_label=model,
            layer=layer,
            category="ordinary_passage",
        )
        endpoint_formula = _row(
            formula_rows,
            model_label=model,
            layer=layer,
            category="needle_endpoint",
            model="endpoint_gated_curve",
        )
        interior_formula = _row(
            formula_rows,
            model_label=model,
            layer=layer,
            category="needle_interior",
            model="needle_span_gated_curve",
        )
        ordinary_formula = _row(
            formula_rows,
            model_label=model,
            layer=layer,
            category="ordinary_passage",
            model="ungated_prefix_curve",
        )
        token_rows.append(
            [
                model,
                f"L{layer}",
                fmt(float(endpoint["nearest_endpoint_count_accuracy"]), 3),
                fmt(float(interior["nearest_endpoint_count_accuracy"]), 3),
                fmt(float(hard["nearest_endpoint_count_accuracy"]), 3),
                fmt(float(ordinary["nearest_endpoint_count_accuracy"]), 3),
                fmt(float(endpoint_formula["incremental_r2_vs_category_baseline"]), 3),
                fmt(float(interior_formula["incremental_r2_vs_category_baseline"]), 3),
                fmt(float(ordinary_formula["incremental_r2_vs_category_baseline"]), 3),
            ]
        )

    nt = extract_js_json(cue_doc, "NT_GEOM")
    prompt = extract_js_json(cue_doc, "PROMPT_GEOM")
    cue_rows = []
    for payload, site, label in (
        (prompt, "prompt_counter", "Prompt running index"),
        (nt, "answer_query", "Answer query"),
    ):
        for model, role_layers in payload["landmarks"].items():
            for landmark_name in ("display", "probe"):
                layer = int(role_layers[landmark_name])
                stat = payload["statistics"][f"{model}|{site}|{layer}"]
                cue_rows.append(
                    [
                        model,
                        label,
                        f"L{layer} ({landmark_name})",
                        fmt(float(stat["centroid_cka"]), 3),
                        f'{fmt(float(stat["count_eta_present"]), 3)} → {fmt(float(stat["count_eta_absent"]), 3)}',
                        fmt_p(float(stat["interaction_q"])),
                        fmt_p(float(stat["count_eta_q"])),
                    ]
                )

    return f"""
<section id="representation-extension">
<h2>4 · 从“看起来像曲线”到可审计的 count representation</h2>

<h3>4.1 轨迹是否真的低维：必须区分原始状态云与十个 count 中心</h3>
<div class="study-preface"><strong>为什么做。</strong><span>3D PCA 中的有序曲线可能只来自十个中心，而每个中心周围仍有很高维的上下文变化。若不区分两者，就会把“中心轨迹低维”误写成“全部 hidden states 低秩”。</span><strong>定义。</strong><span>对中心化状态矩阵 <code>X</code>，stable rank 定义为 <code>||X||²_F / ||X||²_2</code>；numeric rank-k% 是累计奇异值平方达到 k% 所需维数。另令十个 count 中心组成 <code>C</code>，分别报告 rank-3 对 <code>X</code> 与 <code>C</code> 的方差解释率。count η² 是 count 分组均值相对总平方和的比例。</span><strong>设定。</strong><span>Prompt 使用 30 seeds×10 个同一 N=10 prompt 内的 needle endpoints；几何只在 discovery seeds 拟合。Answer 使用 20 seeds×counts 1–10；逐层报告，不以 3D 可视距离替代 full-space 指标。</span></div>
<div class="figure-grid"><figure>{prompt_rank_svg}<figcaption><strong>图 4A · Prompt endpoint 的 rank-3 方差。</strong>横轴是 decoder layer，纵轴是前三个奇异方向解释的方差比例。深色线为所有 endpoint states，浅色线为十个 count centroids。两线间距量化“有序中心轨迹”之外的上下文维度。</figcaption></figure><figure>{answer_rank_svg}<figcaption><strong>图 4B · Answer-query 的 rank-3 方差。</strong>坐标定义与图 4A 相同；每个 count 的 answer-query state 来自独立 V4.4 prompt。该图用于检查进入答案位置后 count-centroid 轨迹是否比原始状态云更集中。</figcaption></figure></div>
{table(["model", "site", "layer", "stable rank", "rank90", "all-state k=3", "centroid k=3", "count η²", "ridge R²", f"{algorithm_labels[global_primary]} acc."], representative_rows, classes="paper-table compact-result-table")}
<div class="conclusion"><strong>本段结论。</strong>两模型在 prompt 与 answer 的十个 count centroids 上都存在近低维的有序轨迹，但原始状态云明显更高维。例如 Qwen prompt L8 的 centroid rank-3 解释 0.988，而全部 endpoint states 只有 0.690；Gemma prompt L9 分别为 0.979 与 0.598。因此严谨表述是“低维 count curve 嵌在高维、上下文依赖的 residual state 中”，而不是“hidden state 整体是三维的”。</div>

<h3>4.2 压缩、解码与聚类：count 线性可读，但 PCA 中心轨迹可以弯曲</h3>
<div class="study-preface"><strong>为什么做。</strong><span>低维中心轨迹若能在 held-out seed 上预测 count，才是可泛化表征；若只在训练样本上形成十个簇，可能是可视化过拟合。这里还必须区分两个问题：“能否用线性函数读出 count”与“十个 centroids 是否落在一条欧氏直线上”不是同一命题。</span><strong>具体例子。</strong><span>十个点可以沿一条弯曲的弧线排列，但它们在某个方向上的标量投影仍随 count 单调增加。此时 ridge 可以准确预测 count，而前三个 PCA 轴中的轨迹仍明显弯曲；因此高线性解码 R²不能被写成“geometry 是直线”。</span><strong>评估。</strong><span>连续解码比较 ridge 与 distance-weighted kNN-5 的 held-out R²、MAE 与 Pearson r；所有预处理只在训练 fold 拟合，5-fold GroupKFold 按 seed 切分。离散分类器与聚类只作为敏感性分析：分类器家族在查看单层结果前固定，correct-only 只检验成功作答路径，不替代平衡 all-sample estimand。</span></div>
<div class="figure-grid"><figure>{prompt_regression_svg}<figcaption><strong>图 4C · Prompt count 连续解码。</strong>横轴为层，纵轴为 held-out R²；紫/绿线分别表示 ridge 与 kNN-5。为保持后层可读性，低于 −0.25 的早层 R² 只在图中截到 −0.25，CSV 保留原值。</figcaption></figure><figure>{answer_regression_svg}<figcaption><strong>图 4D · Answer count 连续解码。</strong>与图 4C 相同，但样本单位是 answer-query state，5-fold GroupKFold 以 seed 分组。</figcaption></figure></div>
<p><strong>为什么不能只报 correct-only。</strong>Correct-only 很适合回答“模型成功作答时，state 是否清楚记录 gold count”，但它是由最终输出决定的 outcome-conditioned subset：Qwen 失去 counts 7/9，Gemma 失去9/10，chance level 由0.1变为0.125，且容易留下较简单样本。只报它会高估总体可解码性并隐藏错误机制。因此正文把 correct-only balanced accuracy 作为<strong>成功计算路径的主敏感性分析</strong>，同时保留平衡 all-sample 作为总体机制诊断。</p>
<p><strong>连续解码结果。</strong>Qwen prompt L8 的 ridge / kNN R²为0.945 / 0.711，Gemma prompt L9为0.719 / 0.374；Qwen answer L29为0.891 / 0.871，Gemma answer L37为0.885 / 0.773。线性 ridge 在四个代表点都不差于kNN，说明 count 可由 hidden state 的线性组合稳定读出。它不说明 centroid path 本身是直线：PCA 图中两模型的轨迹都有可见曲率，而ridge只要求存在一个线性投影能够预测标量count。kNN在高维中还会受到seed/context dispersion和有限样本的影响，因此“ridge更好”也不能用于否定曲率。</p>
<p><strong>离散解码与聚类结果。</strong>固定 L2 logistic 在 correct-only answer代表层的 raw / balanced accuracy 分别为Qwen 0.880 / 0.759、Gemma 0.918 / 0.625。代表层 cosine silhouette 为Qwen prompt −0.075、Gemma prompt −0.043、Qwen answer 0.056、Gemma answer −0.038；十类并未形成紧密球状clusters。最一致的几何描述因此是<strong>“线性可读的、有曲率的有序中心轨迹，外加seed/context形变”</strong>。</p>
<div class="conclusion"><strong>本段结论。</strong>Qwen 与 Gemma 的 prompt/answer count 都能在 held-out seeds 上被线性模型读出；这里的“线性”描述 decoder，不描述 PCA 轨迹形状。两模型的 PCA centroid trajectory 均可弯曲，且低 silhouette 表明它们不是十个紧密小球。Correct-only 证实成功计算时 answer state 明确携带 gold count，但因类别截断不能替代平衡的 all-sample 结果。</div>

<h3>4.3 “Noise”是什么：确定性的上下文形变，而不是采样随机性</h3>
<div class="study-preface"><strong>为什么做。</strong><span>同一个 running count 在不同 prompt 中不会落在同一点。我们要回答的不只是“点有多散”，而是这些偏离究竟来自整条 prompt 轨迹一起平移，还是来自不同 prompt 的轨迹形状本身不同。</span><strong>具体例子。</strong><span>假设 seed A 的 count 1–10 全部比总体轨迹向右移同样距离，这叫 <em>seed/context offset</em>；如果 seed A 在 count 1–4 与总体一致、到 count 8–10 却被压缩或弯向另一方向，这叫 <em>count×seed interaction</em>。两者都让点偏离 count 中心，但计算含义不同。</span><strong>Noise 的直接定义。</strong><span>模型是确定性 forward，因此这里的 noise 不是重复采样的随机波动，而是同一 count 在不同 prompt 间的确定性 dispersion。令 <code>H[c,s]</code> 为 seed <code>s</code> 中第 <code>c</code> 个 needle endpoint 的 hidden state，<code>μ<sub>c</sub></code> 为 count <code>c</code> 跨 seeds 的中心，则 <code>ε<sub>c,s</sub>=H[c,s]−μ<sub>c</sub></code>。它表示该点离“所有 prompt 共享的 count 中心”有多远。</span><strong>Seed 与 context。</strong><span><code>seed</code> 只是 prompt 身份的<strong>类别标签</strong>：它一起决定 haystack 文本、needle identity、位置、顺序与普通段落。计算中使用 one-hot 固定效应；不会把 1255 当成比 1254 大 1 的连续变量。因为一个 seed 只对应一个整体 context，本实验能分离“哪条 prompt”造成的变异，却不能进一步断言是某个词、某个位置或某种句法单独造成。</span></div>
<div class="noise-decomposition-grid" aria-label="Noise decomposition definition">
  <div><strong>① 共享 count signal</strong><code>μ<sub>c</sub>−μ</code><span>不同 prompts 共同拥有的 running-count 轨迹。</span></div>
  <div><strong>② Prompt-wide offset</strong><code>μ<sub>s</sub>−μ</code><span>同一 seed 的十个 count states 一起平移的部分。</span></div>
  <div><strong>③ Seed-specific deformation</strong><code>H<sub>c,s</sub>−μ<sub>c</sub>−μ<sub>s</sub>+μ</code><span>不能由“共享轨迹 + 整体平移”解释的弯曲、压缩或局部偏差。</span></div>
</div>
<div class="equation"><code>H<sub>c,s</sub>−μ = (μ<sub>c</sub>−μ) + (μ<sub>s</sub>−μ) + (H<sub>c,s</sub>−μ<sub>c</sub>−μ<sub>s</sub>+μ)</code></div>
<p><strong>如何得到三个比例。</strong>在平衡的 <code>count×seed</code> 网格中，分别对上面三个向量项求所有样本、所有 hidden dimensions 的平方和：<code>SS=Σ||component||²</code>，再除以总平方和 <code>SS<sub>total</sub>=Σ||H<sub>c,s</sub>−μ||²</code>。平衡设计使三项正交，所以比例之和为 1（表中因三位小数舍入可能相差 0.001）。这也可逐维写成含 count、seed 与 count×seed one-hot 项的 two-way ANOVA；它是方差记账，不是拿 seed 编号做数值预测。</p>
<figure>{noise_svg}<figcaption><strong>图 4E · Prompt endpoint 的 balanced two-way variance decomposition。</strong>横轴是层，纵轴是 confirmation states 的 Frobenius 总平方和比例；三条线之和为 1。count 主效应是跨 seed 共享的 count trajectory，seed/context 是跨 count 共享的 prompt offset，interaction 是 seed 特异的轨迹形变。</figcaption></figure>
{table(["model", "layer", "count", "seed/context", "count×seed"], noise_landmark_rows, classes="paper-table compact-result-table")}
<p><strong>Qwen：noise 来自什么。</strong>Qwen L8 的总变异中，59.9%是跨 prompts 共享的 count trajectory，剩余40.2%可视为相对count centroid的noise。以<strong>noise本身</strong>为分母，prompt identity所对应的整条轨迹offset解释16.1/(16.1+24.1)=40.0%，count×prompt interaction解释60.0%。因此Qwen的noise约四成是“整条轨迹一起平移”，约六成是“不同prompt使轨迹发生count-dependent弯曲、压缩或局部偏移”。</p>
<p><strong>Gemma：noise 来自什么。</strong>Gemma L9 的共享 count trajectory解释38.5%的总变异，剩余61.4%是noise。以<strong>noise本身</strong>为分母，prompt-wide offset解释22.8/(22.8+38.6)=37.1%，count×prompt interaction解释62.9%。Gemma比Qwen更noisy，具体表现为共享count signal占比更低，而且绝对interaction份额更大（38.6% vs 24.1%的总变异）。</p>
<p><strong>“Feature 能解释多少”必须谨慎回答。</strong>当前non-thinking V4.4中，唯一可作为主效应解释变量的是categorical <code>prompt identity</code>：它解释Qwen 40.0%、Gemma 37.1%的within-count noise。余下60.0%/62.9%被记为<code>count×prompt interaction</code>；这是从每个count×prompt cell留下的形变项，不是一个能对新prompt做held-out预测的具体feature。因此不能说我们已经用feature解释了全部noise。更具体的needle位置、lexical identity、间距或段落长度没有被独立操纵或回归；它们的解释率是<strong>尚未识别，而不是0%</strong>。</p>
<p><strong>下一步如何识别具体来源。</strong>需要factorial prompts：固定文本只改needle位置、固定位置只改literal identity、固定二者再改ordinary context，并在每个组合下重复多个seeds；随后用按seed held-out的增量R²或variance partitioning报告每个feature block对<code>||ε<sub>c,s</sub>||</code>或完整residual vector的额外解释量。此前V4.4.0中的metadata feature regression使用的是<strong>native-thinking</strong>样本，不能把它的R²移植为本non-thinking V4.4的feature解释率。V4.4高-count correct strata又很稀疏，尤其Qwen confirmation的N=10没有完整correct trajectory，因此本节不把correct-only ANOVA写成有充分power的主结论。</p>
<div class="conclusion"><strong>本段结论。</strong>Qwen的within-count noise约40.0%来自prompt-wide offset、60.0%来自count-dependent prompt deformation；Gemma分别为37.1%和62.9%。所以两模型的主要noise形式都是轨迹形状随context改变，Gemma的总noise比例更高。现有实验只把noise分到prompt identity及其与count的interaction，尚不能把它归因到needle位置、词汇或段落结构等具体features。</div>

<h3>4.4 把同一 endpoint PCA 投到所有 token：count curve 是 endpoint-gated</h3>
<div class="study-preface"><strong>为什么做。</strong><span>若每个 needle 内部 token 都携带同一 running index，公式应对整段成立；若只有记录结束时形成汇总状态，只有 endpoint 应沿 count curve 排列。</span><strong>设定。</strong><span>用 discovery needle endpoints 拟合 PCA-32、十个 centroids 与 curve；冻结后投影 confirmation 的全部 needle endpoints、needle interiors、hard negatives，以及每个 prompt 128 个按深度分层抽取的 ordinary passage tokens。比较 category-only baseline、endpoint-gated curve、whole-needle-span gated curve 与把 prefix count 强加给普通 token 的 ungated curve。</span><strong>指标。</strong><span>表中 acc. 是 token 到最近 endpoint count centroid 的十分类准确率；ΔR² 是加入指定 count curve 后相对 category mean baseline 减少的 full-space SSE 比例，负值表示公式比只用 token category 更差。</span></div>
<div class="figure-grid token-role-comparison"><figure>{all_token_scatter_svg(projection_rows, model="Qwen3-8B", layer=8)}<figcaption><strong>图 4F-left · Qwen L8：同一 endpoint basis 下的 token roles。</strong>横轴/纵轴是只用 discovery endpoints 拟合的 PC1/PC2。黄色=needle endpoint，紫色=同一句内部 token，棕色=外形相似但不是目标记录的 hard negative，灰色=普通 passage。问题不是各颜色是否视觉分开，而是哪一类能被冻结 count centroids 正确排序。</figcaption></figure><figure>{all_token_scatter_svg(projection_rows, model="Gemma4-E4B", layer=9)}<figcaption><strong>图 4F-right · Gemma L9：与左图完全相同的定义。</strong>两图并列用于比较模型，而正式判定使用下表的 full-space nearest-centroid accuracy 与ΔR²。endpoint-gated ΔR²问“只给 endpoint 加 count curve能减少多少误差”；span/ungated ΔR²问把同一公式错误推广到其他 token 会怎样。</figcaption></figure></div>
{table(["model", "layer", "endpoint acc.", "interior acc.", "hard-neg. acc.", "ordinary acc.", "endpoint-gated ΔR²", "span-gated interior ΔR²", "ungated ordinary ΔR²"], token_rows, classes="paper-table compact-result-table")}
<div class="equation">h_t = b_{{type(t)}} + 1[t is a needle endpoint] · γ(N(s≤t)) + ε_t.</div>
<p>Qwen L8 的 endpoint 最近中心准确率为 0.49、MAE 0.87，而 interior / hard-negative / ordinary 仅为 0.016 / 0.008 / 0.025；endpoint-gated curve 增量 R²=0.551。Gemma L9 的 endpoint accuracy=0.28、增量 R²=0.326，而三类非 endpoint 均约机会水平。把 curve 扩展到整个 needle interior 或普通 passage 会得到负的 ΔR²。因此原 memo 中的 <code>1(t∈needle span)</code> 需要收紧为 endpoint gate。</p>
<div class="conclusion"><strong>本段结论。</strong>两个模型都不是在 needle 的每个 token 上持续写同一个 running index；可泛化 count curve 主要形成在 needle endpoint。最符合数据的经验公式是 category baseline 加 endpoint-gated count curve，其他 token 保持高维 context state。</div>

<h3>4.5 开头 instruction cue 是否生成了 running counter</h3>
<div class="study-preface"><strong>为什么做。</strong><span>我们要区分“提示语创建了 counter”与“提示语只调节一个由重复 record 结构自然形成的 counter”。</span><strong>设定。</strong><span>V4.4.2 保持同一 non-thinking flag 与 passage，只删去开头两句 city-score 定义。Centroid CKA 比较十个 count 中心的相对 topology；full-space count×cue interaction 检验提示造成的变化是否依赖 count；count η² 的变化按 layer 做 BH-FDR。</span></div>
{table(["model", "site", "layer", "centroid CKA", "count η² present → absent", "count×cue q", "Δ strength q"], cue_rows, classes="paper-table compact-result-table")}
<p>所有列出的 centroid CKA 为 0.981–0.997，说明移除提示后 count 的相对拓扑高度保留；同时 count×cue interaction 显著，说明 full state 并非只做共同平移。换言之，开头提示能调制增益、局部方向或 role offset，但重复 record 结构与上下文本身已足以形成 running-index ordering。</p>
<div class="conclusion"><strong>本段结论。</strong>开头 cue 不是 running-index topology 的必要生成源，但会显著调制完整 residual geometry；因此正文可说 counter 对 cue removal 稳健，不能说 cue 对机制完全无影响。现有主因果链仍限定在 cue-present V4.4。</div>
</section>
"""


def build_endpoint_formation_section(
    attention_stats: list[dict[str, str]],
    earlier_heads: list[dict[str, str]],
    attention_audit: dict[str, Any],
) -> str:
    """Explain the endpoint-attention tests without treating masks as pure ablations."""

    primary = [
        row
        for row in attention_stats
        if row["primary_layer"] == "True" and row["estimand"] == "specificity"
    ]
    metric_labels = {
        "continuous_absolute_error": "continuous count absolute error",
        "nearest_centroid_accuracy": "nearest-centroid count accuracy",
        "normalized_noise": "gold-centroid normalized distance",
    }
    specificity_rows = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for metric in (
            "continuous_absolute_error",
            "nearest_centroid_accuracy",
            "normalized_noise",
        ):
            row = next(
                item
                for item in primary
                if item["model_label"] == model and item["metric"] == metric
            )
            specificity_rows.append(
                [
                    model,
                    f'L{row["layer"]}',
                    metric_labels[metric],
                    f'{fmt(float(row["mean"]), 3)} [{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}]',
                    fmt_p(float(row["p_value"])),
                    fmt_p(float(row["holm_p_primary_family"])),
                ]
            )

    improvement_rows = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        layer = 8 if model == "Qwen3-8B" else 9
        for metric in (
            "continuous_absolute_error",
            "nearest_centroid_accuracy",
            "normalized_noise",
        ):
            row = next(
                item
                for item in attention_stats
                if item["model_label"] == model
                and int(item["layer"]) == layer
                and item["metric"] == metric
                and item["estimand"] == "needle_only_improvement"
            )
            improvement_rows.append(
                [
                    model,
                    metric_labels[metric],
                    fmt(float(row["mean"]), 3, signed=True),
                    f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}]',
                    fmt_p(float(row["p_value"])),
                ]
            )

    earlier_by_model = {
        model: sorted(
            [row for row in earlier_heads if row["model_label"] == model],
            key=lambda item: int(item["rank"]),
        )
        for model in ("Qwen3-8B", "Gemma4-E4B")
    }
    if any(len(rows) != 10 for rows in earlier_by_model.values()):
        raise RuntimeError(
            "Earlier-span confirmation must contain ten frozen heads for each model"
        )

    def earlier_head_plot(model: str, color: str) -> str:
        rows = earlier_by_model[model]
        return forest_svg(
            [
                {
                    "label": f'rank {row["rank"]} · L{row["layer"]}H{row["head"]}',
                    "mean": float(row["confirmation_preference_mean"]),
                    "low": float(row["ci95_low"]),
                    "high": float(row["ci95_high"]),
                    "value": (
                        f'{fmt(float(row["confirmation_preference_mean"]), 3)} · '
                        f'Holm p={fmt_p(float(row["holm_p_within_model"]))}'
                    ),
                    "color": color,
                }
                for row in rows
            ],
            title=f"{model} earlier-needle-span attention confirmation",
            description=(
                "Discovery-ranked heads are evaluated on confirmation seeds. The effect "
                "is attention mass to prior needle spans minus equal-length ordinary spans "
                "at matched depth."
            ),
            x_label="prior-needle full-span mass minus matched ordinary-span mass",
            left=250,
            right=270,
            zero=0.0,
        )

    q_head_svg = earlier_head_plot("Qwen3-8B", "#6750E8")
    g_head_svg = earlier_head_plot("Gemma4-E4B", "#00D4B4")
    q_earlier = earlier_by_model["Qwen3-8B"]
    g_earlier = earlier_by_model["Gemma4-E4B"]

    q_specificity = {
        row["metric"]: row
        for row in primary
        if row["model_label"] == "Qwen3-8B"
    }
    g_specificity = {
        row["metric"]: row
        for row in primary
        if row["model_label"] == "Gemma4-E4B"
    }
    return f"""
<section id="formation-tests">
<h2>5 · Prompt running state 如何形成：earlier-span aggregation 与上下文支架</h2>

<h3>5.1 Earlier-span attention：Qwen 与 Gemma 的同口径独立确认</h3>
<div class="study-preface"><strong>为什么做。</strong><span>重复记录可能使模型在第 n 个 needle endpoint 更新 running index 时回看前 n−1 个 needle spans。若成立，这提供一种形成累计状态的候选计算，而不要求某一个 head 独自完成计数。</span><strong>具体例子。</strong><span>当 query 位于第6条 needle 的末端时，若某个 head 对前5条 needle 完整文本的总注意力明显大于对5段同长度、同相对深度普通文本的注意力，就称它表现出 earlier-span aggregation。</span><strong>定义。</strong><span>每个模型分别在 discovery rows 上按“query endpoint 指向更早 needle 的 full-span literal mass”冻结前10个 heads；在 confirmation seeds 上计算 <code>P=mass(prior needle spans)−mass(equal-length, same-depth ordinary spans)</code>。这里使用整个 literal span，而不是只看最后一个 key token。</span><strong>推断。</strong><span>每个 head 先对同一 seed 的 occurrences 2/4/6/8/10 求均值，再对10个 seed effects 做双侧 exact sign-flip；各模型10个冻结 heads 内分别做 Holm 校正。两模型的 estimand、seeds、occurrences、control 与推断完全一致。Gemma 仅在工程上重建冻结 head 的单个 query row，以避免保存全层二次方 attention tensor；逐行 softmax 审计的最大归一化误差为5.6×10<sup>−7</sup>。</span></div>
<div class="figure-grid token-role-comparison"><figure>{q_head_svg}<figcaption><strong>图 5A-left · Qwen earlier-span attention。</strong>纵轴为 discovery 冻结的top-10 heads；横轴为 prior-needle full-span mass减去等长度、等深度ordinary-span mass。点为10个confirmation seeds的均值，横线为95% seed bootstrap CI；右侧为模型内10-head Holm p。</figcaption></figure><figure>{g_head_svg}<figcaption><strong>图 5A-right · Gemma earlier-span attention。</strong>坐标、冻结规则、10个seeds与推断均与左图相同；绿色只编码模型身份，不改变统计含义。</figcaption></figure></div>
<p><strong>Qwen结果。</strong>十个冻结heads的confirmation preference均为正，范围为{fmt(min(float(row['confirmation_preference_mean']) for row in q_earlier), 3)}–{fmt(max(float(row['confirmation_preference_mean']) for row in q_earlier), 3)}；每个head的exact p=0.001953，模型内Holm p=0.019531。</p>
<p><strong>Gemma结果。</strong>十个冻结heads同样全部为正，范围为{fmt(min(float(row['confirmation_preference_mean']) for row in g_earlier), 3)}–{fmt(max(float(row['confirmation_preference_mean']) for row in g_earlier), 3)}；每个head的exact p=0.001953，模型内Holm p=0.019531。最强的L5H5/L5H0/L5H3 effects分别为0.960、0.945与0.922；说明Gemma虽有sliding-window层，早期full-attention layers仍存在强烈的跨已出现needle spans聚合。</p>
<div class="conclusion"><strong>本段结论。</strong><strong>共同结果：</strong>Qwen与Gemma都在全新confirmation seeds上确认了endpoint→prior-needle full-span的特异聚合，而且效应不是由等长度、等深度ordinary context解释。该结果支持“endpoint更新时会汇聚更早目标记录”的候选形成机制；它仍不足以证明previous-token matching、复制操作等完整经典induction-circuit定义，也不等于这些heads单独完成counting。</div>

<h3>5.2 把 outside-needle attention 当作 noise 会怎样？</h3>
<div class="study-preface"><strong>为什么做。</strong><span>§4.3 只把 dispersion 分成 seed/context offset 与 count×seed deformation，并不知道它们由什么计算产生。这里检验一个具体因果解释：ordinary context 是否通过 attention 向 endpoint 注入无用变化；如果是，屏蔽它应减少 centroid noise。</span><strong>干预。</strong><span>在 occurrences 2/4/6/8/10 的 endpoint query 上比较三种 forward：clean；<code>needle-only</code>（只允许该 query 读取所有 active needle spans）；<code>matched ordinary-only</code>（只允许读取相同 key 数、相近相对深度的 ordinary passage tokens）。其余 queries 不改。Qwen/Gemma 主层分别冻结为 L8/L9，共10个 confirmation seeds、每模型50个 endpoints。</span><strong>指标。</strong><span>continuous error、nearest-centroid accuracy 与 normalized gold-centroid distance 都在冻结 geometry 中计算。误差型 improvement=<code>clean error−masked error</code>，accuracy improvement=<code>masked acc−clean acc</code>；正值统一表示 mask 后更好。specificity=<code>needle-only improvement−ordinary-only improvement</code>。推断以 seed 为单位做 bootstrap 与 exact sign-flip，并对6个主 endpoints 做 Holm 校正。</span></div>
{table(["model", "metric", "needle-only improvement", "95% CI", "exact p"], improvement_rows, classes="paper-table compact-result-table")}
{table(["model", "layer", "specificity metric", "needle−ordinary [95% CI]", "exact p", "Holm p"], specificity_rows, classes="paper-table compact-result-table")}
<p>两模型的三项 needle-only improvement 均未显示 clean 之上的改善：Qwen L8 的 continuous error、accuracy 与 normalized distance effects 分别为 −0.261、−0.280、−1.610；Gemma L9 为 −3.688、−0.080、−1.568。也就是说，去掉所有 ordinary/context keys 会破坏而不是净化 running state。相对同 key-budget 的 ordinary-only control，Qwen 仍有明确的 needle-specific information：continuous、accuracy、distance specificity 分别为 {fmt(float(q_specificity['continuous_absolute_error']['mean']), 3)}、{fmt(float(q_specificity['nearest_centroid_accuracy']['mean']), 3)}、{fmt(float(q_specificity['normalized_noise']['mean']), 3)}，三项 Holm 均≤0.023438。Gemma 只有 normalized-distance specificity 通过 familywise 校正（{fmt(float(g_specificity['normalized_noise']['mean']), 3)}，Holm p={fmt_p(float(g_specificity['normalized_noise']['holm_p_primary_family']))}）；continuous 与 accuracy 的方向不支持更强行为主张。</p>
<div class="conclusion"><strong>本段结论。</strong><strong>共同结果：</strong>Qwen与Gemma在needle-only条件下都没有比clean更少noise，因此“outside context只是应被删除的噪声来源”不成立。<strong>Qwen：</strong>needle-only相对matched ordinary-only在continuous error、accuracy和normalized distance三项都有特异性（Holm≤0.023438），支持needle keys携带额外count信息。<strong>Gemma：</strong>只有normalized-distance specificity通过Holm（3.307，p=0.011719），continuous error与accuracy不支持相同的行为主张。Ordinary context可能提供位置、语法或格式支架，但当前实验不区分这些具体作用。审计状态：{html.escape(str(attention_audit['status']))}。</div>
</section>
"""


def _extension_stat(
    rows: Sequence[dict[str, str]], **criteria: Any
) -> dict[str, str]:
    hits = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one extension statistic for {criteria}; got {len(hits)}")
    return hits[0]


def build_prompt_direct_causal_subsections(
    token_stats: list[dict[str, str]],
    token_audit: dict[str, Any],
    subspace_stats: list[dict[str, str]],
    subspace_audit: dict[str, Any],
) -> str:
    """Direct prompt-side interventions requested by the extension memo."""

    behavior_endpoints = ("accuracy_drop", "absolute_error_increase")
    populations = ("all", "clean_correct_only")
    models = ("Qwen3-8B", "Gemma4-E4B")
    token_behavior_rows = []
    for population in populations:
        for model in models:
            for endpoint in behavior_endpoints:
                needle = _extension_stat(
                    token_stats,
                    population=population,
                    model_label=model,
                    endpoint=endpoint,
                    estimand="needle_damage",
                )
                control = _extension_stat(
                    token_stats,
                    population=population,
                    model_label=model,
                    endpoint=endpoint,
                    estimand="control_damage",
                )
                specific = _extension_stat(
                    token_stats,
                    population=population,
                    model_label=model,
                    endpoint=endpoint,
                    estimand="specificity",
                )
                token_behavior_rows.append(
                    [
                        population.replace("_", " "),
                        model,
                        endpoint.replace("_", " "),
                        fmt(float(needle["mean"]), 3, signed=True),
                        fmt(float(control["mean"]), 3, signed=True),
                        f'{fmt(float(specific["mean"]), 3, signed=True)} '
                        f'[{fmt(float(specific["ci95_low"]), 3)}, {fmt(float(specific["ci95_high"]), 3)}]',
                        f'{fmt_p(float(specific["p_value"]))} / '
                        f'{fmt_p(float(specific["holm_p_within_population"]))}',
                    ]
                )

    token_rep_rows = []
    for row in token_stats:
        if row["estimand"] != "specificity" or not row["endpoint"].startswith(
            "gold_centroid_distance_increase_L"
        ):
            continue
        token_rep_rows.append(
            [
                row["population"].replace("_", " "),
                row["model_label"],
                row["endpoint"].replace("gold_centroid_distance_increase_", ""),
                f'{fmt(float(row["mean"]), 3, signed=True)} '
                f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}]',
                fmt_p(float(row["p_value"])),
                fmt_p(float(row["holm_p_within_population"])),
            ]
        )

    token_forest_rows = []
    for model in models:
        row = _extension_stat(
            token_stats,
            population="all",
            model_label=model,
            endpoint="absolute_error_increase",
            estimand="specificity",
        )
        token_forest_rows.append(
            {
                "label": model,
                "mean": float(row["mean"]),
                "low": float(row["ci95_low"]),
                "high": float(row["ci95_high"]),
                "value": f'{fmt(float(row["mean"]), 3)} · Holm p={fmt_p(float(row["holm_p_within_population"]))}',
                "color": "#6750E8" if model == "Qwen3-8B" else "#00D4B4",
            }
        )
    token_svg = forest_svg(
        token_forest_rows,
        title="Needle-token corruption specificity",
        description=(
            "Needle corruption damage minus equal-token-budget ordinary-passage "
            "corruption damage on generated-count absolute error."
        ),
        x_label="specific absolute-error increase (needle corruption minus control)",
        left=230,
        right=290,
    )
    token_specific_rows = [
        row
        for row in token_stats
        if row["estimand"] == "specificity"
        and float(row["ci95_low"]) > 0
        and float(row["holm_p_within_population"]) <= 0.05
    ]
    token_all_q_error = _extension_stat(
        token_stats,
        population="all",
        model_label="Qwen3-8B",
        endpoint="absolute_error_increase",
        estimand="specificity",
    )
    token_all_g_error = _extension_stat(
        token_stats,
        population="all",
        model_label="Gemma4-E4B",
        endpoint="absolute_error_increase",
        estimand="specificity",
    )
    token_all_q_acc = _extension_stat(
        token_stats,
        population="all",
        model_label="Qwen3-8B",
        endpoint="accuracy_drop",
        estimand="specificity",
    )
    token_all_g_acc = _extension_stat(
        token_stats,
        population="all",
        model_label="Gemma4-E4B",
        endpoint="accuracy_drop",
        estimand="specificity",
    )

    subspace_behavior_rows = []
    for population in populations:
        for model in models:
            for condition in ("actual_rank3_remove", "centroid_curve_remove"):
                for endpoint in behavior_endpoints:
                    row = _extension_stat(
                        subspace_stats,
                        population=population,
                        model_label=model,
                        condition=condition,
                        endpoint=endpoint,
                    )
                    subspace_behavior_rows.append(
                        [
                            population.replace("_", " "),
                            model,
                            condition.replace("_", " "),
                            endpoint.replace("_", " "),
                            f'{fmt(float(row["mean"]), 3, signed=True)} '
                            f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}]',
                            f'{fmt_p(float(row["p_value"]))} / '
                            f'{fmt_p(float(row["holm_p_within_population"]))}',
                        ]
                    )
    subspace_rep_rows = []
    for row in subspace_stats:
        if not row["endpoint"].startswith("gold_centroid_distance_increase_L"):
            continue
        subspace_rep_rows.append(
            [
                row["population"].replace("_", " "),
                row["model_label"],
                row["condition"].replace("_", " "),
                row["endpoint"].replace("gold_centroid_distance_increase_", ""),
                f'{fmt(float(row["mean"]), 3, signed=True)} '
                f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}]',
                f'{fmt_p(float(row["p_value"]))} / '
                f'{fmt_p(float(row["holm_p_within_population"]))}',
            ]
        )

    def significant_summary(rows: Sequence[dict[str, str]], population: str) -> str:
        registered = [row for row in rows if row["population"] == population]
        hits = [
            row
            for row in registered
            if float(row["holm_p_within_population"]) <= 0.05
            and float(row["ci95_low"]) > 0
        ]
        if not hits:
            return "没有 endpoint 在 Holm 校正后呈正向 specificity"
        behavior_hits = [
            row
            for row in hits
            if row["endpoint"] in {"accuracy_drop", "absolute_error_increase"}
        ]
        behavior_text = "；".join(
            f'{row["model_label"]} {row["condition"].replace("_", " ")} '
            f'{row["endpoint"].replace("_", " ")} '
            f'({fmt(float(row["mean"]), 3)}, Holm p={fmt_p(float(row["holm_p_within_population"]))})'
            for row in behavior_hits
        )
        if not behavior_text:
            behavior_text = "behavior endpoints 无 Holm-significant 正效应"
        return (
            f"{len(hits)}/{len(registered)} 个注册 endpoints 为正且通过 Holm；"
            f"behavior：{behavior_text}。下游 geometry 逐项见折叠表"
        )

    subspace_forest_rows = []
    subspace_colors = {
        ("Qwen3-8B", "actual_rank3_remove"): "#6750E8",
        ("Qwen3-8B", "centroid_curve_remove"): "#00C2FF",
        ("Gemma4-E4B", "actual_rank3_remove"): "#00D4B4",
        ("Gemma4-E4B", "centroid_curve_remove"): "#39E58C",
    }
    for model in models:
        for condition in ("actual_rank3_remove", "centroid_curve_remove"):
            row = _extension_stat(
                subspace_stats,
                population="all",
                model_label=model,
                condition=condition,
                endpoint="absolute_error_increase",
            )
            subspace_forest_rows.append(
                {
                    "label": f'{model} · {condition.replace("_", " ")}',
                    "mean": float(row["mean"]),
                    "low": float(row["ci95_low"]),
                    "high": float(row["ci95_high"]),
                    "value": f'{fmt(float(row["mean"]), 3)} · Holm p={fmt_p(float(row["holm_p_within_population"]))}',
                    "color": subspace_colors[(model, condition)],
                }
            )
    subspace_svg = forest_svg(
        subspace_forest_rows,
        title="Set-wide prompt count-subspace removal",
        description=(
            "Specific absolute-error increase for removing the actual or idealized "
            "rank-3 prompt count component relative to an equal-norm orthogonal removal."
        ),
        x_label="specific absolute-error increase (count-subspace minus orthogonal removal)",
        left=330,
        right=290,
    )

    return f"""
<h3>5.3 Needle token corruption：直接移除输入证据</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Attention mask 只改某一个 endpoint query 的可见 keys；模型仍可能在别的位置读取原始 needle。要检验输入证据是否必要，必须让整条 forward 都看不到原 needle 内容。</span><strong>到底 corrupt 什么。</strong><span>对 seeds 1254–1263、gold count 1–10（每模型100条），定位从每条 active needle 第一个 token 到最后一个 token 的完整 span，并逐 span 替换成同一 prompt 中抽取的<strong>等 token 数 ordinary sequence</strong>。不是删除、打乱位置或改 answer query。matched control 在普通 passage 中选取总 token 数和深度匹配的 spans，也替换成 ordinary sequence。两种条件保持序列长度、所有后续绝对位置、query position 与改动 token budget相同。</span><strong>评估。</strong><span>clean、needle-corrupt、ordinary-control 都从头 greedy 生成。behavior damage 是 accuracy drop 与 absolute-error increase；representation damage 是 answer residual 到 discovery-frozen gold centroid 的 squared-distance increase；specificity=needle damage−ordinary-control damage。先在 seed 内平均 counts，再以10 seeds做bootstrap、exact sign-flip和population内 Holm校正。clean-correct-only 仅保留原 clean forward 答对的 Qwen 44/100、Gemma 36/100 rows。</span></div>
<figure>{token_svg}<figcaption><strong>图 5B · Needle-token corruption 的特异行为损伤。</strong>纵向各行分别是 model × analysis population；横轴是 needle corruption 相对等 token-budget ordinary corruption 多增加的 absolute count error。0 表示两类同规模文本破坏影响相同，正值表示 needle tokens 有额外的计数必要性。点为10个等权 seed effects 的均值，横线为95% seed-bootstrap CI。</figcaption></figure>
{table(["population", "model", "endpoint", "needle damage", "ordinary damage", "specificity [95% CI]", "exact / Holm p"], token_behavior_rows, classes="paper-table compact-result-table")}
{details_table("Needle corruption 对下游 answer geometry 的全部结果", ["population", "model", "answer layer", "specificity [95% CI]", "exact p", "Holm p"], token_rep_rows)}
<p>全样本中，ordinary control 的 absolute-error damage 接近零（Qwen −0.010；Gemma +0.040），而 needle corruption 相对它额外增加的 absolute error 为 Qwen {fmt(float(token_all_q_error['mean']), 3)}、Gemma {fmt(float(token_all_g_error['mean']), 3)}；额外 accuracy drop 为 {fmt(float(token_all_q_acc['mean']), 3)} / {fmt(float(token_all_g_acc['mean']), 3)}。两模型行为与两个下游 answer layers 的 specificity 都为正；all 与 clean-correct-only 合计 {len(token_specific_rows)}/16 个注册 specificity endpoints 通过各自 population 的 Holm 校正（这些最小 exact p=0.001953 的检验经24项保守校正后 Holm p=0.046875）。</p>
<p><strong>这项实验能推出什么、不能推出什么。</strong>干预发生在<strong>输入token层</strong>：Qwen与Gemma只在真实needle内容被替换时出现远大于ordinary-control的损伤，所以两模型都必须使用needle中携带的信息。它没有直接改某个内部hidden state；一旦输入被替换，后面所有attention、MLP和residual states都会随之改变。因此该实验不能区分“信息先存在哪个endpoint”“由哪一组heads搬运”或“哪一个subspace是唯一carrier”，这些定位必须由后续state/head intervention完成。</p>
<div class="conclusion"><strong>本小节结论。</strong>Token corruption 审计为 {html.escape(str(token_audit['status']))}。<strong>Qwen：</strong>needle-specific absolute-error increase为{fmt(float(token_all_q_error['mean']), 3)}，accuracy drop为{fmt(float(token_all_q_acc['mean']), 3)}。<strong>Gemma：</strong>对应为{fmt(float(token_all_g_error['mean']), 3)}与{fmt(float(token_all_g_acc['mean']), 3)}；两模型主行为结果均Holm p=0.046875。结论只到“原始needle内容是必要输入证据”，不把input-level necessity误写成某个内部carrier的唯一必要性。</div>

<h3>5.4 Subspace 因果实验：从固定锚点到跨层 removal、transport 与 map 稳定性</h3>
<div class="callout evidence-note"><strong>先分清逻辑。</strong>实验A与B改变的层、方向定义和因果问题都不同，所以结果不矛盾：A问一个<strong>固定早层PCA平面是否必要</strong>；B问一个<strong>允许跨层旋转的晚层transport方向是否足以搬运count</strong>。必要性检验为null、局部充分性检验为positive，可以同时成立。</div>
<div class="subspace-logic-grid">
  <article><span>实验 A · necessity</span><strong>删掉固定 prompt plane，模型会坏吗？</strong><p>位置：Qwen L8 / Gemma L9 的全部active endpoints。方向：跨seeds平均得到的静态rank-3 PCA plane。结果：相对等范数正交删除，32个注册tests无一通过Holm；行为effects接近0。</p><p><b>只说明：</b>这个特定静态平面不是模型无法绕开的必要瓶颈。模型可能冗余存储、从literal spans重算，或在后层旋转方向。</p></article>
  <article><span>实验 B · local sufficiency</span><strong>沿跨层对齐方向注入，下一层会读到吗？</strong><p>位置：Qwen L28→L29 / Gemma L36→L37 的answer query。方向：discovery学习的source→target rank-3 transport basis，允许两层坐标不同。结果：aligned 1×远大于等范数orthogonal，2×又大于1×；四项exact p=0.001953。</p><p><b>只说明：</b>测试的相邻晚层存在可用、方向特异、近似剂量响应的局部count通道；不说明它是唯一通道。</p></article>
</div>
<h4>5.4A · 实验 A：固定 prompt rank-3 PCA plane 是否是必要瓶颈？——单锚点负结果</h4>
<div class="study-preface"><strong>为什么做。</strong><span>旧 single-endpoint ablation 接近零，可能是因为 count state 分散在多个 active endpoints，删一个位置会被其他位置补偿。新实验同时干预全部 active endpoints，并区分“当前 prompt 实际落在 count subspace 中的变化”与“discovery 数据给出的理想 centroid curve”。</span><strong>干预。</strong><span>对 seeds 1254–1263、counts 2–10，在 prompt 代表层 Qwen L8 / Gemma L9 的所有 active needle endpoints 同时执行两种 rank-3 removal，随后让未修改的后续层继续生成完整答案。对照从 within-count residual 的主方向中构造与 count basis 正交的 rank-3 basis，并把每个 prompt 的删除量缩放到与候选删除相同的 Frobenius norm。</span><strong>评估。</strong><span>behavior 使用 accuracy drop 与 absolute-error increase；representation 使用 Qwen L29/L35、Gemma L37/L41 answer residual 到冻结 gold centroid 的 squared-distance increase。所有主 effects 都是 count-subspace damage−orthogonal damage；先在 seed 内平均，再做10-seed exact sign-flip、50,000次 bootstrap，并在 all / clean-correct-only populations 内分别对2 models×2 removals×4 endpoints=16项做 Holm 校正。</span></div>
<p><strong>精确定义。</strong>用 discovery endpoints 对每个 running count <em>n</em> 求 centroid <code>μ<sub>n</sub></code>，把十个 centroids 整体中心化后做 SVD，取前三个右奇异向量组成正交 basis <code>U∈R<sup>d×3</sup></code>。对一个含 <em>N</em> 个 active endpoints 的样本，将状态堆为 <code>H∈R<sup>N×d</sup></code>。<code>actual rank-3 remove</code> 使用 <code>H′=H−(H−rowmean(H))UUᵀ</code>，删除该样本跨 endpoints 的实际 count-aligned variation，但保留共同 offset；<code>centroid-curve remove</code> 使用 <code>H′<sub>n</sub>=H<sub>n</sub>−(μ<sub>n</sub>−mean(μ<sub>1:N</sub>))UUᵀ</code>，删除 discovery 所预测的理想 running-index step。控制 basis <code>V</code> 来自 <code>h−μ<sub>count</sub></code> 的前三个 residual PCs，经 Gram–Schmidt 与 <code>U</code> 正交化；删除 <code>(H−rowmean(H))VVᵀ</code> 后再缩放，使其删除范数与对应 <code>U</code>-removal 完全相同。</p>
<figure>{subspace_svg}<figcaption><strong>图 5C · Set-wide prompt count-subspace removal 的特异行为效应。</strong>横轴为删除 count subspace 相对删除等 Frobenius 范数正交 subspace 多增加的 absolute count error；0 表示两类同强度 residual intervention 没有差别，正值表示 count-aligned component 更具行为作用。每点先在 seed 内平均 counts 2–10，再对10个 seed 等权求均值；横线为95% seed-bootstrap CI。actual 与 centroid-curve 是两种不同删除量，不作为重复试验合并。</figcaption></figure>
{table(["population", "model", "removed subspace", "behavior endpoint", "specificity [95% CI]", "exact / Holm p"], subspace_behavior_rows, classes="paper-table compact-result-table")}
{details_table("Prompt subspace ablation 对下游 answer geometry 的全部结果", ["population", "model", "removed subspace", "answer layer", "specificity [95% CI]", "exact / Holm p"], subspace_rep_rows)}
<p><strong>全样本 familywise 正向结果：</strong>{significant_summary(subspace_stats, 'all')}。</p>
<p><strong>Clean-correct-only familywise 正向结果：</strong>{significant_summary(subspace_stats, 'clean_correct_only')}。</p>
<p><strong>结果分析。</strong>32 个注册的 population×model×removal×endpoint specificity 中，没有一项在各自 population 的 16 项 Holm family 内通过；本轮所有 Holm p 均为 1.0。全样本 absolute-error specificity 也很小：Qwen 的 actual / centroid-curve removal 为 +0.056 / +0.067，Gemma 为 −0.022 / −0.011，四个95% CI 均跨0；clean-correct-only 的行为效应同样接近0。下游 L29/L35/L37/L41 centroid-distance 也没有稳定、方向一致的 specificity。</p>
<p><strong>为什么会是 null。</strong>这里删除的是一个在 prompt L8/L9、跨 seeds 平均后得到的静态 rank-3 PCA basis，并假设它同时是自然计算的瓶颈。这个假设可能过强：模型可以把同一 count 以冗余方式分布在多个 token；后层可以从未干预的 literal spans 重新计算；真正的 carrier 可以随 layer 旋转、随 context 改变或高于3维；PCA 最大化描述方差，也不保证找到最有因果作用的方向。等范数正交 control 还可能同时删除通用的格式/位置支架，使 candidate−control specificity 变小。</p>
<div class="conclusion"><strong>实验 A 的结论（negative）。</strong>没有证据表明 Qwen L8 或 Gemma L9 中跨 seed 平均得到的固定三维 PCA plane，是模型无法绕开的必要 count bottleneck。这个 null 只否定“全网络沿同一个静态三维平面传递 count”的强假设，不否定 residual stream 中存在会旋转、可重算或更高维的 count carrier。</div>

<h4>5.4B · 实验 B：允许方向随层变化后，局部 residual subspace 能否搬运 count？——单锚点正结果</h4>
<div class="test-card"><p><strong>为什么做。</strong>若第 ℓ 层把 count 写在方向 <code>Uℓ</code>，下一层把它写在不同方向 <code>Uℓ+1</code>，直接删除或复制同一个 PCA vector 会漏掉真实传输；因此我们学习一个低秩 source→target map，而不要求两个层共享同一欧氏坐标。</p><p><strong>具体例子。</strong>对 receiver=1、donor=2，只把 source centroid 差 <code>μsource,2−μsource,1</code> 在冻结 transport basis 内的分量注入 receiver。若下一层沿 target 的1→2 centroid chord 前进约1单位，而等范数正交注入接近0，说明这个局部 subspace 能传递一单位 count；把同一分量加到2×后若前进接近2单位，则形成剂量响应。</p><p><strong>如何构造。</strong>只用 discovery count centroids，以 reduced-rank regression 将 source residual 映射到 target 的 rank-3 count coordinates，再对回归权重 row-space 做QR，得到冻结 source transport basis <code>B</code>。Confirmation 中注入 <code>ProjB(μD−μR)</code> 的1×和2×；control 位于 <code>B</code> 的正交补中，并与1× intervention 等范数。</p><p><strong>如何评估。</strong><code>target donor fraction</code> 是干预后 target state 沿 receiver→donor centroid chord 移动的比例：0表示未移动，1表示到达 donor centroid。要求同时满足 <code>aligned 1× &gt; orthogonal</code>（方向特异性）和 <code>aligned 2× &gt; aligned 1×</code>（剂量响应）。每个seed先平均1↔2、5↔6的双向 pairs，再对10个confirmation seeds做exact sign-flip与bootstrap。</p><p><strong>结果。</strong>Qwen L28→L29 的 orthogonal / aligned 1× / aligned 2× donor fraction 为0.007 / 0.949 / 1.810；Gemma L36→L37为0.002 / 0.976 / 1.801。两模型的1×方向特异性与2×增量均为 exact <code>p=0.001953</code>。完整并列图和置信区间见第7节。</p></div>
<div class="conclusion"><strong>实验 B 的结论（positive）。</strong>两个模型在测试的相邻 answer-query layers 间都存在 discovery-frozen、方向特异且近似剂量响应的局部 count transport subspace。该结果证明“局部 residual relay 可用于搬运 count”，但不证明所有层共享一个固定 basis，也不证明该局部通道是唯一通道。</div>
<!--TRANSPORT_ALIGNED_FULL-->
{table(
    ["更强的 subspace 证据", "具体做法", "它比静态 PCA removal 多解决什么", "当前状态"],
    [
        ["跨层表示对齐", "在 discovery 上用 reduced-rank regression / PLS / CCA 学 source→target count map，确认 basis 后冻结", "允许 count axis 随层旋转，不要求同一个 PCA 向量贯穿网络", "已实现：rank-3 transport basis"],
        ["方向特异的 subspace patch", "只 patch donor−receiver 在 transport basis 内的分量，并与等范数正交分量比较", "把‘能解码’升级为‘该方向足以搬运 count’", "已确认：Qwen L28→L29、Gemma L36→L37"],
        ["剂量响应", "比较 aligned 1× 与2×，要求 target movement 近似随剂量增加", "排除偶然的大范数扰动或单点非线性", "两模型 dose2−dose1 exact p=0.001953"],
        ["路径中介", "先做 source subspace patch，再在下游 writer/mediator 删除诱导分量", "检验信息是否真的沿指定 residual→writer 路径使用", "Qwen/Gemma 的后续 OV/residual tests 已分别闭合"],
        ["分布式 causal scrubbing", "跨多个 token、多个相邻层同时 patch或remove frozen low-rank coordinates", "处理冗余、重算和跨 token relay", "建议的下一步；尚未作为本文确认结果"],
    ],
    classes="paper-table compact-result-table",
)}
<p><strong>什么才足以支持“residual stream 传递 count”。</strong>仅有跨层 CKA/CCA 或 decoder accuracy 不够，因为那只是相似性。当前最强证据是第7节的 transport-aligned intervention：discovery-only 学到 source transport basis，confirmation 中只注入其 donor−receiver 分量；aligned 1× 明显超过等范数正交 control，2× 又超过1×。Qwen L28→L29 与 Gemma L36→L37 的两项 exact p 均为0.001953。这说明<strong>测试的 answer-query 相邻层之间确实存在可用、方向特异、近似剂量响应的 count-carrying residual subspace</strong>。</p>
<div class="conclusion"><strong>本小节结论。</strong>Subspace ablation 审计为 {html.escape(str(subspace_audit['status']))}。确认性 null 只否定“prompt L8/L9 的单一静态 rank-3 PCA curve 是必要瓶颈”；它不否定 subspace 传递。用允许跨层旋转的 transport-aligned basis，Qwen 与 Gemma 都得到局部因果 positive。严谨表述因此是“存在经测试相邻 answer layers 的局部 transport subspace”，而不是“全网络共享一个固定三维 counter plane”。</div>
<!--LAYERWISE_SUBSPACE_SWEEP-->
"""


def _build_extension_question_audit_legacy(
    token_stats: list[dict[str, str]],
    subspace_stats: list[dict[str, str]],
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    """One-row-per-question audit of reports/non-thinking extension.md."""

    def token_value(model: str, endpoint: str) -> str:
        row = _extension_stat(
            token_stats,
            population="all",
            model_label=model,
            endpoint=endpoint,
            estimand="specificity",
        )
        return (
            f'{fmt(float(row["mean"]), 3, signed=True)} '
            f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}], '
            f'Holm p={fmt_p(float(row["holm_p_within_population"]))}'
        )

    subspace_positive = [
        row
        for row in subspace_stats
        if row["population"] == "all"
        and float(row["ci95_low"]) > 0
        and float(row["holm_p_within_population"]) <= 0.05
    ]
    subspace_answer = (
        f"{len(subspace_positive)}/{len([row for row in subspace_stats if row['population'] == 'all'])} "
        "all-sample endpoints 为正且通过 population-wise Holm"
    )
    q_patch = causal_v2["primary_confirmation_family_summary"][
        "Qwen3-8B::answer_patching"
    ]
    g_patch = causal_v2["primary_confirmation_family_summary"][
        "Gemma4-E4B::answer_patching"
    ]
    rows = [
        ["Step 1", "Needle-end hidden states 是否低维？", "rank / variance", "十个 count centroids 的 rank-3 capture 很高（Qwen L8 0.988；Gemma L9 0.979），但全部 endpoint states 只有0.690/0.598。", "低维的是嵌在高维 residual 中的 count-centroid curve，不是全部 state cloud。", "§4.1"],
        ["Step 1", "Running-index curve 对 state 变化解释多少？线性还是非线性？", "held-out ridge / kNN", "Prompt L8/L9 ridge R²=0.945/0.719；代表层中 ridge 优于 distance-weighted kNN。", "存在可泛化、近似线性的 ordinal coordinate；不需要用非线性模型才能读出。", "§4.2"],
        ["Step 1", "常见分类器能否预测 count？", "六个固定 classifier", "所有层公开六条曲线；全局固定 primary 为 L2 logistic，机会水平0.1，未逐层挑最优算法。", "count 可离散读出；classifier 选择规则不利用单层结果。", "§4.2"],
        ["Step 1", "十个 count 是否形成紧密 clusters？", "silhouette / CH / DB", "代表层 cosine silhouette 很小或为负，尽管连续 count 解码良好。", "更像连续有序曲线加 context deformation，不像十个各向同性小球。", "§4.2"],
        ["Step 1", "Prompt geometry 的 noise 来自哪里？", "balanced two-way decomposition", "Qwen L8 count/seed-context/interaction=0.599/0.161/0.241；Gemma L9=0.385/0.229/0.386。", "Gemma 的 context-dependent deformation 更强；这里的 noise 是确定性 residual variation，不是采样噪声。", "§4.3"],
        ["Step 1", "所有 token 是否都遵循 prefix-count curve？", "frozen endpoint PCA + gated formulas", "Endpoint-gated incremental R² 为 Qwen 0.551、Gemma 0.326；ordinary/interior 的 ungated或span-gated曲线增量 R² 接近0或为负。", "经验式应含 endpoint gate：count curve 主要出现在 needle endpoint，不贯穿全部 prompt tokens。", "§4.4"],
        ["Step 1", "开头 instruction cue 是否生成 running counter？", "cue-present / cue-absent frozen-basis comparison", "去 cue 后 centroid CKA 仍为0.981–0.997，但 count×cue interaction 显著。", "Cue 不是有序拓扑的来源，但会调制 full-space state；主因果链仍只覆盖 cue-present。", "§4.5"],
        ["Step 1", "Earlier-span / induction-like attention 是否存在？", "discovery-ranked full-span head confirmation", "Qwen top-10 preference=0.859–0.979；Gemma=0.273–0.960；两模型每个head的模型内Holm p=0.019531。", "Qwen与Gemma均支持earlier-span aggregation；尚不足以命名为完整经典induction circuit。", "§5.1"],
        ["Step 1", "删除 outside-needle attention 会不会减少 noise？", "needle-only vs equal-key ordinary-only mask", "Needle-only 相对 clean 在两模型三项指标上均未改善；Qwen 相对 ordinary-only 有三项特异性，Gemma 仅 normalized-distance specificity 过 Holm。", "Outside context 不是纯噪声，而是格式/位置支架；该原始假说被否定。", "§5.2"],
        ["Step 1", "直接破坏 needle tokens 会怎样？", "equal-token-budget token corruption", f"All-sample specific absolute-error increase：Qwen {token_value('Qwen3-8B','absolute_error_increase')}；Gemma {token_value('Gemma4-E4B','absolute_error_increase')}。", "这回答输入 needle evidence 的必要性；不把 token corruption 等同于某一 hidden-state carrier 的必要性。", "§5.3"],
        ["Step 1", "Set-wide running-index subspace 是否因果影响 answer？", "all-endpoint rank-3 / centroid-curve removal", subspace_answer, "只把相对等范数正交控制、并经 Holm 的 behavior/answer-geometry endpoint 写成支持；其余明确作为 null 或几何扰动。", "§5.4"],
        ["Step 2", "Broad retrieval heads 是否真正有功能？", "full-span nested top-K matched ablation", "两模型均出现 Holm-significant ranked-minus-layer-matched-random behavior effects；剂量曲线非单调。", "存在分布式 retrieval bank；K 不是独立同质 head 的线性剂量，也不推出唯一最小 circuit。", "§8–9.3"],
        ["Step 2", "Retrieval 与 OV rewriting 是否跨层？", "pre-O OV / residual mediation", f"Qwen L28 H16/H19 natural-OV global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}；Gemma K2→L37→L41 global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}。", "Qwen 已定位 localized OV set；Gemma 已定位 distributed residual write，二者均支持 retrieval 后再写入。", "§10"],
        ["Step 3", "Answer-query state 是否也有可解码 count geometry？", "rank / regression / fixed classifiers / clustering", "Qwen L29 / Gemma L37 ridge R²=0.891/0.885；centroid rank-3 capture=0.947/0.981；correct-only 另作敏感性分析。", "Answer query 含稳定 count coordinate，但仍嵌在高维 state 中。", "§4.1–4.2, §6"],
        ["Step 3", "Answer state 是否能驱动最终数字？", "all-sample + correct-only state patch", f"All-sample mean transport Qwen={q_patch['mean_effect']:.3f}、Gemma={g_patch['mean_effect']:.3f}；correct-only donor adoption=96.6%/96.0%。", "Answer representation 是可执行 state，而不只是与 gold count 相关。", "§9.4"],
        ["Step 3", "不同层间是否存在可传播的 count subspace？", "transport-aligned adjacent-layer intervention", "Qwen L28→L29 与 Gemma L36→L37 的 aligned−orthogonal、dose2−dose1 均 exact p=0.001953；旧 final prompt endpoint support 接近0。", "两模型均有方向特异、近似剂量响应的局部 answer-query residual relay；不声称全层共享静态 PCA basis。", "§7"],
        ["Step 3", "层越深是否发生 consolidation？", "layerwise rank / decoding + causal write trace", "Answer centroids 维持高 rank-3 capture，held-out decodability 在中后层升高；Qwen write 沿 L29–L35 保留，Gemma K2 effect 经 L37 到 L41。", "Consolidation 定义为可解码性增强、相对 noise 减少和 late-state 可执行性，而不是只凭3D视觉变紧。", "§4, §7, §10"],
    ]
    return f"""
<section id="question-audit">
<h2>12 · Extension memo 逐题回答</h2>
<p>下表逐行对应 <code>reports/non-thinking extension.md</code>。正、负和范围受限的结果都保留；“未支持某个更强解释”不等于删除该问题，也不会被其他显著实验替代。</p>
{table(["阶段", "原问题", "主实验", "结果", "当前可写结论", "正文位置"], rows, classes="paper-table causal-ledger")}
<div class="conclusion"><strong>逐题审计结论。</strong>Memo 中的 representation、noise、attention、token/subspace causal、broad retrieval、OV/relay 与 answer consolidation 问题均已有对应实验和明确判定。当前最完整的共同机制是“endpoint-gated prompt state → distributed full-span retrieval → coordinate-changing residual write → local answer-state relay → executable digit output”；模型间差异只按通过的因果定位粒度书写。</div>
</section>
"""


def build_extension_question_audit(
    token_stats: list[dict[str, str]],
    subspace_stats: list[dict[str, str]],
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    gemma_residual: dict[str, Any],
    earlier_heads: list[dict[str, str]],
) -> str:
    """Readable, one-question-at-a-time audit of reports/non-thinking extension.md."""

    def token_value(model: str, endpoint: str) -> str:
        row = _extension_stat(
            token_stats,
            population="all",
            model_label=model,
            endpoint=endpoint,
            estimand="specificity",
        )
        return (
            f'{fmt(float(row["mean"]), 3, signed=True)} '
            f'[{fmt(float(row["ci95_low"]), 3)}, {fmt(float(row["ci95_high"]), 3)}], '
            f'Holm p={fmt_p(float(row["holm_p_within_population"]))}'
        )

    all_subspace = [row for row in subspace_stats if row["population"] == "all"]
    positive_subspace = [
        row
        for row in all_subspace
        if float(row["ci95_low"]) > 0
        and float(row["holm_p_within_population"]) <= 0.05
    ]
    q_patch = causal_v2["primary_confirmation_family_summary"][
        "Qwen3-8B::answer_patching"
    ]
    g_patch = causal_v2["primary_confirmation_family_summary"][
        "Gemma4-E4B::answer_patching"
    ]
    q_ov_p = fmt_p(ov["primary_decision"]["global_intersection_union_p"])
    g_ov_p = fmt_p(
        gemma_residual["primary_decision"]["global_intersection_union_p"]
    )
    earlier_ranges: dict[str, tuple[float, float]] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        values = [
            float(row["confirmation_preference_mean"])
            for row in earlier_heads
            if row["model_label"] == model
        ]
        if len(values) != 10:
            raise RuntimeError(f"Expected ten earlier-span heads for {model}")
        earlier_ranges[model] = (min(values), max(values))

    questions = [
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "Needle-end hidden states 是否真的是低维的？",
            "method": "把两件事分开算：一是全部 endpoint states 的协方差秩；二是十个 count centroids 组成的中心轨迹秩。报告 rank-3 对两者各自的方差捕获率，而不是只看三维 PCA 图。",
            "result": "在 prompt 代表层，rank-3 对 count-centroid curve 的捕获率为 Qwen L8 0.988、Gemma L9 0.979；但对全部 endpoint states 只有0.690和0.598，numeric rank-90 分别为30和25。",
            "answer": "低维的是嵌在高维 residual 中的共享 count-centroid trajectory；不能说全部 hidden-state cloud 只有三维。",
            "boundary": "三维图适合展示中心轨迹，不足以证明所有样本都被压在同一个三维平面。",
            "where": "§4.1",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "Count 能否被线性读出？这是否意味着 PCA 轨迹是直线？",
            "method": "PCA、标准化和预测器都只在 training seeds 拟合，再按 seed 做 held-out prediction。连续 count 同时比较 ridge 与 distance-weighted kNN-5；主要指标为 full-state held-out R²。",
            "result": "Prompt代表层ridge/kNN R²为Qwen L8 0.945/0.711、Gemma L9 0.719/0.374；answer代表层为Qwen L29 0.891/0.871、Gemma L37 0.885/0.773。两模型的PCA centroid paths仍有明显曲率。",
            "answer": "两模型的count都<strong>线性可读</strong>：存在hidden-state的线性组合可预测count。但这不等于geometry是一条直线；弯曲轨迹仍可在某个方向上单调投影。准确表述是“线性可读的有曲率ordinal trajectory”。",
            "boundary": "Ridge优于kNN可能还受高维context noise与样本量影响，不能据此否定曲率；高R²也不自动证明该decoder方向是自然forward的唯一必要通路。",
            "where": "§4.2",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "常见分类器能否从 hidden state 预测 count？如何避免每层挑最优算法？",
            "method": "固定比较 L2 logistic、ridge classifier、linear SVM、nearest centroid、shrinkage LDA 与 cosine kNN-5。先跨四个 model×site 的全部层求平均，只选一个全局 primary；结果是 L2 logistic。所有六条逐层曲线都公开。",
            "result": "全局平均 accuracy 最高的是 L2 logistic 0.371，nearest centroid 为0.370。Answer 代表层 all-sample accuracy 为 Qwen L29 0.560、Gemma L37 0.530；在 clean-correct rows 上 raw/balanced accuracy 分别为 Qwen 0.880/0.759、Gemma 0.918/0.625，均只剩8个 count classes。",
            "answer": "Count 可以被多类简单分类器离散读出；结论不是逐层挑算法所得。Correct-only accuracy 更高，但它改变了类别组成，因此只作为答案状态的敏感性分析。",
            "boundary": "Prompt running-index 必须保留完整 count×seed 网格，不能用最终答案 correct-only 筛选，否则会整条删除高-count trajectories。",
            "where": "§4.2",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "十个 count 是十个紧密 clusters，还是一条带 context 形变的连续轨迹？",
            "method": "在 train-fitted PCA 中对 held-out seeds 计算 cosine silhouette、Calinski–Harabasz 与 Davies–Bouldin；同时与连续 ridge decoding 对照。",
            "result": "Cosine silhouette 在 Qwen prompt L8为−0.075、Gemma prompt L9为−0.043；answer L29/L37为0.056/−0.038。尽管 cluster separation 很弱，连续 ridge R²仍为0.719–0.945（prompt）和0.885–0.891（answer）。",
            "answer": "Count geometry 更像一条有序曲线叠加 prompt-dependent offset/deformation，而不是十个互相孤立、近似球形的小簇。",
            "boundary": "Silhouette 低不代表没有 count information；它只否定“十个紧密球状簇”这个更强形状假说。",
            "where": "§4.2",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "Prompt geometry 的 noise 到底是什么，来自哪里？",
            "method": "对平衡的 count×seed endpoint tensor 做 two-way Frobenius/ANOVA：共享 count trajectory、prompt-wide seed/context offset、count×seed trajectory deformation。Seed 使用 categorical one-hot，不把 seed 编号当连续数值。",
            "result": "Qwen L8总变异的count/offset/interaction为0.599/0.161/0.241：noise占40.2%，其中prompt identity main effect解释40.0%，剩余60.0%是count×prompt deformation。Gemma L9为0.385/0.228/0.386：noise占61.4%，prompt identity解释37.1%，interaction占62.9%。",
            "answer": "两模型的noise都主要表现为轨迹形状随prompt改变，而不只是整条轨迹平移；Gemma的总noise比例和interaction绝对份额都更高。当前真正可称为feature解释量的是prompt-identity main effect：Qwen 40.0%、Gemma 37.1%的within-count noise。",
            "boundary": "Interaction是未被prompt-wide offset解释的cell-level形变，不是可对新prompt预测的具体feature。Seed把词汇、位置、顺序与段落内容捆在一起，因此各具体metadata feature的解释率尚未识别，而不是0%；V4.4.0的feature-regression来自native-thinking样本，不能移植到这里。",
            "where": "§4.3",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "每个 prompt token 都沿 prefix-count curve 排列吗？",
            "method": "只用 discovery needle endpoints 冻结 PCA-32、count centroids 与 curve，再投影 confirmation 的 endpoints、needle interiors、hard negatives 和 ordinary passage tokens。比较 endpoint-gated、whole-span-gated 与 ungated formulas。",
            "result": "Qwen L8 的 endpoint nearest-centroid accuracy为0.490，而 interior/hard-negative/ordinary仅0.016/0.008/0.025；endpoint-gated ΔR²=0.551，错误推广到 interior/ordinary 后均为−0.057。Gemma L9 endpoint accuracy=0.280，其他三类为0.116/0.110/0.096；endpoint-gated ΔR²=0.326，span/ordinary为−0.036/−0.041。",
            "answer": "Running-index geometry 是 endpoint-gated：完成一条 needle 后的 endpoint 最清楚，不能把同一 count curve 泛化为整个 needle span 或全部 prompt tokens 的普遍属性。",
            "boundary": "Interior 在 Gemma 有少量可解码性，但预注册的 endpoint curve formula 并未对其提供增量解释。",
            "where": "§4.4",
        },
        {
            "group": "12.1 · Prompt-side representation 与 noise",
            "question": "开头 instruction cue 创建了 running counter，还是只调节已有表征？",
            "method": "保持模型 mode 与 passage 完全相同，只删除开头 city-score 定义；用冻结 basis 比较 cue-present 与 cue-absent 的 centroid CKA、count η²与 full-space count×cue interaction。",
            "result": "各代表/探针层的 centroid CKA 为0.981–0.997，说明有序 topology 基本保留；同时 count×cue q 值约0.0010–0.0012，表明 cue 对 full-space state 的影响依赖 count。",
            "answer": "Cue 不是 running-counter topology 的唯一来源，但会显著调制该表征在 full residual space 中的具体位置与强度。",
            "boundary": "后续主机制因果链只在 cue-present 条件确认，不能据此声称 cue-absent 使用完全相同的 circuit。",
            "where": "§4.5",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "较早层是否会在当前 endpoint 回看已经出现的 needle spans？",
            "method": "用 discovery 按“当前 endpoint query 指向 prior needle full spans”的分数冻结top-10 heads；在10个 confirmation seeds 比较 prior-needle mass 与等长度、等深度 ordinary spans。",
            "result": f"Qwen top-10 heads 的 prior-needle minus matched-ordinary preference 为{earlier_ranges['Qwen3-8B'][0]:.3f}–{earlier_ranges['Qwen3-8B'][1]:.3f}；Gemma 为{earlier_ranges['Gemma4-E4B'][0]:.3f}–{earlier_ranges['Gemma4-E4B'][1]:.3f}。两模型每个 head 的 exact p=0.001953，模型内 Holm p=0.019531。",
            "answer": "<strong>Qwen与Gemma：</strong>均有明确的earlier-span aggregation；当前endpoint会特异回看先前needle full spans，而不是只偏好同长度、同深度的普通上下文。",
            "boundary": "该结果确认的是跨更早目标记录的聚合，不足以命名为包含previous-token matching与复制操作的完整经典induction circuit，也不证明top-10 heads单独完成计数。",
            "where": "§5.1",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "Outside-needle attention 是不是给 running counter 注入了纯 noise？",
            "method": "只改 occurrences 2/4/6/8/10 的 endpoint query：比较 clean、needle-only keys 与等 key-budget 的 matched ordinary-only keys。指标统一写成“mask后改善”，再比较 needle-only 与 ordinary-only specificity。",
            "result": "Needle-only 相对 clean 没有净化：Qwen continuous/accuracy/distance improvements 为−0.261/−0.280/−1.610；Gemma为−3.688/−0.080/−1.568。相对 ordinary-only，Qwen 三项 specificity 为1.174、0.160、2.221，Holm≤0.023438；Gemma只有 normalized distance 3.307 通过Holm p=0.011719。",
            "answer": "<strong>General：</strong>两模型删除ordinary context后都没有变干净，因此outside context不是纯噪声。<strong>Qwen：</strong>needle keys相对ordinary keys在三项指标上都有特异信息。<strong>Gemma：</strong>只确认normalized-distance specificity，行为误差与accuracy没有对应支持。",
            "boundary": "Mask 同时改变 attention normalization 与可见 key set，因此它回答“该受限读取条件是否更好”，不是对单个自然通路的纯 removal。",
            "where": "§5.2",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "如果把输入中的 needle literal spans 真正破坏掉，会怎样？",
            "method": "把每条 active needle 的完整 token span 替换为同 prompt 中等 token 数 ordinary sequence；matched control 替换位置深度和总 token budget 相同的 ordinary spans。序列长度、后续位置与 answer query 全部保持不变。",
            "result": f"相对 ordinary control，all-sample absolute-error specificity 为 Qwen {token_value('Qwen3-8B', 'absolute_error_increase')}；Gemma {token_value('Gemma4-E4B', 'absolute_error_increase')}。Accuracy-drop specificity 为 Qwen +0.450、Gemma +0.360，二者 Holm p=0.046875；clean-correct-only accuracy-drop specificity 为+0.950/+1.000。",
            "answer": "<strong>Qwen与Gemma：</strong>真实needle内容都是必要输入证据；替换等量ordinary tokens几乎没有相同损伤，因此结果不是一般文本扰动造成。",
            "boundary": "干预发生在输入层，所以会连带改变所有下游states。它能证明“模型需要needle信息”，但不能定位信息究竟由哪个endpoint、subspace或head set传递。",
            "where": "§5.3",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "是否存在一个固定的、跨 prompt endpoints 共用的 rank-3 count plane？",
            "method": "在 Qwen L8/Gemma L9 同时从全部 active endpoints 删除 discovery-frozen actual rank-3 plane 或 centroid-curve plane，并与等删除范数的正交 rank-3 control 比较。",
            "result": f"All-sample 的16个主 endpoints 中，正且通过 population-wise Holm 的结果为{len(positive_subspace)}/{len(all_subspace)}。例如 Qwen actual-plane removal 的 absolute-error specificity仅+0.056 [−0.011,0.133]，Gemma为−0.022 [−0.089,0.033]；二者Holm p=1。",
            "answer": "<strong>Qwen与Gemma：</strong>都没有证据表明代表性prompt层中的固定rank-3 PCA plane是模型无法绕开的必要瓶颈。删除该平面并不比等范数正交删除造成更大损伤。",
            "boundary": "Null只作用于“固定、三维、prompt-endpoint、必要性”这一组合。冗余存储、从literal spans重算、方向随层旋转或高于三维都能使该removal接近零；下一题改测晚层旋转通道的局部充分性。",
            "where": "§5.4A",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "如果允许 count direction 随层旋转，是否能直接证明 residual subspace 搬运 count？",
            "method": "只在 discovery centroids 上学习 source→target 的 rank-3 transport basis。Confirmation 中向 source state 注入 receiver→donor 的 aligned 1×或2×分量，并与等范数、位于 transport basis 正交补的方向比较。",
            "result": "Qwen L28→L29：aligned1−orthogonal=0.942 [0.913,0.967]，dose2−dose1=0.861 [0.842,0.879]；Gemma L36→L37为0.974 [0.962,0.986]与0.825 [0.818,0.833]。四个 exact p 均为0.001953。",
            "answer": "<strong>Qwen：</strong>L28→L29存在局部count-transport subspace。<strong>Gemma：</strong>L36→L37存在同类通道。两者都满足方向特异性和剂量响应。",
            "boundary": "实验A问“删掉固定早层plane是否必要”，实验B问“沿学习到的旋转晚层方向注入是否足以搬运”；因果问题、位置和basis都不同，所以A为null与B为positive不矛盾。B也不证明该局部通道是唯一通道。",
            "where": "§5.4B",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "按 full-span attention 排名的 broad-retrieval heads 是否真的影响 counting？",
            "method": "在 fresh seeds 1316–1335、counts 1–5 上，只在 final Total: query 将冻结 top-K heads 的 pre-O output slices 置零；每个K与三个 layer-matched random banks比较。主 effect 是 ranked ablation damage 减去 random damage。",
            "result": "Qwen all-sample 在K=4/16/32过Holm，effects分别0.0833/0.5200/1.6233；correct-only在K=16/32过Holm。Gemma all-sample K=1/2/4/8/16/32全部过Holm，correct-only在K=8/16过Holm；例如K=8 all effect=0.7667 [0.6067,0.9501]，Holm p=2.29e-05。",
            "answer": "两模型都使用分布式 full-span retrieval bank；影响超过删除同层随机 heads，因而 attention ranking 具有功能意义。",
            "boundary": "K-sweep 非单调，说明 heads 有冗余、协同或反向成员；不能把 K 当成同质 heads 的线性剂量，也不证明这是唯一 retrieval circuit。",
            "where": "§8.1–§8.3",
        },
        {
            "group": "12.2 · State formation 与因果必要性",
            "question": "读到 needle 之后，count 如何经过 OV/residual channel 写回？",
            "method": "在真实 pre-O head-space <code>z</code> 边界检验 natural carrier、count-step injection、centered removal 与 donor-path mediation；所有方向由 discovery 冻结，并与同层 matched sets 或同一W<sub>O</sub> span内等范数正交方向比较。Global IUT 取全部必要 families 中最大的p。",
            "result": f"Qwen L28 H16/H19：natural carrier 0.2174、pre-O injection 0.0640、centered-removal error 0.0732、path mediation 0.0136，global IUT p={q_ov_p}。Gemma K2 source→L37→L41：source transport 0.0889、exact L37 mediation 0.0864、count-axis mediation 0.0458、L41 adoption 0.2256，global IUT p={g_ov_p}。",
            "answer": "Qwen 的证据定位到局部两头OV write；Gemma 的证据定位到 full-attention K2 bank 写入L37 residual、再传播到L41。二者都支持“retrieval 后进行坐标转换并写入 answer residual”。",
            "boundary": "Qwen 与 Gemma 的定位粒度不同：不能把 Gemma 的 distributed residual write 强行写成和 Qwen 相同的单一 head-level transporter。",
            "where": "§9",
        },
        {
            "group": "12.3 · Answer execution、transport 与 consolidation",
            "question": "Answer-query residual 是否也有稳定的 count geometry？",
            "method": "在第一个答案 token 生成前保存 final Total: query residual；对全部层做 discovery-fit/held-out regression、rank、固定分类器与聚类分析。",
            "result": "Qwen L29/Gemma L37 的 ridge R²为0.891/0.885，centroid rank-3 capture为0.947/0.981；all-state rank-3 capture为0.729/0.799。",
            "answer": "Answer query 含有稳定、低维中心轨迹上的 count coordinate，但完整 answer states 仍然是高维的。",
            "boundary": "相关几何只说明可读出；下一题用 state patching 判断它是否可执行。",
            "where": "§4.1–§4.2、§6",
        },
        {
            "group": "12.3 · Answer execution、transport 与 consolidation",
            "question": "Answer state 能否因果驱动最终数字，而不只是和 gold count 相关？",
            "method": "把 donor count 的完整 answer-query residual patch 到 receiver count；all-sample 测 control-adjusted normalized transport，correct-only 只保留 donor/receiver clean 都正确的 pairs，测 receiver 是否改为 donor gold。",
            "result": f"All-sample mean transport 为 Qwen {q_patch['mean_effect']:.3f}、Gemma {g_patch['mean_effect']:.3f}，正CI条件分别149/149与177/177。Correct-only donor adoption 为96.6%/96.0%；fresh low-count source gain为Qwen +0.6776 [0.4564,0.8978], p=1.43e-05，Gemma +12.1734 [11.3239,12.9635], p=9.54e-07。",
            "answer": "Answer residual 是可执行的 count state：替换它会系统把输出改向 donor count，而不仅是读取到一个相关标签。",
            "boundary": "Full-state patch 同时搬运 count 与其他 donor-specific components；方向特异性由前一题的 low-rank transport/OV controls补充。",
            "where": "§8.4",
        },
        {
            "group": "12.3 · Answer execution、transport 与 consolidation",
            "question": "更深层是否在把 count state consolidation 成最终可输出数字？",
            "method": "联合查看逐层 centroid rank-3 capture、held-out decoding，以及已经冻结的跨层 transport与OV/residual causal trace；consolidation 定义为可解码性增强、相对 dispersion下降和终层可执行性收敛。",
            "result": "Answer centroids 在中后层保持0.947–0.981的代表层 rank-3 capture，Qwen 的已确认 write/transport 落在L28→L29并沿后层保留，Gemma 的K2 write在L37形成后由L41显著采用（effect 0.2256，p=9.54e-07）。",
            "answer": "数据支持中后层把检索后的 count state 转换并巩固为可供LM head执行的 answer coordinate；Qwen路径更局部，Gemma因sliding window表现为full-attention写入后由局部层传播。",
            "boundary": "“Consolidation”是多项预先定义指标的机制性综合，不是一项单独的 omnibus p-value，也不表示3D点云必须视觉上越来越紧。",
            "where": "§4、§5.4B、§9",
        },
    ]

    rendered: list[str] = []
    current_group = ""
    for index, item in enumerate(questions, start=1):
        if item["group"] != current_group:
            current_group = item["group"]
            rendered.append(f'<h3 class="audit-group-title">{current_group}</h3>')
        rendered.append(
            f"""
<details class="audit-question">
  <summary class="audit-question-summary"><span class="audit-index">Q{index:02d}</span><span class="audit-question-title">{item['question']}</span><span class="audit-location">正文 {item['where']}</span></summary>
  <div class="audit-question-body">
  <div class="audit-answer-grid">
    <div><strong>怎么测</strong><p>{item['method']}</p></div>
    <div><strong>具体结果</strong><p>{item['result']}</p></div>
    <div><strong>直接回答</strong><p>{item['answer']}</p></div>
  </div>
  <p class="audit-boundary"><strong>证据边界。</strong>{item['boundary']}</p>
  </div>
</details>
"""
        )

    return f"""
<section id="question-audit">
<h2>12 · Extension memo 逐题回答</h2>
<p>本节逐题对应 <code>reports/non-thinking extension.md</code>。每题固定分成“怎么测—具体结果—直接回答—证据边界”：正结果、负结果和范围受限的结果都保留，避免用一个显著实验替代另一个不同问题。</p>
{''.join(rendered)}
<div class="conclusion"><strong>逐题审计结论。</strong>目前最完整、同时有 representation 与 causal support 的共同机制是：needle endpoint 形成可读出的 running state；full-span retrieval bank 在 answer query 取回证据；模型把读出的内容转换到新的 answer-count coordinates；该 state 在相邻 answer layers 的方向特异 subspace 中传播，最后成为可执行数字。固定 prompt rank-3 plane 的全局必要性没有得到支持，但允许跨层旋转的局部 transport subspace 得到两模型的剂量响应因果支持。</div>
</section>
"""


def relay_gate_svg(metrics: list[dict[str, Any]]) -> str:
    width, height = 1040, 300
    box_w, gap, y = 185, 22, 70
    parts = [
        f'<svg class="stat-svg relay-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="relay-title relay-desc">',
        '<title id="relay-title">Registered tail-64 relay hypothesis fails downstream causal gates</title>',
        '<desc id="relay-desc">The selected late position set carries count and permits a mechanical first stage, but behavioral transport, OV mediation and natural removal fail their registered directional tests.</desc>',
    ]
    for idx, metric in enumerate(metrics):
        x = 20 + idx * (box_w + gap)
        passed = bool(metric["passed"])
        klass = "relay-pass" if passed else "relay-fail"
        parts.append(
            f'<rect class="relay-box {klass}" x="{x}" y="{y}" width="{box_w}" height="138" rx="7"/>'
        )
        parts.append(
            f'<text class="relay-mark" x="{x + box_w / 2}" y="{y + 32}" text-anchor="middle">{"✓" if passed else "×"}</text>'
        )
        parts.append(
            f'<text class="relay-heading" x="{x + box_w / 2}" y="{y + 60}" text-anchor="middle">{html.escape(str(metric["label"]))}</text>'
        )
        parts.append(
            f'<text class="relay-value" x="{x + box_w / 2}" y="{y + 91}" text-anchor="middle">{html.escape(str(metric["value"]))}</text>'
        )
        parts.append(
            f'<text class="relay-p" x="{x + box_w / 2}" y="{y + 118}" text-anchor="middle">{html.escape(str(metric["p"]))}</text>'
        )
        if idx < len(metrics) - 1:
            x1 = x + box_w + 4
            x2 = x + box_w + gap - 4
            parts.append(
                f'<line class="relay-arrow" x1="{x1}" y1="{y + 69}" x2="{x2}" y2="{y + 69}"/>'
            )
    parts.append(
        '<text class="relay-summary" x="520" y="258" text-anchor="middle">global IUT p = 0.9981 · registered serial path NOT SUPPORTED</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def mechanism_svg() -> str:
    width, height = 1180, 520
    nodes = [
        (
            20,
            82,
            200,
            122,
            "Prompt running index",
            ["needle-end states", "ordered across occurrences"],
        ),
        (
            252,
            82,
            200,
            122,
            "Broad retrieval bank",
            ["L23H28, L23H29", "L26H20, L27H18"],
        ),
        (
            484,
            82,
            220,
            122,
            "L28 mixed read",
            ["α-routing + V-content", "H16–H19 mediator set"],
        ),
        (
            736,
            82,
            196,
            122,
            "Natural OV write",
            ["pre-O z → W_O span", "H19 nonredundant"],
        ),
        (
            964,
            82,
            196,
            122,
            "Late answer state",
            ["L29–L35 propagation", "LM-head count distribution"],
        ),
    ]
    parts = [
        f'<svg class="stat-svg mechanism-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="mech-title mech-desc">',
        '<title id="mech-title">Supported non-thinking counting mechanism</title>',
        '<desc id="mech-desc">Prompt-side running-index representations are read by an early broad retrieval bank. Independent serial mediation supports transport through the L28 H16 to H19 set, which reads both attention routing and value content and writes a natural count component that propagates to the late answer state. A tested tail-64 relay is rejected.</desc>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#6750E8"/></marker><marker id="arrow-dashed" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#718096"/></marker></defs>',
    ]
    for idx, (x, y, w, h, title, lines) in enumerate(nodes):
        parts.append(
            f'<rect class="mech-node mech-{idx}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
        )
        parts.append(
            f'<text class="mech-heading" x="{x + w / 2}" y="{y + 34}" text-anchor="middle">{html.escape(title)}</text>'
        )
        for line_idx, line in enumerate(lines):
            parts.append(
                f'<text class="mech-sub" x="{x + w / 2}" y="{y + 68 + line_idx * 23}" text-anchor="middle">{html.escape(line)}</text>'
            )
        if idx < len(nodes) - 1:
            next_x = nodes[idx + 1][0]
            parts.append(
                f'<line class="mech-arrow" x1="{x + w + 5}" y1="{y + h / 2}" x2="{next_x - 8}" y2="{y + h / 2}" marker-end="url(#arrow)"/>'
            )
    parts.append(
        '<text class="mech-evidence" x="600" y="38" text-anchor="middle">solid arrows: causal transport/mediation supported · boxes: localization granularity of current evidence</text>'
    )
    parts.append(
        '<rect class="mech-negative" x="454" y="318" width="272" height="110" rx="10"/>'
    )
    parts.append(
        '<text class="mech-heading" x="590" y="351" text-anchor="middle">Rejected relay candidate</text>'
    )
    parts.append(
        '<text class="mech-sub" x="590" y="381" text-anchor="middle">pre-query non-slot tail-64</text>'
    )
    parts.append(
        '<text class="mech-sub" x="590" y="405" text-anchor="middle">carrier present; natural mediation absent</text>'
    )
    parts.append(
        '<line class="mech-dashed" x1="590" y1="318" x2="590" y2="214" marker-end="url(#arrow-dashed)"/>'
    )
    parts.append(
        '<text class="mech-boundary" x="590" y="479" text-anchor="middle">Not established: clean-run necessity of each early head, a unique tokenwise +1 operator, or cross-model identity of this microcircuit.</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def ablation_topk_svg_legacy() -> str:
    """Plot the observed magnitude of ranked-minus-random count-shift contrasts."""
    width, height = 1040, 520
    left, right, top, bottom = 108, 72, 58, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    series = {
        "Qwen3-8B": {"color": "#6750E8", "values": [(4, 0.425), (8, 0.025)]},
        "Gemma4-E4B": {"color": "#00D4B4", "values": [(4, 2.025), (8, 2.625)]},
    }
    ymax = 3.0

    def x(k: int) -> float:
        return left + (k - 4) / 4 * plot_w

    def y(value: float) -> float:
        return top + (ymax - value) / ymax * plot_h

    parts = [
        f'<svg class="stat-svg ablation-topk" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ablation-topk-title ablation-topk-desc">',
        '<title id="ablation-topk-title">Top-k ranked-bank ablation effect magnitude</title>',
        '<desc id="ablation-topk-desc">The horizontal axis is k, either four or eight heads. The vertical axis is the absolute ranked-minus-layer-matched-random generated-count shift. Qwen decreases from 0.425 to 0.025; Gemma increases from 2.025 to 2.625.</desc>',
    ]
    for tick in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
        yy = y(tick)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width - right}" y2="{yy:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{left - 14}" y="{yy + 4:.1f}" text-anchor="end">{tick:.1f}</text>'
        )
    for k in (4, 8):
        xx = x(k)
        parts.append(
            f'<line class="x-guide" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height - bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 28}" text-anchor="middle">{k}</text>'
        )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>'
    )
    parts.append(
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>'
    )
    for label, payload in series.items():
        color = payload["color"]
        values = payload["values"]
        path = " ".join(
            ("M" if idx == 0 else "L") + f" {x(k):.1f} {y(value):.1f}"
            for idx, (k, value) in enumerate(values)
        )
        parts.append(f'<path class="series-line" d="{path}" style="stroke:{color}"/>')
        for k, value in values:
            xx, yy = x(k), y(value)
            label_y = yy - 15 if value > 0.2 else yy - 17
            anchor = "start" if k == 4 else "end"
            label_x = xx + 12 if k == 4 else xx - 12
            parts.append(
                f'<circle class="series-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="8" style="fill:{color}"/>'
            )
            parts.append(
                f'<text class="point-label" x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" style="fill:{color}">{html.escape(label)} · {value:.3f}</text>'
            )
    parts.append(
        f'<text class="axis-label" x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle">top-k head-set size</text>'
    )
    parts.append(
        f'<text class="axis-label" transform="translate(25 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">|ranked − random count shift|</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def ablation_topk_svg(seed_confirmation: dict[str, Any]) -> str:
    """Fresh-seed frozen top-k ablation effects with seed-cluster CIs."""
    width, height = 1120, 520
    panels = [
        {
            "x0": 80,
            "width": 455,
            "title": "All examples: |generated-count shift|",
            "metric": "all_absolute_shift",
            "ymax": 0.30,
            "ylabel": "ranked - random |count shift|",
        },
        {
            "x0": 635,
            "width": 405,
            "title": "Clean-correct: correct-to-wrong excess",
            "metric": "clean_correct_to_wrong",
            "ymax": 0.20,
            "ylabel": "ranked - random failure rate",
        },
    ]
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00D4B4"}
    top, bottom = 82, 96
    plot_h = height - top - bottom
    parts = [
        f'<svg class="stat-svg ablation-topk" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ablation-topk-title ablation-topk-desc">',
        '<title id="ablation-topk-title">Fresh-seed frozen top-k head-bank ablation</title>',
        '<desc id="ablation-topk-desc">Two panels show ranked-minus-layer-matched-random ablation effects at frozen top-k values. Points are effects and vertical bars are seed-cluster bootstrap 95 percent confidence intervals.</desc>',
    ]
    for panel_index, panel in enumerate(panels):
        x0 = float(panel["x0"])
        panel_w = float(panel["width"])
        ymax = float(panel["ymax"])
        plot_left, plot_right = x0 + 58, x0 + panel_w - 18

        def x(k: int) -> float:
            return plot_left + (int(k) - 1) / 3 * (plot_right - plot_left)

        def y(value: float) -> float:
            return top + (ymax - float(value)) / ymax * plot_h

        parts.append(
            f'<text class="panel-title" x="{x0 + panel_w / 2:.1f}" y="34" text-anchor="middle">{html.escape(str(panel["title"]))}</text>'
        )
        for tick_index in range(5):
            tick = ymax * tick_index / 4
            yy = y(tick)
            parts.append(
                f'<line class="grid" x1="{plot_left:.1f}" y1="{yy:.1f}" x2="{plot_right:.1f}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{plot_left - 9:.1f}" y="{yy + 4:.1f}" text-anchor="end">{tick:.2f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{height - bottom}" x2="{plot_right:.1f}" y2="{height - bottom}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{top}" x2="{plot_left:.1f}" y2="{height - bottom}"/>'
        )
        for k in (1, 2, 3, 4):
            xx = x(k)
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{height - bottom + 25}" text-anchor="middle">{k}</text>'
            )
        parts.append(
            f'<text class="axis-label" x="{(plot_left + plot_right) / 2:.1f}" y="{height - 34}" text-anchor="middle">frozen top-k</text>'
        )
        parts.append(
            f'<text class="axis-label" transform="translate({x0 + 12:.1f} {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(str(panel["ylabel"]))}</text>'
        )
        for model in ("Qwen3-8B", "Gemma4-E4B"):
            model_rows = seed_confirmation["models"][model]
            points = []
            for k_text, metrics in sorted(
                model_rows.items(), key=lambda item: int(item[0])
            ):
                item = metrics[str(panel["metric"])]
                points.append((int(k_text), item))
            if len(points) > 1:
                path = " ".join(
                    ("M" if index == 0 else "L")
                    + f" {x(k):.1f} {y(item['effect']):.1f}"
                    for index, (k, item) in enumerate(points)
                )
                parts.append(
                    f'<path class="series-line" d="{path}" style="stroke:{colors[model]}"/>'
                )
            for k, item in points:
                xx, yy = x(k), y(item["effect"])
                low_y, high_y = y(item["ci95_low"]), y(item["ci95_high"])
                parts.append(
                    f'<line class="ci" x1="{xx:.1f}" y1="{low_y:.1f}" x2="{xx:.1f}" y2="{high_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<line class="cap" x1="{xx - 5:.1f}" y1="{low_y:.1f}" x2="{xx + 5:.1f}" y2="{low_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<line class="cap" x1="{xx - 5:.1f}" y1="{high_y:.1f}" x2="{xx + 5:.1f}" y2="{high_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                parts.append(
                    f'<circle class="series-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="7" style="fill:{colors[model]}"/>'
                )
                label_y = max(top + 12, high_y - 9)
                parts.append(
                    f'<text class="point-label" x="{xx:.1f}" y="{label_y:.1f}" text-anchor="middle" style="fill:{colors[model]}">{item["effect"]:.3f}</text>'
                )
        if panel_index == 0:
            parts.append(
                '<text class="legend-label" x="150" y="61" style="fill:#6750E8">● Qwen3-8B</text>'
            )
            parts.append(
                '<text class="legend-label" x="285" y="61" style="fill:#00D4B4">● Gemma4-E4B</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def ablation_topk_svg_fullspan(seed_confirmation: dict[str, Any]) -> str:
    """Plot the preregistered full-span K grid on a log2-like categorical axis."""

    width, height = 1180, 540
    models = ("Qwen3-8B", "Gemma4-E4B")
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00D4B4"}
    ks = sorted(
        {
            int(k)
            for model in models
            for k in seed_confirmation["models"][model]
        }
    )
    panels = (
        {
            "x0": 62,
            "width": 500,
            "title": "All samples",
            "metric": "all_absolute_shift",
            "ylabel": "ranked − random |count shift|",
        },
        {
            "x0": 650,
            "width": 468,
            "title": "Clean-correct only",
            "metric": "clean_correct_to_wrong",
            "ylabel": "ranked − random failure probability",
        },
    )
    top, bottom = 86, 102
    plot_h = height - top - bottom
    parts = [
        f'<svg class="stat-svg ablation-topk" viewBox="0 0 {width} {height}" role="img" aria-labelledby="fullspan-topk-title fullspan-topk-desc">',
        '<title id="fullspan-topk-title">Full-span-ranked nested head-bank ablation</title>',
        '<desc id="fullspan-topk-desc">For Qwen and Gemma, K equals 1, 2, 4, 8, 16, or 32. The left panel shows ranked-minus-layer-matched-random absolute generated-count shift over all samples. The right panel shows correct-to-wrong probability excess among baseline-correct samples. Filled markers pass Holm correction across twelve model-by-K tests within the endpoint; hollow markers do not.</desc>',
    ]
    for panel_index, panel in enumerate(panels):
        x0 = float(panel["x0"])
        panel_w = float(panel["width"])
        plot_left, plot_right = x0 + 70, x0 + panel_w - 18
        lows = [0.0]
        highs = [0.0]
        for model in models:
            for metrics in seed_confirmation["models"][model].values():
                item = metrics[str(panel["metric"])]
                lows.append(float(item["ci95_low"]))
                highs.append(float(item["ci95_high"]))
        raw_low, raw_high = min(lows), max(highs)
        span = max(raw_high - raw_low, 1e-6)
        ymin = min(0.0, raw_low - 0.08 * span)
        ymax = raw_high + 0.14 * span

        def x(k: int) -> float:
            return plot_left + ks.index(int(k)) / max(len(ks) - 1, 1) * (
                plot_right - plot_left
            )

        def y(value: float) -> float:
            return top + (ymax - float(value)) / (ymax - ymin) * plot_h

        parts.append(
            f'<text class="panel-title" x="{x0 + panel_w / 2:.1f}" y="34" text-anchor="middle">{html.escape(str(panel["title"]))}</text>'
        )
        for tick in _nice_ticks(ymin, ymax, 6):
            if tick < ymin or tick > ymax:
                continue
            yy = y(tick)
            parts.append(
                f'<line class="grid" x1="{plot_left:.1f}" y1="{yy:.1f}" x2="{plot_right:.1f}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{plot_left - 9:.1f}" y="{yy + 4:.1f}" text-anchor="end">{fmt(tick, 2)}</text>'
            )
        zero_y = y(0.0)
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{zero_y:.1f}" x2="{plot_right:.1f}" y2="{zero_y:.1f}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{plot_left:.1f}" y1="{top}" x2="{plot_left:.1f}" y2="{height-bottom}"/>'
        )
        for k in ks:
            xx = x(k)
            parts.append(
                f'<line class="x-guide" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}"/>'
            )
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle">{k}</text>'
            )
        parts.append(
            f'<text class="axis-label" x="{(plot_left+plot_right)/2:.1f}" y="{height-30}" text-anchor="middle">full-span-ranked head-set size K</text>'
        )
        parts.append(
            f'<text class="axis-label" transform="translate({x0 + 14:.1f} {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(str(panel["ylabel"]))}</text>'
        )
        for model in models:
            points = []
            for k_text, metrics in sorted(
                seed_confirmation["models"][model].items(),
                key=lambda item: int(item[0]),
            ):
                points.append((int(k_text), metrics[str(panel["metric"])]))
            path = " ".join(
                ("M" if index == 0 else "L")
                + f" {x(k):.1f} {y(item['effect']):.1f}"
                for index, (k, item) in enumerate(points)
            )
            parts.append(
                f'<path class="series-line" d="{path}" style="stroke:{colors[model]}"/>'
            )
            for k, item in points:
                xx, yy = x(k), y(float(item["effect"]))
                low_y, high_y = y(float(item["ci95_low"])), y(
                    float(item["ci95_high"])
                )
                holm = float(item["holm_p_across_twelve_frozen_sets"])
                fill = colors[model] if holm <= 0.05 else "#FBFAF5"
                parts.append(
                    f'<line class="ci" x1="{xx:.1f}" y1="{low_y:.1f}" x2="{xx:.1f}" y2="{high_y:.1f}" style="stroke:{colors[model]}"/>'
                )
                if model == "Qwen3-8B":
                    parts.append(
                        f'<circle class="series-dot" cx="{xx:.1f}" cy="{yy:.1f}" r="7" style="fill:{fill};stroke:{colors[model]}"/>'
                    )
                else:
                    parts.append(
                        f'<rect x="{xx-7:.1f}" y="{yy-7:.1f}" width="14" height="14" rx="2" style="fill:{fill};stroke:{colors[model]};stroke-width:2"/>'
                    )
        if panel_index == 0:
            parts.append(
                '<text class="legend-label" x="146" y="61" style="fill:#6750E8">● Qwen3-8B</text>'
            )
            parts.append(
                '<text class="legend-label" x="288" y="61" style="fill:#00D4B4">■ Gemma4-E4B</text>'
            )
            parts.append(
                '<text class="legend-label" x="442" y="61">filled = 12-way Holm p≤.05</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def build_mechanism_overview(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_story: dict[str, Any],
) -> str:
    global_ov_p = float(ov["primary_decision"]["global_intersection_union_p"])
    upstream_p = float(upstream["primary_decision"]["intersection_union_p"])
    read_metric_names = {
        "read_routing_behavior_transport",
        "read_value_behavior_transport",
    }
    rw = {
        row["metric"]: row
        for row in read_write["summary"]
        if row.get("stratum") == "all" and row.get("metric") in read_metric_names
    }
    routing_p = float(rw["read_routing_behavior_transport"]["exact_sign_flip_p"])
    value_p = float(rw["read_value_behavior_transport"]["exact_sign_flip_p"])
    gemma_status = html.escape(str(gemma_story["summary"]))
    gemma_p = (
        fmt_p(gemma_story.get("global_p"))
        if gemma_story.get("global_p") is not None
        else "未形成全局通过值"
    )
    return f"""
<section id="mechanism-overview" class="mechanism-main">
<div class="main-figure-kicker">PAPER MAIN FIGURE · SHARED COMPUTATION + MODEL-SPECIFIC CAUSAL RESOLUTION</div>
<h2>Non-thinking counting：从读 prompt 到写出 <code>Total:N</code></h2>
<p class="figure-intro">点击“下一步”或“播放一次”查看五阶段计算。关键新增点是：<strong>OV 不是可省略的命名，而是把 attention head-space 中读出的 state 写入 residual 坐标系的线性变换</strong>。框内 layer/head 是 Qwen 已闭合路径的具体实例；Gemma 采用由强到弱的冻结证据阶梯，不强迫复制 Qwen 的 head identity。当前 Gemma 判定为：{gemma_status}</p>
<figure class="mechanism-walkthrough" aria-labelledby="walkthrough-caption">
<div class="mechanism-canvas-wrap">
<svg viewBox="0 0 1180 430" role="img" aria-labelledby="walk-main-title walk-main-desc">
<title id="walk-main-title">Stepwise non-thinking counting mechanism</title>
<desc id="walk-main-desc">Five stages show repeated records forming a running-index state, a broad retrieval bank, a routing and value read, an output projection that changes representation coordinates while writing to the residual stream, and a late answer state that produces Total colon N.</desc>
<defs><marker id="walk-arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="currentColor"/></marker></defs>
<g class="walk-input" data-walk-step="0">
  <rect x="18" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="113" y="112" text-anchor="middle">Repeated records</text>
  <text class="walk-token" x="43" y="151">① city · score</text><text class="walk-token" x="43" y="184">② city · score</text>
  <text class="walk-token" x="43" y="217">…</text><text class="walk-token" x="43" y="250">⑩ city · score</text>
  <text class="walk-sub" x="113" y="282" text-anchor="middle">10k-token haystack</text>
</g>
<path class="walk-edge" data-walk-edge="1" d="M214 191 L272 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="1">
  <rect x="280" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="375" y="112" text-anchor="middle">Running index</text>
  <path class="mini-manifold" d="M310 258 C325 241 334 229 346 212 S369 178 386 168 S419 142 442 128"/>
  <circle cx="310" cy="258" r="7"/><circle cx="346" cy="212" r="7"/><circle cx="386" cy="168" r="7"/><circle cx="442" cy="128" r="7"/>
  <text class="walk-sub" x="375" y="282" text-anchor="middle">needle-end residuals</text>
</g>
<path class="walk-edge" data-walk-edge="2" d="M476 191 L534 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="2">
  <rect x="542" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="637" y="112" text-anchor="middle">Broad retrieval</text>
  <text class="walk-head" x="637" y="157" text-anchor="middle">L23H28 · L23H29</text>
  <text class="walk-head" x="637" y="190" text-anchor="middle">L26H20 · L27H18</text>
  <path class="fan-line" d="M578 235 L620 209 M610 248 L632 209 M660 209 L692 246"/>
  <text class="walk-sub" x="637" y="282" text-anchor="middle">distributed slot-state read</text>
</g>
<path class="walk-edge" data-walk-edge="3" d="M738 191 L796 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node" data-walk-step="3">
  <rect x="804" y="80" width="190" height="222" rx="12"/>
  <text class="walk-title" x="899" y="112" text-anchor="middle">Read → OV coordinate write</text>
  <text class="walk-head" x="899" y="151" text-anchor="middle">Qwen L28 · H16/H19</text>
  <text class="walk-formula" x="899" y="190" text-anchor="middle">z<tspan baseline-shift="sub">S</tspan> = {{Σ α<tspan baseline-shift="sub">h</tspan>V<tspan baseline-shift="sub">h</tspan>}}</text>
  <text class="walk-formula" x="899" y="225" text-anchor="middle">w<tspan baseline-shift="sub">S</tspan> = Σ W<tspan baseline-shift="sub">O</tspan><tspan baseline-shift="super">h</tspan>z<tspan baseline-shift="sub">h</tspan></text>
  <text class="walk-sub" x="899" y="258" text-anchor="middle">count preserved; coordinates may rotate</text>
  <text class="walk-sub" x="899" y="282" text-anchor="middle">u<tspan baseline-shift="sub">P</tspan> need not be parallel to u<tspan baseline-shift="sub">A</tspan></text>
</g>
<path class="walk-edge" data-walk-edge="4" d="M1000 191 L1050 191" marker-end="url(#walk-arrow)"/>
<g class="walk-node walk-output" data-walk-step="4">
  <rect x="1058" y="80" width="104" height="222" rx="12"/>
  <text class="walk-title" x="1110" y="112" text-anchor="middle">Answer</text>
  <text class="walk-answer" x="1110" y="188" text-anchor="middle">Total:</text>
  <text class="walk-answer-number" x="1110" y="234" text-anchor="middle">N</text>
  <text class="walk-sub" x="1110" y="282" text-anchor="middle">L29–L35</text>
</g>
<text class="walk-boundary walk-model-status" x="590" y="342" text-anchor="middle">Qwen: localized natural OV confirmed, global IUT p={fmt_p(global_ov_p)}</text>
<text class="walk-boundary walk-model-status" x="590" y="366" text-anchor="middle">Gemma: {html.escape(str(gemma_story["label"]))} effective residual write, p={gemma_p}; localized OV set unresolved</text>
<text class="walk-boundary" x="590" y="402" text-anchor="middle">solid path = causal transport/mediation support; node width does not encode effect size</text>
</svg>
</div>
<div class="mechanism-controls" aria-label="Mechanism animation controls">
  <button type="button" id="mechanism-prev">← 上一步</button>
  <button type="button" id="mechanism-play">▶ 播放一次</button>
  <button type="button" id="mechanism-next">下一步 →</button>
  <div class="step-dots" role="group" aria-label="直接选择机制阶段">
    <button type="button" data-mechanism-step="0" aria-label="步骤 1">1</button><button type="button" data-mechanism-step="1" aria-label="步骤 2">2</button>
    <button type="button" data-mechanism-step="2" aria-label="步骤 3">3</button><button type="button" data-mechanism-step="3" aria-label="步骤 4">4</button>
    <button type="button" data-mechanism-step="4" aria-label="步骤 5">5</button>
  </div>
</div>
<div id="mechanism-live" class="mechanism-live" aria-live="polite"></div>
<figcaption id="walkthrough-caption"><strong>Main Figure · Stepwise non-thinking mechanism.</strong> 图中没有数值坐标轴；高亮按时间顺序展示抽象证据链。Qwen early→L28 fresh-seed IUT p={fmt_p(upstream_p)}，routing/value p={fmt_p(routing_p)} / {fmt_p(value_p)}，natural-OV global IUT p={fmt_p(global_ov_p)}；Gemma 的最强完整证据层级为 <code>{html.escape(str(gemma_story["kind"]))}</code>，联合 p={gemma_p}。OV 框表示 head output 经 <em>W</em><sub>O</sub> 写回 residual，而不是假定 prompt 与 answer 的 count axis 是同一向量。所有 effect、CI 与单门定义见第 8–10 节。</figcaption>
</figure>
<div class="plain-protocol ov-coordinate-note">
<h4>为什么 prompt counter 与 answer counter 可以不在同一方向？</h4>
<ol>
  <li>在 prompt 位置，用单位向量 <code>u<sub>P</sub></code> 表示 occurrence index 的 residual-space count direction。</li>
  <li>answer query 的 head set 先形成 <code>z<sub>S</sub>(q,c)={{Σ<sub>j</sub>α<sub>h</sub>(q,j)W<sub>V</sub><sup>g(h)</sup>x<sub>j</sub>}}<sub>h∈S</sub></code>；这是 head-space state，不要求与 <code>u<sub>P</sub></code> 共线。</li>
  <li>OV 写回为 <code>w<sub>S</sub>(c)=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q,c)</code>；后续 attention/MLP 的局部 Jacobian 继续传播，因此 <code>u<sub>A</sub> ∝ J<sub>ℓ→A</sub>w<sub>S</sub></code>。</li>
</ol>
<div class="equation">保留的是 count ordering / decodability / causal transport；不要求 u<sub>P</sub> ∥ u<sub>A</sub>。</div>
</div>
{table(
    ["模型", "当前写入证据", "允许的机制表述"],
    [
        [
            "Qwen3-8B",
            f"L28 H16/H19 natural signal + true pre-O injection + centered removal + mediation；global IUT p={fmt_p(global_ov_p)}",
            "已定位 set-level OV 坐标变换/写回，再沿 L29–L35 传播",
        ],
        [
            "Gemma4-E4B",
            f"localized OV 候选未闭合；{html.escape(str(gemma_story['label']))} distributed residual path p={gemma_p}",
            "已确认有效的分布式 residual 写入；尚不能指定唯一 W_O head set",
        ],
    ],
)}
<div class="conclusion"><strong>这张图的主张</strong>non-thinking counting 是“prompt running state 被读取，经 OV/后续 block 改换坐标并写成 answer state”。OV 在计算图上必然存在，但“存在 OV 运算”与“某个冻结 head set 已被因果定位”是两个命题：Qwen 两者都成立；Gemma 目前只闭合到 distributed effective write。第 3–11 节给出表征、干预、p 值与边界。</div>
</section>
"""


def build_running_index_block() -> str:
    return """
<div class="figure-block running-index-block">
<h3>3.1 Running-index 3D · 逐个 occurrence 播放</h3>
<div class="study-preface"><strong>为什么做。</strong><span>如果模型在读 prompt 时维护累计状态，那么同一条 N=10 prompt 中，第 1 到第 10 个 active needle endpoint 的 residual 应随读取进度系统变化，而不是只在最终答案位置突然出现 count。</span><strong>定义与评估。</strong><span>第 n 个 endpoint state 定义为 <code>h<sup>P</sup><sub>s,n,ℓ</sub>=h<sub>ℓ</sub>(t<sup>end</sup><sub>s,n</sub>)</code>。PCA basis 只在 disjoint discovery rows 上拟合并冻结；V4.4 的30个 seeds只做 out-of-sample 投影。3D 图用于展示轨迹，正式效应由完整 residual 空间的 count regression、cross-validation 与后续因果实验判断。</span><strong>图中怎么看。</strong><span>PC1/PC2/PC3 是冻结 basis 的前三个方向；颜色表示 n=1…10，半透明小点是 seed-level states，大点是每个 n 的 centroid。轨迹有序说明 running index 可解码，但不单独证明模型必然使用它。</span></div>
<p class="figure-intro">这张图先把 prompt counter 解释成“读取进度”：播放 n=1→10 时，彩色大点沿冻结 PCA 空间前进；当前 n 的半透明小点是 30 个 V4.4 seeds 的真实 needle-end states。它先展示 centroid trajectory，下一张 3D 再提供 layer、split、outcome 与 PC 轴的完整交互。</p>
<figure>
<div class="running-controls">
  <label>model <select id="running-model"><option value="Qwen3-8B">Qwen3-8B</option><option value="Gemma4-E4B">Gemma4-E4B</option></select></label>
  <button type="button" id="running-prev">← n−1</button>
  <button type="button" id="running-play">▶ 播放一次</button>
  <button type="button" id="running-next">n+1 →</button>
  <label class="running-slider">running index <input id="running-step" type="range" min="1" max="10" value="1" step="1" aria-label="Running index from one to ten"></label>
</div>
<div class="plot-shell running-shell"><canvas id="running-index-canvas" aria-label="Interactive three-dimensional running-index centroid trajectory"></canvas></div>
<div id="running-status" class="running-status" aria-live="polite"></div>
<figcaption><strong>图 3A · Frozen-PCA 中的 running-index trajectory。</strong>横轴、纵轴、深度轴分别是 PC1、PC2、PC3；Qwen 默认使用 prompt manifold-display L8，Gemma 使用其数据中标记的 manifold-display layer。PCA basis 在 disjoint V4.1 discovery rows 上冻结，随后投影全部30个 V4.4 seeds（1234–1263）；所以播放不会重新拟合 basis。拖拽旋转、滚轮缩放。三维距离只作可视化，显著性与效应大小仍由 full-space 统计和因果实验判断。</figcaption>
</figure>
<div class="conclusion"><strong>图的判读</strong>轨迹有序说明 occurrence index 在 residual 中可解码；它本身不说明模型是否读取这条轴，也不等于每个 n 都被一个单独 head 保存。</div>
</div>
"""


def extract_embedded_json(document: str, variable_name: str) -> dict[str, Any]:
    marker = f"const {variable_name}="
    start = document.find(marker)
    if start < 0:
        raise RuntimeError(f"Could not locate embedded {variable_name}")
    start += len(marker)
    end = document.find(";\nconst ", start)
    if end < 0:
        end = document.find(";</script>", start)
    if end < 0:
        raise RuntimeError(f"Could not locate end of embedded {variable_name}")
    return json.loads(document[start:end])


def _centroid_distance_correlation(
    left_rows: list[list[Any]], right_rows: list[list[Any]]
) -> float:
    def centroids(rows: list[list[Any]]) -> list[list[float]]:
        grouped: dict[int, list[list[float]]] = {}
        for row in rows:
            grouped.setdefault(int(row[5]), []).append(
                [float(row[6]), float(row[7]), float(row[8])]
            )
        result: list[list[float]] = []
        for count in range(1, 11):
            points = grouped[count]
            result.append(
                [
                    sum(point[axis] for point in points) / len(points)
                    for axis in range(3)
                ]
            )
        return result

    def distances(points: list[list[float]]) -> list[float]:
        return [
            math.sqrt(
                sum((points[i][axis] - points[j][axis]) ** 2 for axis in range(3))
            )
            for i in range(len(points))
            for j in range(i + 1, len(points))
        ]

    left = distances(centroids(left_rows))
    right = distances(centroids(right_rows))
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    return numerator / math.sqrt(left_ss * right_ss)


def _answer_error_rows(answer_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Recover one copy of the embedded V4.4 discovery behavior labels.

    The self-contained base HTML repeats identical behavior metadata at every
    saved layer.  Its compact row schema is
    ``seed, split, outcome, parsed_count, count_error, gold_count, PCs...``.
    Selecting the lowest all-fit layer avoids double counting while retaining
    the exact per-prompt predictions used by the geometry viewer.
    """

    result: dict[str, list[dict[str, Any]]] = {}
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        candidates = sorted(
            (
                item
                for item in answer_data.values()
                if str(item.get("model")) == model
                and str(item.get("fit_cohort")) == "all"
            ),
            key=lambda item: int(item["layer"]),
        )
        if not candidates:
            raise RuntimeError(f"No all-fit answer dataset for {model}")
        rows: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for raw in candidates[0]["rows"]:
            if len(raw) < 6:
                raise RuntimeError("Embedded answer row has fewer than six fields")
            seed = int(raw[0])
            prediction = None if raw[3] is None else int(raw[3])
            error = None if raw[4] is None else int(raw[4])
            gold = int(raw[5])
            key = (seed, gold)
            if key in seen:
                raise RuntimeError(f"Duplicate embedded behavior row: {model}/{key}")
            seen.add(key)
            if prediction is None or error is None:
                if str(raw[2]) != "invalid":
                    raise RuntimeError("Only invalid rows may lack a parsed count")
            elif prediction - gold != error:
                raise RuntimeError(
                    f"Embedded count error mismatch: {model}/seed{seed}/N{gold}"
                )
            rows.append(
                {
                    "seed": seed,
                    "outcome": str(raw[2]),
                    "prediction": prediction,
                    "error": error,
                    "gold": gold,
                }
            )
        if len(rows) != 200 or len({row["seed"] for row in rows}) != 20:
            raise RuntimeError(
                f"Expected 20 discovery seeds x 10 counts for {model}; got {len(rows)}"
            )
        result[model] = rows
    return result


def _cluster_bootstrap_error_ci(
    rows: list[dict[str, Any]], *, repetitions: int = 5000, seed: int = 20260806
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seeds = sorted(by_seed)
    estimates: dict[str, list[float]] = {
        "overall_mae": [],
        "wrong_mae": [],
        "signed_error": [],
        "wrong_signed_error": [],
    }
    for _ in range(int(repetitions)):
        sampled = [rng.choice(seeds) for _ in seeds]
        sample = [row for sampled_seed in sampled for row in by_seed[sampled_seed]]
        valid_errors = [float(row["error"]) for row in sample if row["error"] is not None]
        wrong_errors = [value for value in valid_errors if value != 0]
        estimates["overall_mae"].append(
            sum(abs(value) for value in valid_errors) / len(valid_errors)
        )
        estimates["wrong_mae"].append(
            sum(abs(value) for value in wrong_errors) / len(wrong_errors)
        )
        estimates["signed_error"].append(sum(valid_errors) / len(valid_errors))
        estimates["wrong_signed_error"].append(
            sum(wrong_errors) / len(wrong_errors)
        )

    def interval(values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        low = ordered[int(0.025 * (len(ordered) - 1))]
        high = ordered[int(0.975 * (len(ordered) - 1))]
        return float(low), float(high)

    return {name: interval(values) for name, values in estimates.items()}


def _absolute_deviation_svg(
    summaries: dict[str, dict[str, Any]], per_count: dict[str, dict[int, float]]
) -> str:
    width, height = 1120, 470
    left, right, top, bottom = 76, 34, 34, 74
    gap = 90
    panel_width = (width - left - right - gap) / 2
    plot_height = height - top - bottom
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00D4B4"}
    parts = [
        f'<svg class="stat-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="absdev-title absdev-desc">',
        '<title id="absdev-title">Absolute deviation among V4.4 discovery errors</title>',
        '<desc id="absdev-desc">Left: number of wrong predictions by absolute deviation. Right: wrong-only mean absolute deviation by gold count. Qwen is purple and Gemma is green.</desc>',
    ]
    # Left panel: histogram counts.
    max_frequency = max(
        max(summary["absolute_histogram"].values(), default=0)
        for summary in summaries.values()
    )
    panel_left = left
    for tick in range(0, max_frequency + 1, 10):
        y = top + plot_height - (tick / max_frequency) * plot_height
        parts.append(f'<line class="grid" x1="{panel_left}" y1="{y:.1f}" x2="{panel_left + panel_width}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{panel_left - 9}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    group_width = panel_width / 5
    bar_width = group_width * 0.30
    for deviation in range(1, 6):
        center = panel_left + (deviation - 0.5) * group_width
        for offset, model in ((-0.55, "Qwen3-8B"), (0.55, "Gemma4-E4B")):
            value = int(summaries[model]["absolute_histogram"].get(deviation, 0))
            bar_height = (value / max_frequency) * plot_height
            x = center + offset * bar_width - bar_width / 2
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[model]}" opacity="0.88"/>')
        parts.append(f'<text class="tick" x="{center:.1f}" y="{top + plot_height + 24}" text-anchor="middle">{deviation}</text>')
    parts.append(f'<text class="axis-label" x="{panel_left + panel_width / 2:.1f}" y="{height - 16}" text-anchor="middle">absolute deviation |prediction − gold|</text>')
    parts.append(f'<text class="axis-label" transform="translate(17 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">number of wrong prompts</text>')

    # Right panel: conditional mean deviation by count.
    panel_left = left + panel_width + gap
    ymax = 3.25
    for tick in (0, 1, 2, 3):
        y = top + plot_height - (tick / ymax) * plot_height
        parts.append(f'<line class="grid" x1="{panel_left}" y1="{y:.1f}" x2="{panel_left + panel_width}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{panel_left - 9}" y="{y + 4:.1f}" text-anchor="end">{tick}</text>')
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        points: list[tuple[float, float]] = []
        for count, value in sorted(per_count[model].items()):
            x = panel_left + (count - 1) / 9 * panel_width
            y = top + plot_height - (value / ymax) * plot_height
            points.append((x, y))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colors[model]}" stroke-width="2.5"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{colors[model]}"/>')
    for count in range(1, 11):
        x = panel_left + (count - 1) / 9 * panel_width
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle">{count}</text>')
    parts.append(f'<text class="axis-label" x="{panel_left + panel_width / 2:.1f}" y="{height - 16}" text-anchor="middle">gold count</text>')
    parts.append(f'<text class="axis-label" transform="translate({panel_left - 54:.1f} {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">wrong-only mean absolute deviation</text>')
    parts.append(f'<circle cx="{panel_left + 15}" cy="18" r="5" fill="{colors["Qwen3-8B"]}"/><text class="tick" x="{panel_left + 26}" y="22">Qwen3-8B</text>')
    parts.append(f'<circle cx="{panel_left + 130}" cy="18" r="5" fill="{colors["Gemma4-E4B"]}"/><text class="tick" x="{panel_left + 141}" y="22">Gemma4-E4B</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_absolute_deviation_section(answer_data: dict[str, Any]) -> str:
    behavior = _answer_error_rows(answer_data)
    summaries: dict[str, dict[str, Any]] = {}
    per_count: dict[str, dict[int, float]] = {}
    for model, rows in behavior.items():
        valid = [row for row in rows if row["error"] is not None]
        wrong = [row for row in valid if int(row["error"]) != 0]
        errors = [int(row["error"]) for row in valid]
        wrong_errors = [int(row["error"]) for row in wrong]
        intervals = _cluster_bootstrap_error_ci(rows)
        histogram = {
            deviation: sum(abs(value) == deviation for value in wrong_errors)
            for deviation in range(1, max(map(abs, wrong_errors)) + 1)
        }
        summaries[model] = {
            "rows": len(rows),
            "valid": len(valid),
            "invalid": len(rows) - len(valid),
            "correct": len(valid) - len(wrong),
            "wrong": len(wrong),
            "accuracy": (len(valid) - len(wrong)) / len(rows),
            "overall_mae": sum(map(abs, errors)) / len(errors),
            "overall_signed_error": sum(errors) / len(errors),
            "total_signed_error": sum(errors),
            "wrong_mae": sum(map(abs, wrong_errors)) / len(wrong_errors),
            "wrong_median": sorted(map(abs, wrong_errors))[len(wrong_errors) // 2],
            "wrong_signed_error": sum(wrong_errors) / len(wrong_errors),
            "under": sum(value < 0 for value in wrong_errors),
            "over": sum(value > 0 for value in wrong_errors),
            "absolute_histogram": histogram,
            "ci": intervals,
        }
        per_count[model] = {}
        for count in range(1, 11):
            count_errors = [
                int(row["error"])
                for row in wrong
                if int(row["gold"]) == count
            ]
            if count_errors:
                per_count[model][count] = sum(map(abs, count_errors)) / len(count_errors)

    summary_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        row = summaries[model]
        overall_ci = row["ci"]["overall_mae"]
        wrong_ci = row["ci"]["wrong_mae"]
        signed_ci = row["ci"]["signed_error"]
        wrong_signed_ci = row["ci"]["wrong_signed_error"]
        summary_rows.append(
            [
                model,
                f"{row['correct']} / {row['rows']} ({100 * row['accuracy']:.1f}%)",
                f"{fmt(row['overall_mae'], 3)} [{fmt(overall_ci[0], 3)}, {fmt(overall_ci[1], 3)}]",
                f"{fmt(row['overall_signed_error'], 3, signed=True)} [{fmt(signed_ci[0], 3)}, {fmt(signed_ci[1], 3)}]",
                f"{fmt(row['wrong_mae'], 3)} [{fmt(wrong_ci[0], 3)}, {fmt(wrong_ci[1], 3)}]",
                f"{fmt(row['wrong_signed_error'], 3, signed=True)} [{fmt(wrong_signed_ci[0], 3)}, {fmt(wrong_signed_ci[1], 3)}]",
                f"{row['under']} / {row['correct']} / {row['over']} / {row['invalid']}",
                f"{100 * row['absolute_histogram'].get(1, 0) / row['wrong']:.1f}% / {100 * sum(value for deviation, value in row['absolute_histogram'].items() if deviation >= 2) / row['wrong']:.1f}%",
            ]
        )
    count_rows: list[list[str]] = []
    for count in range(1, 11):
        count_rows.append(
            [
                str(count),
                "NA" if count not in per_count["Qwen3-8B"] else fmt(per_count["Qwen3-8B"][count], 3),
                "NA" if count not in per_count["Gemma4-E4B"] else fmt(per_count["Gemma4-E4B"][count], 3),
            ]
        )
    figure = _absolute_deviation_svg(summaries, per_count)
    return f"""
<div class="absolute-deviation-block" id="absolute-deviation-block">
<h3>4.1 错误答案偏离正确 count 多远？</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Accuracy 只把所有错误都记为 0，无法区分少算 1 个与少算 4–5 个。若 absolute deviation 随 gold count 增大，就支持累积遗漏、饱和或 geometry compression，而不是均匀的随机读出错误。</span><strong>定义与评估。</strong><span>对可解析输出定义 signed error <code>e=prediction−gold</code> 和 absolute deviation <code>|e|</code>。主表同时报告全样本 MAE 与只在错误样本中的 conditional MAE；invalid 输出必须单列，不能被任意赋予数值误差。区间按 20 个 discovery seeds 做 5,000 次 cluster bootstrap。</span><strong>样本。</strong><span>这里使用 base HTML 内嵌的逐样本 V4.4 non-thinking discovery labels：每模型 20 seeds×10 counts=200 条。相同标签在所有 layer viewer 中重复，因此构建器只读取最低 all-fit layer 并按 seed×gold 去重。</span></div>
{table(["model", "correct / 200", "all-sample MAE [95% CI]", "all-sample signed error [95% CI]", "wrong-only MAE [95% CI]", "wrong-only signed error [95% CI]", "under / correct / over / invalid", "|e|=1 / |e|≥2"], summary_rows)}
<figure>{figure}<figcaption><strong>图 6A · V4.4 error magnitude。</strong>左图横轴是错误输出的 absolute deviation，纵轴是相应错误 prompt 数；右图横轴是真实 count，纵轴是该 count 下只在错误样本中计算的平均 absolute deviation。紫色为 Qwen3-8B，绿色为 Gemma4-E4B。某个 count 没有错误时不画点；折线只连接相邻的已有条件均值，不是拟合曲线。</figcaption></figure>
{details_table("Wrong-only deviation by gold count", ["gold count", "Qwen3-8B", "Gemma4-E4B"], count_rows)}
<p><strong>如果不取 absolute value。</strong>Signed error 保留方向，但高估与低估会互相抵消，因此它回答“模型整体偏高还是偏低”，MAE 才回答“平均偏离多远”。Qwen 的全样本 mean signed error 为 −0.815 [−0.995, −0.620]，即200条样本合计少算163；其中89条低估、100条正确、11条高估、0条不可解析。在100个错误中，89.0%是低估，wrong-only signed error 为−1.630 [−1.888, −1.353]。Gemma 的全样本 mean signed error 为−1.320 [−1.525, −1.110]，合计少算264；其中125条低估、73条正确、2条高估、0条不可解析。在127个错误中，98.4%是低估，wrong-only signed error 为−2.079 [−2.303, −1.853]。因此负偏差并不是少量极端样本造成，而是错误方向高度一致。</p>
<p><strong>Absolute magnitude。</strong>Qwen 的 wrong-only MAE 为1.850，且52.0%的错误至少偏离2；Gemma 为2.110，67.7%的错误至少偏离2。两模型在N=8–10的 conditional deviation 接近2–3，说明高-count failure不是单纯的偶发 off-by-one。</p>
<div class="conclusion"><strong>本段结论</strong>V4.4 counting error 具有明显的负向、count-dependent absolute deviation：随着 gold count 增大，输出被压缩在更低的 count 范围。该结果把后续 noise 分析的目标从“解释 accuracy 波动”细化为“解释何时 counter state 沿负方向偏移以及偏移幅度为何增大”。</div>
</div>
"""


def build_answer_fit_sensitivity(answer_data: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    correlations: list[float] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        selected = sorted(
            (
                item
                for item in answer_data.values()
                if item["model"] == model
                and item["fit_cohort"] == "all"
                and (item.get("manifold_display") or item.get("probe_optimal"))
            ),
            key=lambda item: int(item["layer"]),
        )
        for all_fit in selected:
            layer = int(all_fit["layer"])
            correct_fit = answer_data[f"{model}|{layer}|correct_only"]
            support = [
                int(value) for value in correct_fit["fit_count_support"].values()
            ]
            missing = [
                str(count)
                for count, value in correct_fit["fit_count_support"].items()
                if int(value) == 0
            ]
            distance_corr = _centroid_distance_correlation(
                all_fit["rows"], correct_fit["rows"]
            )
            correlations.append(distance_corr)
            role = (
                "M · manifold display"
                if all_fit.get("manifold_display")
                else "P · probe optimal"
            )
            rows.append(
                [
                    model,
                    f"L{layer} · {role}",
                    f"{int(all_fit['fit_rows'])} → {int(correct_fit['fit_rows'])}",
                    f"{min(support)}–{max(support)}"
                    + (f"; missing N={','.join(missing)}" if missing else ""),
                    f"{fmt(all_fit['common_v41_variance_capture'][2], 3)} → {fmt(correct_fit['common_v41_variance_capture'][2], 3)}",
                    f"{fmt(all_fit['pca3_discovery_cv_r2'], 3)} → {fmt(correct_fit['pca3_discovery_cv_r2'], 3)}",
                    f"{fmt(all_fit['count_signal_capture_pc1_3'], 3)} → {fmt(correct_fit['count_signal_capture_pc1_3'], 3)}",
                    fmt(distance_corr, 3),
                ]
            )
    return f"""
<div class="fit-sensitivity-block">
<h3>5.1 all-fit 与 correct-only-fit：错误样本是否制造了 geometry？</h3>
<div class="study-preface"><strong>为什么做。</strong><span>主 PCA 若把答错样本也用于拟合，视觉上的 count trajectory 可能被错误模式或不同 accuracy composition 扭曲；因此需要用只含 clean-correct discovery rows 的 basis 做敏感性复核。</span><strong>定义与评估。</strong><span><code>all-fit</code> 与 <code>correct-only-fit</code> 都只在 V4.1 discovery 上拟合，然后投影完全相同的 V4.4 answer states。我们比较 full-space count decodability、前三PC捕获的方差/信号，以及十个 V4.4 count centroids 的45个两两距离相关；最后一项对 PCA 的旋转、镜像与轴交换不敏感。</span><strong>图表判定。</strong><span>高 centroid-distance correlation 表示两种 fit 得到相近的 count geometry；但 correct-only 在某些 count 无样本时属于类别截断的敏感性分析，不能自动替代平衡的 all-fit 主分析。</span></div>
<p>两种 basis 都在 V4.1 discovery 上拟合，再投影完全相同的 V4.4 answer-query states。<code>common capture</code> 使用共同的 V4.1 全样本方差分母；最后一列比较相同 V4.4 states 所形成的十个 count centroids 的 45 个两两距离，因此不受 PCA 旋转和轴正负号影响。</p>
{table(["model", "layer/use", "fit n all→correct", "correct per-count support", "common capture PC1–3", "PCA3 CV R²", "count-axis capture", "V4.4 centroid distance corr"], rows)}
<p>Gemma 两层的 centroid-distance correlation 为 0.994–1.000，PCA3 CV R² 只变化约 0.003–0.007；Qwen 为 0.956–0.980，correct-only basis 的 R² 与 count-axis capture 有更明显下降。关键原因是 Qwen correct-only fit 在 N=7、9、10 没有样本，而 Gemma 每个 count 至少仍有 1 个正确样本；所以 correct-only 不是平衡的“更干净主分析”，而是有类别截断的敏感性分析。</p>
<div class="conclusion"><strong>本段结论</strong>四个主层的 V4.4 centroid-distance correlation 均≥{min(correlations):.3f}，因此有序 answer geometry 不是由错误样本凭空制造；但 Qwen 的 correct-only basis 因高 count 缺类而较不稳定。主文继续使用 all-fit，correct-only 只用于确认结论方向。</div>
</div>
"""


def build_answer_geometry_preface() -> str:
    return """
<div class="study-preface"><strong>为什么做。</strong><span>prompt-side running index 只有在最终 query 被整理成可供 LM head 使用的 answer state，才能解释模型为何输出具体数字。本节先问 answer residual 是否按 gold count 有序，再用 patching 检验该 state 是否可执行。</span><strong>定义与评估。</strong><span>在第一个答案 token 生成前保存 prompt-final <code>Total:</code> query 的 post-block residual <code>h<sup>A</sup><sub>s,N,ℓ</sub></code>；count step 与 PCA basis 均只在独立 discovery rows 上拟合。正式几何证据使用 full-space cross-validated count regression，correct-only fit 作为 outcome-conditioned sensitivity。</span><strong>可视化。</strong><span>交互 3D 图的 PC1/PC2/PC3 是冻结 basis 的前三轴，颜色表示 gold count 1–10；点间视觉距离不是显著性检验。共同坐标图只用于显示 prompt 与 answer roles 的旋转/缩放，不能因两条轨迹不平行而否定传递。</span></div>
"""


def build_causal_design(
    ov: dict[str, Any],
    upstream: dict[str, Any],
    seed_confirmation: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
) -> str:
    gemma_rows: list[list[str]] = []
    for label, document in [
        ("L37 H1/H2", gemma_l37_ov),
        *((ov_candidate_label(doc), doc) for doc in gemma_singles.values()),
    ]:
        cfg = document["config"]
        gemma_rows.append(
            [
                "Gemma natural OV",
                f"{label}: natural carrier、true pre-O injection、centered removal、mediation",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"direction {seed_span(cfg['direction_discovery_seeds'])}; center {seed_span(cfg['center_seeds'])}",
                f"candidate+matched IUT；global p={fmt_p(document['primary_decision']['global_intersection_union_p'])}",
            ]
        )
    if gemma_cross_layer is not None:
        cfg = gemma_cross_layer["config"]
        gemma_rows.append(
            [
                "Gemma cross-layer K2",
                "joint natural OV + L29 donor patch + exact L35 block",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"{len(cfg['mediation_pairs'])} directed pairs × candidate/3 controls",
                f"OV p={fmt_p(gemma_cross_layer['primary_decision']['global_intersection_union_p'])}; relay p={fmt_p(gemma_cross_layer['relay_decision']['intersection_union_p'])}",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        cfg = gemma_residual["config"]
        endpoint_count = len(gemma_residual["primary_decision"]["families"])
        gemma_rows.append(
            [
                f"Gemma {residual_name.upper()} residual relay",
                (
                    "clean bank ablation + "
                    if cfg.get("require_clean_necessity", False)
                    else ""
                )
                + "source patch + exact/count-axis blocks + L41 adoption",
                f"{seed_span(cfg['confirmation_seeds'])} confirmation",
                f"{len(cfg['donor_pairs'])} pairs × 5 conditions × candidate/3 controls",
                f"{2 * endpoint_count}-component IUT；global p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}",
            ]
        )
    for label, document in gemma_read_writes.items():
        cfg = document["config"]
        gemma_rows.append(
            [
                "Gemma read/write derivative",
                f"{label}: sliding-window-aware crossed α/V + downstream trace",
                seed_span(cfg["evaluation_seeds"]),
                f"{len(cfg['donor_pairs'])} directed pairs",
                "复用 parent seeds；机制分解，不是独立 replication",
            ]
        )
    seed_rows = [
        [
            "Macro V4.4",
            "ranked-bank ablation",
            "10 confirmation seeds 1254–1263",
            "N=7–10；每模型 40 prompts；ranked vs layer-matched random",
            "先在每 seed 内求差，再跨 seed 推断",
        ],
        [
            "Macro V4.4",
            "needle-end / answer-query patch；steering",
            "10 confirmation seeds 1254–1263",
            "paired nested prompts；层/剂量为重复条件",
            "seed-cluster mean；bootstrap CI；family correction",
        ],
        [
            "causal-v2",
            "baseline + clean-correct patch/ablation supplement",
            "20 discovery + 10 confirmation",
            "N=0–10：220/110 examples 每模型",
            "计划在 discovery 冻结；confirmation 独立评估",
        ],
        [
            "correct-only frozen confirmation",
            "broad-retrieval top-k ablation vs 3 layer-matched random sets",
            "20 fresh seeds 1296–1315",
            "N=1–5；100 examples/model；Qwen K=2/4、Gemma K=1/2 事先冻结",
            f"seed-cluster bootstrap 95% CI；audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "Natural OV",
            "carrier、pre-O injection、centered removal、mediation",
            "20 direction + 10 center/control + 20 confirmation",
            "1234–1253；1264–1273；1274–1293",
            f"四证据族 IUT；global p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}",
        ],
        *gemma_rows,
        [
            "Read/write",
            "crossed α/V + downstream write",
            "20 evaluation seeds 1274–1293",
            "六个 directed donor pairs",
            "复用 parent seeds；机制扩展，不算独立 replication",
        ],
        [
            "Relay screen",
            "carrier→edge patch→behavior→OV→removal",
            "20 confirmation seeds 1274–1293",
            "冻结 tail-64 position set",
            "五门 conjunction；任一失败即不支持",
        ],
        [
            "Upstream confirmation",
            "early donor patch + L28 exact block + LOO",
            "20 fresh seeds 1294–1313",
            "六个 directed donor pairs；120 primary rows",
            f"early 与 mediation conjunction；IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}",
        ],
    ]
    method_rows = [
        [
            "Ablation",
            "把 ranked head set 的输出置零，并与同层、同数量随机 head set 比较",
            "这组 heads 对维持生成 count 是否有特异贡献",
        ],
        [
            "State patching",
            "把 donor 的 hidden state / z state 拷到 receiver",
            "某个位置或通道是否足以运输 donor count 信息",
        ],
        [
            "Directional injection",
            "在真实 pre-O z slice 加入 ±β natural count step",
            "该 OV channel 是否具备有符号充分性与 dose response",
        ],
        [
            "Centered removal",
            "只删除自然 count component；与 same-span equal-norm orthogonal removal 比较",
            "模型自然运行是否特异依赖该 component",
        ],
        [
            "Serial mediation",
            "先做 upstream donor patch，再在 L28 阻断自然通道；与正交阻断比较",
            "upstream effect 是否确实经过冻结的 L28 channel",
        ],
        [
            "Crossed α/V",
            "分别替换 routing α 与 value content V，构造 RR/RD/DR/DD",
            "读取来自“看哪里”、还是“读到什么”，或两者都有",
        ],
    ]
    return f"""
<h3>7.1 因果实验总设计：如何把不同方法放在同一证据链中</h3>
<p><strong>先说聚合原则：</strong>这些实验没有被平均成一个“总机制分数”。每种实验检验不同 estimand、量纲也不同；我们先在同一个 seed 内对干预与匹配对照求 paired difference，再把 <strong>seed 作为独立统计单位</strong>。结论通过多种方法的方向一致与 conjunction/IUT 收敛，而不是把 patch accuracy、log-odds、count shift 和几何距离直接相加。</p>
{table(["方法", "具体做什么", "回答的问题"], method_rows)}
{table(["campaign", "主要方法", "独立 seeds", "行/条件规模", "如何折算与判定"], seed_rows)}
<div class="plain-protocol">
<h4>统一统计流程</h4>
<ol>
  <li><strong>行级计算：</strong>对每个 receiver 或 donor→receiver pair，先计算 intervention−control；有方向的 donor test 统一转成“正值=向 donor count 移动”。</li>
  <li><strong>seed 内折算：</strong>同一 seed 的 counts、donor pairs、layers 或 doses 先取平均，避免把同一 haystack 的多行当成独立样本。</li>
  <li><strong>跨 seed 推断：</strong>以 seed means 做 cluster bootstrap 95% CI；exact two-sided sign-flip 检验 seed-level paired effect 是否以 0 为中心。</li>
  <li><strong>多重比较：</strong>同一 family 的 layers、K 或 LOO heads 使用 Holm；natural OV 与 serial path 使用 IUT，global p 取各必要门中最大的 p，只有所有门都过线才通过。</li>
  <li><strong>显著性口径：</strong>本报告统一以校正后或预注册 exact <em>p</em>&lt;0.05 且方向符合假说为“显著”。p 值不是 effect size；效应大小和 95% CI 必须同时报告。</li>
</ol>
</div>
<div class="callout warning"><strong>独立性边界。</strong>Qwen natural-OV confirmation（1274–1293）与 upstream confirmation（1294–1313）不重叠；read/write 与 relay 复用 parent seeds，因此是机制分解而非第二次独立复制。Gemma 每个 evidence-ladder 分支都把 discovery/center 与 confirmation seeds 分开，但后备分支是在前一分支失败后才启动，整个搜索树没有全局 family-wise 校正；Gemma read/write 同样复用各自 parent seeds。correct-only frozen ablation 使用 1296–1315，与 Qwen upstream 的 1296–1313 有重叠，所以不能把这些 p 当作完全独立研究再相乘或 meta-combine。</div>
<div class="conclusion"><strong>本段结论</strong>后续因果证据按“功能定位 → 自然 OV 充分/必要 → α/V 读取分解 → 上游串行 mediation”逐级加严；不同实验互相约束，但不被压成一个不可解释的 pooled number。</div>
"""


def extract_js_json(document: str, name: str) -> dict[str, Any]:
    match = re.search(rf"const {re.escape(name)}=(.*?);\n", document)
    if not match:
        raise RuntimeError(f"Could not find embedded JavaScript object: {name}")
    return json.loads(match.group(1))


def quantile(values: Iterable[float], probability: float) -> float:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return 0.0
    position = (len(finite) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] + weight * (finite[upper] - finite[lower])


def _mix_rgb(
    left: tuple[int, int, int], right: tuple[int, int, int], weight: float
) -> str:
    weight = min(max(float(weight), 0.0), 1.0)
    channels = [
        round(start + (end - start) * weight)
        for start, end in zip(left, right, strict=True)
    ]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def cue_attention_color(value: float | None, cap: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "#d8d3ca"
    if cap <= 0:
        return "#e5e0d7"
    scaled = min(max(float(value) / cap, 0.0), 1.0)
    if scaled < 0.55:
        return _mix_rgb((247, 243, 234), (88, 139, 210), scaled / 0.55)
    return _mix_rgb((88, 139, 210), (35, 22, 92), (scaled - 0.55) / 0.45)


def cue_attention_svg(atlas: dict[str, Any], model: str) -> str:
    """Render the V4.4.2 non-thinking cue comparison as a literal head-by-layer table."""
    mode = atlas["models"][model]["modes"]["nonthinking"]
    layers = [int(layer) for layer in mode["layers"]]
    heads = int(mode["heads"])
    present = mode["conditions"]["cue_present"]["layer_head_score"]
    absent = mode["conditions"]["cue_absent"]["layer_head_score"]
    all_values = [
        float(value)
        for matrix in (present, absent)
        for row in matrix
        for value in row
        if value is not None and math.isfinite(float(value))
    ]
    positive_values = [value for value in all_values if value > 0]
    cap = quantile(positive_values, 0.995) if positive_values else 0.0

    cell_width = 12.0
    cell_height = 12.0 if heads >= 16 else 24.0
    plot_width = len(layers) * cell_width
    plot_height = heads * cell_height
    left_margin = 48.0
    right_margin = 12.0
    top_margin = 58.0
    bottom_margin = 88.0
    panel_gap = 64.0
    panel_width = left_margin + plot_width + right_margin
    total_width = 2 * panel_width + panel_gap
    total_height = top_margin + plot_height + bottom_margin
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", model).strip("-")

    parts = [
        f'<svg class="cue-attention-svg" role="img" aria-labelledby="{safe_id}-cue-title {safe_id}-cue-desc" '
        f'viewBox="0 0 {total_width:.1f} {total_height:.1f}" style="display:block;width:100%;height:auto">',
        f'<title id="{safe_id}-cue-title">{html.escape(model)} non-thinking broad-retrieval attention with and without the opening cue</title>',
        f'<desc id="{safe_id}-cue-desc">Two head-by-layer heat maps share one raw broad-retrieval score scale capped at the pooled 99.5th percentile. Layer is horizontal and head is vertical.</desc>',
        "<defs>",
        f'<linearGradient id="{safe_id}-cue-gradient" x1="0" x2="1" y1="0" y2="0">',
        '<stop offset="0%" stop-color="#f7f3ea"/><stop offset="55%" stop-color="#588bd2"/><stop offset="100%" stop-color="#23165c"/>',
        "</linearGradient></defs>",
    ]
    for panel_index, (condition, label, matrix) in enumerate(
        (
            ("cue_present", "有开头提示 · cue-present", present),
            ("cue_absent", "无开头提示 · cue-absent", absent),
        )
    ):
        origin_x = panel_index * (panel_width + panel_gap)
        plot_x = origin_x + left_margin
        parts.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="25" text-anchor="middle" '
            f'font-size="16" font-weight="700" fill="#172033">{html.escape(label)}</text>'
        )
        for head in range(heads):
            for layer_index, layer in enumerate(layers):
                raw_value = matrix[layer_index][head]
                value = (
                    float(raw_value)
                    if raw_value is not None and math.isfinite(float(raw_value))
                    else None
                )
                x = plot_x + layer_index * cell_width
                y = top_margin + head * cell_height
                title_value = "N/A" if value is None else f"{value:.6g}"
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width + 0.2:.1f}" '
                    f'height="{cell_height + 0.2:.1f}" fill="{cue_attention_color(value, cap)}">'
                    f"<title>{html.escape(condition)} · L{layer} H{head} · S_broad={title_value}</title></rect>"
                )
        parts.append(
            f'<rect x="{plot_x:.1f}" y="{top_margin:.1f}" width="{plot_width:.1f}" '
            f'height="{plot_height:.1f}" fill="none" stroke="#8e887f" stroke-width="1"/>'
        )
        head_step = 4 if heads > 12 else 1
        for head in range(0, heads, head_step):
            y = top_margin + (head + 0.67) * cell_height
            parts.append(
                f'<text x="{plot_x - 6:.1f}" y="{y:.1f}" text-anchor="end" '
                f'font-size="9" font-family="Consolas,monospace" fill="#626b78">H{head}</text>'
            )
        layer_step = 5
        for layer_index, layer in enumerate(layers):
            if layer_index % layer_step != 0 and layer_index != len(layers) - 1:
                continue
            x = plot_x + (layer_index + 0.5) * cell_width
            y = top_margin + plot_height + 12
            parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(55 {x:.1f} {y:.1f})" '
                f'font-size="9" font-family="Consolas,monospace" fill="#626b78">L{layer}</text>'
            )
        parts.append(
            f'<text x="{plot_x + plot_width / 2:.1f}" y="{top_margin + plot_height + 52:.1f}" '
            f'text-anchor="middle" font-size="11" fill="#626b78">layer</text>'
        )
        parts.append(
            f'<text x="{origin_x + 10:.1f}" y="{top_margin + plot_height / 2:.1f}" '
            f'transform="rotate(-90 {origin_x + 10:.1f} {top_margin + plot_height / 2:.1f})" '
            f'text-anchor="middle" font-size="11" fill="#626b78">head</text>'
        )
        if cap <= 0:
            parts.append(
                f'<text x="{plot_x + plot_width / 2:.1f}" y="{top_margin + plot_height / 2:.1f}" '
                f'text-anchor="middle" font-size="15" font-weight="700" fill="#6d665d">'
                "capture mask 内 direct raw-needle score 全为 0</text>"
            )

    legend_width = min(330.0, plot_width)
    legend_x = (total_width - legend_width) / 2
    legend_y = total_height - 22
    parts.extend(
        [
            f'<rect x="{legend_x:.1f}" y="{legend_y:.1f}" width="{legend_width:.1f}" height="9" fill="url(#{safe_id}-cue-gradient)"/>',
            f'<text x="{legend_x:.1f}" y="{legend_y - 4:.1f}" font-size="9" fill="#626b78">0</text>',
            f'<text x="{legend_x + legend_width:.1f}" y="{legend_y - 4:.1f}" text-anchor="end" font-size="9" fill="#626b78">p99.5 cap {cap:.6g}</text>',
            "</svg>",
        ]
    )
    return "".join(parts)


def cue_attention_summary_rows(atlas: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        mode = atlas["models"][model]["modes"]["nonthinking"]
        present = [
            float(value)
            for layer in mode["conditions"]["cue_present"]["layer_head_score"]
            for value in layer
        ]
        absent = [
            float(value)
            for layer in mode["conditions"]["cue_absent"]["layer_head_score"]
            for value in layer
        ]
        present_norm = math.sqrt(sum(value * value for value in present))
        absent_norm = math.sqrt(sum(value * value for value in absent))
        if present_norm == 0 or absent_norm == 0:
            rows.append(
                [
                    model,
                    "not defined (both maps are zero)",
                    "not defined",
                    "not defined",
                    f"{sum(present):.4f} → {sum(absent):.4f}",
                ]
            )
            continue
        cosine = sum(
            left * right for left, right in zip(present, absent, strict=True)
        ) / (present_norm * absent_norm)
        denominator = 0.5 * (
            sum(abs(value) for value in present) + sum(abs(value) for value in absent)
        )
        relative_l1 = (
            sum(abs(left - right) for left, right in zip(present, absent, strict=True))
            / denominator
        )
        top_k = min(10, len(present))
        present_top = {
            index
            for index, _ in sorted(
                enumerate(present), key=lambda item: item[1], reverse=True
            )[:top_k]
        }
        absent_top = {
            index
            for index, _ in sorted(
                enumerate(absent), key=lambda item: item[1], reverse=True
            )[:top_k]
        }
        rows.append(
            [
                model,
                fmt(cosine, 3),
                fmt(relative_l1, 3),
                f"{len(present_top & absent_top)}/{top_k}",
                f"{sum(present):.4f} → {sum(absent):.4f}",
            ]
        )
    return rows


def find_summary(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    stratum: str = "all",
    layer: int | None = None,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("stratum") == stratum
        and (layer is None or int(row.get("layer")) == int(layer))
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one summary row for {metric}/{stratum}/{layer}, got {len(matches)}"
        )
    return matches[0]


def sig_badge(
    p_value: float | None,
    *,
    label: str | None = None,
    alpha: float = 0.05,
) -> str:
    significant = (
        p_value is not None
        and math.isfinite(float(p_value))
        and float(p_value) <= float(alpha)
    )
    klass = "sig-yes" if significant else "sig-no"
    text = label if label is not None else ("显著" if significant else "不显著")
    return f'<span class="{klass}">{html.escape(text)}</span>'


def evidence_badge(
    supported: bool,
    positive: str = "确认",
    negative: str = "未确认",
) -> str:
    klass = "confirmed" if supported else "rejected"
    label = positive if supported else negative
    return f'<span class="evidence {klass}">{html.escape(label)}</span>'


def seed_span(values: Iterable[int]) -> str:
    seeds = [int(value) for value in values]
    if not seeds:
        return "none"
    if len(seeds) == 1:
        return str(seeds[0])
    return f"{min(seeds)}–{max(seeds)}（{len(seeds)} seeds）"


def append_to_section(section_html: str, appendix_html: str) -> str:
    marker = "</section>"
    index = section_html.rfind(marker)
    if index < 0:
        raise RuntimeError("Could not append to generated section")
    return (
        section_html[:index]
        + "\n"
        + appendix_html.strip()
        + "\n"
        + section_html[index:]
    )


def build_scope(
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_story: dict[str, Any],
) -> str:
    q_baseline = causal_v2["baseline"]["Qwen3-8B"]["confirmation"]
    g_baseline = causal_v2["baseline"]["Gemma4-E4B"]["confirmation"]
    g_supported = bool(gemma_story["support"])
    l37_supported = bool(
        gemma_l37_ov["primary_decision"]["full_natural_ov_transporter_support"]
    )
    claim_rows = [
        [
            "Prompt-side running index",
            "PCA / frozen-basis generalization; cue-present/absent shared-basis audit",
            "两模型均保留有序 occurrence geometry；提示改变 full-space state，但不创造序结构",
            '<span class="evidence descriptive">表征证据</span>',
        ],
        [
            "Distributed broad retrieval",
            "all-layer attention atlas + correct-only frozen top-k ablation",
            "Gemma K=1/K=2 的 clean-correct failure 与 ΔMAE 均过四比较 Holm；Qwen K=4 的 ΔMAE 过 Holm，但 clean-correct failure 仅 pointwise/CI 支持",
            '<span class="evidence functional">bank-level 功能支持</span>',
        ],
        [
            "Late answer count state",
            "answer-query donor patch + norm-matched steering",
            "完整 donor state 高概率运输 donor prediction；count direction 可定向操纵输出",
            '<span class="evidence functional">功能因果</span>',
        ],
        [
            "Qwen L28 natural OV transporter",
            "natural signal + true pre-O injection + centered removal + mediation IUT",
            f"H16/H19 四个证据族全部通过；global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}",
            '<span class="evidence confirmed">确认</span>',
        ],
        [
            "Qwen L28 mixed read/write",
            "crossed α/V decomposition + L28→L35 propagation",
            "routing 与 value/content 都贡献；写入沿冻结 count axes 存活到 L35",
            '<span class="evidence supported">机制扩展</span>',
        ],
        [
            "Qwen early slot-state → L28 → answer",
            "fresh-seed donor patch / exact block / orthogonal-control serial mediation",
            f"独立确认；IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}；H19 为 set 内非冗余成员",
            '<span class="evidence confirmed">独立确认</span>',
        ],
        [
            "Gemma L37 terminal natural OV",
            "与 Qwen 同构的四族 pre-O IUT",
            f"global IUT p={fmt_p(gemma_l37_ov['primary_decision']['global_intersection_union_p'])}",
            evidence_badge(l37_supported, "确认", "否定该候选"),
        ],
        [
            f"Gemma strongest completed path: {html.escape(str(gemma_story['label']))}",
            "冻结 evidence ladder；matched controls；fresh confirmation seeds",
            html.escape(str(gemma_story["summary"])),
            evidence_badge(g_supported, "机制支持", "未闭合"),
        ],
        [
            "Qwen tail-64 terminal relay",
            "registered carrier / edge patch / mediation / removal conjunction",
            f"不支持；global IUT p={fmt_p(relay['primary_decision']['global_intersection_union_p'])}",
            '<span class="evidence rejected">否定该候选</span>',
        ],
    ]
    return f"""
<section id="scope">
<h2>1 · 结论先行：当前最小可辩护机制</h2>
<p class="abstract"><strong>核心结论。</strong>在 non-thinking V4.4 中，模型并不是依赖一个严格单头、单位置的显式整数寄存器。两模型都在 prompt needle-end residual 中形成随 occurrence index 有序变化的分布式 state，并以 broad-retrieval head bank 汇集与计数相关的 slot states；late answer-query state 则携带可执行的 count prediction。prompt counter 与 answer counter 不需要在 residual space 中共线：attention 先在 head-space 形成 <em>z</em>，<em>W</em><sub>O</sub> 再把它写入新的 residual direction，后续 blocks 还可继续旋转/整合。Qwen3-8B 已将这一步闭合为“early broad set → L28 H16/H19 mixed α/V read → natural OV write → L29–L35 answer state”的受限因果链。Gemma 的最强可辩护结果是：{html.escape(str(gemma_story["summary"]))}</p>
<p>证据强度被严格分层：PCA 与 attention map 只定位可解码结构和候选路径；patching、steering 与 frozen top-k ablation 建立功能关系；真实 pre-O injection、centered z-space removal、same-span equal-norm control 和 fresh-seed serial mediation才用于自然机制主张。跨模型比较共享 estimand 与判定规则，不强迫两模型共享层号、head identity 或注意力可见窗口。</p>
{table(["机制命题", "直接检验", "当前结果", "证据等级"], claim_rows)}
<div class="baseline-strip">
  <div><span>Qwen confirmation</span><strong>{100 * q_baseline["accuracy"]:.1f}%</strong><small>accuracy · MAE {q_baseline["mean_absolute_error"]:.3f} · signed error {q_baseline["mean_signed_error"]:.3f}</small></div>
  <div><span>Gemma confirmation</span><strong>{100 * g_baseline["accuracy"]:.1f}%</strong><small>accuracy · MAE {g_baseline["mean_absolute_error"]:.3f} · signed error {g_baseline["mean_signed_error"]:.3f}</small></div>
  <div><span>Fine-grained scope</span><strong>Qwen L28 · {html.escape(str(gemma_story["label"]))}</strong><small>各模型只按实际通过的最强 evidence layer 表述</small></div>
</div>
<div class="conclusion"><strong>本节结论</strong>论文级主张应写成“分布式 prompt representation → broad retrieval → tested write/relay → late answer state”，而不是“某个 head 自己从原始 needle 数数”。Qwen 路径已经闭合；Gemma 的结论停在 <code>{html.escape(str(gemma_story["kind"]))}</code> 层级，不能自动升级为 Qwen 的逐头复制。</div>
</section>
"""


def build_methods(
    ov: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37_ov: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
) -> str:
    prompt_text = """You will need to count all city-score audit records in the passage below.\nA city-score audit record names one city and gives that city's numeric score.\n\n<passage>\n... approximately 10,000 tokens ...\n</passage>\n\nHow many city-score audit records are in the passage?\nDo not explain, reason aloud, quote, or list any records.\nWrite the count using ordinary decimal digits, with no space after the colon.\nYour entire response must be exactly one line:\nTotal:<integer>"""
    gemma_seed_rows: list[list[str]] = []
    for label, document in [
        ("L37 H1/H2 retained negative", gemma_l37_ov),
        *((ov_candidate_label(doc), doc) for doc in gemma_singles.values()),
    ]:
        candidate_cfg = document["config"]
        gemma_seed_rows.append(
            [
                "Gemma natural OV",
                f"Gemma4-E4B {label}",
                f"{seed_span(candidate_cfg['direction_discovery_seeds'])} direction; {seed_span(candidate_cfg['center_seeds'])} center/control; {seed_span(candidate_cfg['confirmation_seeds'])} confirmation",
                "N=1…10; causal counts 2/5/8",
                f"四证据族 IUT；matched sets；branch α={fmt(float(document['primary_decision']['alpha']), 3)}",
            ]
        )
    if gemma_cross_layer is not None:
        cross_cfg = gemma_cross_layer["config"]
        gemma_seed_rows.append(
            [
                "Gemma cross-layer fallback",
                ov_candidate_label(gemma_cross_layer),
                f"{seed_span(cross_cfg['direction_discovery_seeds'])} direction; {seed_span(cross_cfg['center_seeds'])} center; {seed_span(cross_cfg['confirmation_seeds'])} confirmation",
                "N=1…10; 3 directed pairs",
                "joint four-family natural OV + frozen L29→L35 relay; α=.025",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        residual_cfg = gemma_residual["config"]
        clean_text = (
            "; clean zero-z necessity"
            if residual_cfg.get("require_clean_necessity", False)
            else ""
        )
        gemma_seed_rows.append(
            [
                f"Gemma {residual_name.upper()} residual fallback",
                f"{residual_variant_label(gemma_residual)} → L{int(gemma_residual['selected_mediator_layer'])} → L41",
                f"{seed_span(residual_cfg['discovery_seeds'])} layer discovery; {seed_span(residual_cfg['confirmation_seeds'])} confirmation",
                f"N=1…10; 3 directed pairs; 5 path conditions{clean_text}",
                f"layer frozen before confirmation; {len(gemma_residual['primary_decision']['families'])} endpoint families × candidate/matched specificity; α=.025",
            ]
        )
    for label, document in gemma_read_writes.items():
        rw_cfg = document["config"]
        gemma_seed_rows.append(
            [
                "Gemma read/write extension",
                f"{label}: L{int(rw_cfg['mediator_layer'])} "
                + "/".join(f"H{int(head)}" for head in rw_cfg["heads"]),
                seed_span(rw_cfg["evaluation_seeds"]),
                f"{len(rw_cfg['donor_pairs'])} directed donor pairs",
                "复用 parent candidate seeds；derivative decomposition，不算独立确认",
            ]
        )
    seed_rows = [
        [
            "V4.4 representation",
            "Qwen3-8B; Gemma4-E4B",
            "1234–1253 discovery; 1254–1263 confirmation",
            "N=1…10",
            "V4.1 discovery 冻结 PCA/layer，再投影 V4.4",
        ],
        [
            "V4.4.2 cue robustness",
            "Qwen3-8B; Gemma4-E4B",
            "1234–1243",
            "N=1…10; cue present/absent",
            "两提示共享 PCA；seed 为 paired cluster",
        ],
        [
            "V4.4.4 natural OV",
            "Qwen3-8B L28 H16/H19",
            "1234–1253 direction; 1264–1273 center/control; 1274–1293 confirmation",
            "N=1…10; causal counts 2/5/8",
            "四证据族 IUT；matched head sets",
        ],
        *gemma_seed_rows,
        [
            "Read/write extension",
            "Qwen3-8B L28 H16/H19",
            "1264–1273 discovery; 1274–1293 evaluation",
            "six directed donor pairs",
            "复用 parent evaluation seeds；非独立复制",
        ],
        [
            "Upstream confirmation",
            "Qwen3-8B early top-4; L28 H16–H19",
            "1294–1313",
            "six directed donor pairs",
            "route/head set/endpoint/control 全部冻结",
        ],
        [
            "Correct-only low-count routes",
            "Qwen3-8B + Gemma4-E4B",
            "20 fresh seeds/model",
            "counts 1–3; six directed donor pairs",
            "仅纳入 donor/receiver clean 均正确；冻结 source/writer sets",
        ],
    ]
    return f"""
<section id="methods">
<h2>2 · 实验设定、符号与统计口径</h2>
<h3>2.1 V4.4 任务与 prompt</h3>
<p>每个 stimulus 是约 10,000-token 的 realistic haystack，内含十个可控 slot。对同一 seed，N 与 N+1 只在一个 slot 的 active/inactive 内容上变化；V4.4 同时跨 seed 随机化 slot 位置、city-score 内容及其顺序，随机 slot 最小间隔为 256 tokens。non-thinking 条件关闭模型原生 thinking flag，并在 assistant 侧预填 <code>Total:</code>，模型只生成十进制续写。主报告使用带开头定义提示的 frozen V4.4 prompt；V4.4.2 另作 cue-absent 表征敏感性分析。</p>
<pre class="prompt-block"><code>{html.escape(prompt_text)}</code></pre>
<div class="conclusion"><strong>本段结论</strong>V4.4 的 running-index geometry 若能跨 seed 保留，就不能只由固定绝对位置、固定 city identity 或固定内容顺序解释；但它仍可能依赖任务格式或分布式上下文。</div>

<h3>2.2 数据分割与推断单位</h3>
{table(["campaign", "模型/候选", "seeds", "counts / pairs", "冻结规则"], seed_rows)}
<p>所有主要置信区间都以 seed 为独立 cluster 做 bootstrap；符号检验使用 seed-level exact sign flip。自然 OV 的四个必要证据族采用 intersection–union test（IUT）：family p 是该族中最弱组成检验的 p，global p 是四个 family p 的最大值。LOO 的四个 head decrement 使用 Holm 校正。正确/错误分层只作 sensitivity analysis，任何 PCA/count axis 均先冻结，不在分层后重新选择。</p>
<div class="conclusion"><strong>本段结论</strong>确认性结论的独立单位是 seed 而非 token、head 或 donor pair；多重比较与 discovery/confirmation 分割必须在解释效应时一起保留。</div>

<h3>2.3 Representation 定义</h3>
<p>令 <code>h<sup>P</sup><sub>s,n,l</sub></code> 表示 seed <em>s</em> 中第 <em>n</em> 个 active needle 最后 token 经第 <em>l</em> 个 block 后的 residual；这就是 prompt running-index state。令 <code>h<sup>A</sup><sub>s,N,l</sub></code> 表示同一 prompt 最终 <code>Total:</code> query 的 residual；这是 answer count state。主 PCA 在 disjoint V4.1 discovery rows 上拟合后投影 V4.4；因此三维图只负责显示，full-space ridge、η²、CKA 与 causal tests 承担统计推断。</p>
<div class="equation">count-signal capture = ||P<sub>PC1:m</sub> b||² / ||b||², &nbsp; where b is the full-space OLS count direction</div>
<div class="equation">count η² = SS<sub>between count centroids</sub>/SS<sub>total</sub>; &nbsp;&nbsp; linear CKA(X,Y)=||X<sup>T</sup>Y||<sub>F</sub>²/(||X<sup>T</sup>X||<sub>F</sub>·||Y<sup>T</sup>Y||<sub>F</sub>).</div>
<p>这里 <code>X</code> 与 <code>Y</code> 是各自减去 grand centroid 后的 count-centroid matrices。η² 衡量完整 hidden space 中 count bucket 解释的变异比例；CKA 比较两条 centroid trajectory 的 Gram geometry，对共同旋转与各 PCA 轴正负号不敏感。二者都不由屏幕上的 PC1–3 距离直接计算。</p>
<p><strong>all-fit 与 correct-only-fit。</strong>all-fit 是主分析，因为它估计模型在真实运行分布中的 representation；correct-only-fit 只检查错误样本是否扭曲可视化。后者在高 count 可能没有任何正确样本，因此不能作为完整 count manifold 的无偏主 basis。报告中的 fit 切换只改变 basis，不改变被投影的 V4.4 states。</p>
<div class="conclusion"><strong>本段结论</strong>PCA 中出现有序轨迹只证明 count/index 可解码，不能单独证明该坐标被模型读取，更不能证明某个单点 state 是充分因果载体。</div>

<h3>2.4 Attention read、OV write 与 mediation 定义</h3>
<p><strong>Broad-retrieval atlas 的分数。</strong>令最终 <code>Total:</code> query 对第 <em>i</em> 个 needle 的 pooled attention 为 <code>m<sub>i</sub></code>：endpoint 视图只取该 needle 最后一个 token，full-span 视图则对该 needle 的全部 tokens 做 literal sum。定义总 needle mass <code>M=Σ<sub>i</sub>m<sub>i</sub></code>、occurrence profile <code>p<sub>i</sub>=m<sub>i</sub>/M</code>、entropy effective number <code>N<sub>eff,H</sub>=exp(−Σp<sub>i</sub>log p<sub>i</sub>)</code> 与 coverage <code>C<sub>H</sub>=N<sub>eff,H</sub>/N</code>；atlas 的 discovery primary score 为：</p>
<div class="equation">S<sub>broad</sub> = M × C<sub>H</sub>; &nbsp;&nbsp; atlas color = log<sub>10</sub>(S<sub>broad</sub>) within each model/pooling.</div>
<p>因此亮色同时要求“读到较多 needle mass”与“不要只压在一个 occurrence 上”，但仍不是 causal importance。phenotype breadth 另用 participation effective number <code>N<sub>eff,2</sub>=1/Σp<sub>i</sub><sup>2</sup></code>；global-broad 的冻结形状门为 mean <code>N<sub>eff,2</sub>≥6</code> 且任一 occurrence 的 mean normalized share≤0.25，并先要求 needle 对 matched hard negatives 的 enrichment&gt;1。只有 key window 覆盖全部 needles 的 heads 才能进入 global atlas；Gemma 的灰色 local-attention layers 表示该全局 estimand 不可定义，不表示 attention=0。</p>
<p>对 query head <em>h</em>，attention 的 pre-O state 与写回 residual 的输出分别为：</p>
<div class="equation">z<sub>h</sub>(q)=Σ<sub>j</sub> α<sub>h</sub>(q,j)V<sub>g(h)</sub>x<sub>j</sub>, &nbsp;&nbsp; o<sub>h</sub>(q)=W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q).</div>
<p>QK/α 决定读哪里，V 决定读出什么内容，W<sub>O</sub> 决定向 residual 写入什么方向。若 prompt residual 的单位 count direction 是 <code>u<sub>P</sub></code>，则 head set <em>S</em> 的写回为 <code>w<sub>S</sub>=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub></code>；到 answer layer 的局部传播可写成 <code>u<sub>A</sub>∝J<sub>ℓ→A</sub>w<sub>S</sub></code>。因此 count ordering 可以保留而 <code>u<sub>P</sub></code> 与 <code>u<sub>A</sub></code> 不共线；跨位置比较应检验可解码性、transport 与轴特异阻断，不应要求两个 PCA 方向视觉平行。</p>
<div class="equation">w<sub>S</sub>(c)=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q,c), &nbsp;&nbsp; u<sub>A</sub>∝J<sub>ℓ→A</sub>w<sub>S</sub>, &nbsp;&nbsp; generally u<sub>P</sub>∦u<sub>A</sub>.</div>
<p>对 head set <em>S</em>，自然一单位 count step 记为 <code>d<sub>S</sub></code>，其 set-output direction 为 <code>m<sub>S</sub>=W<sub>O</sub><sup>S</sup>d<sub>S</sub></code>。centered removal 从 <code>z<sub>S</sub>−z<sub>0,S</sub></code> 中移除沿 <code>m<sub>S</sub></code> 的自然成分；matched control 位于同一 <code>W<sub>O</sub><sup>S</sup></code> span、具有相同 post-O norm，并与 <code>m<sub>S</sub></code> 正交。</p>
<div class="equation">injection: z<sub>S</sub>←z<sub>S</sub>+βd<sub>S</sub>; &nbsp;&nbsp; u<sub>m</sub>=m<sub>S</sub>/||m<sub>S</sub>||; &nbsp;&nbsp; c<sub>S</sub>=⟨W<sub>O</sub><sup>S</sup>(z<sub>S</sub>−z<sub>0,S</sub>),u<sub>m</sub>⟩; &nbsp;&nbsp; removal: Δz<sub>S</sub>=−c<sub>S</sub>d<sub>S</sub>/||m<sub>S</sub>||.</div>
<p>由此 <code>W<sub>O</sub><sup>S</sup>Δz<sub>S</sub>=−c<sub>S</sub>u<sub>m</sub></code>，所以 removal 真正在 pre-O z-space 中完成，却精确删除 selected-head output span 内的自然 count component；没有把 answer axis 直接注入 residual。<code>z<sub>0,S</sub></code> 只用独立 center/control seeds 估计，避免把静态 offset 当成 count signal。</p>
<div class="equation">G = ([ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>intervention</sub> − [ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>clean</sub>), &nbsp;&nbsp; M = G<sub>orth</sub> − G<sub>natural-block</sub>.</div>
<p><code>G</code> 是 donor-vs-receiver candidate-sequence log-odds gain；<code>M</code> 是自然轴阻断相对 same-span orthogonal control 额外消除的 donor effect。成员分析定义 <code>D<sub>h</sub>=M<sub>full</sub>−M<sub>−h</sub></code>；正值表示移除 head <em>h</em> 后 set mediation 下降。</p>
<div class="equation">Δz<sub>value</sub>=½[(z<sub>RD</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>DR</sub>)], &nbsp; Δz<sub>route</sub>=½[(z<sub>DR</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>RD</sub>)].</div>
<p>其中第一个字母指定 receiver/donor attention routing，第二个字母指定 receiver/donor V content；<code>Δz<sub>full</sub>=Δz<sub>value</sub>+Δz<sub>route</sub></code>。这一分解不要求 QK heads 与 OV heads 是同一集合。</p>
<div class="conclusion"><strong>本节结论</strong>OV 作为 attention block 的写回变换在架构上必然存在；但只有“自然 carrier + true pre-O sufficiency + centered necessity + path mediation”同时成立，才支持模型自然使用某个指定 OV head set。Qwen 达到后者；Gemma 当前只定位到分布式 residual write。</div>
</section>
"""


def build_cue_section(cue_doc: str) -> str:
    nt = extract_js_json(cue_doc, "NT_GEOM")
    prompt = extract_js_json(cue_doc, "PROMPT_GEOM")
    atlas = extract_js_json(cue_doc, "ATLAS")
    rows: list[list[str]] = []
    for payload, site, label in (
        (prompt, "prompt_counter", "Prompt running index"),
        (nt, "answer_query", "Answer query"),
    ):
        for model, landmarks in payload["landmarks"].items():
            for role in ("display", "probe"):
                layer = int(landmarks[role])
                stat = payload["statistics"][f"{model}|{site}|{layer}"]
                rows.append(
                    [
                        model,
                        label,
                        f"L{layer} ({role})",
                        fmt(stat["centroid_cka"], 3),
                        f"{fmt(stat['count_eta_present'], 3)} → {fmt(stat['count_eta_absent'], 3)}",
                        fmt_p(stat["interaction_q"]),
                        fmt_p(stat["count_eta_q"]),
                    ]
                )
    attention_figures = "\n".join(
        f"""
<figure class="cue-attention-figure">
{cue_attention_svg(atlas, model)}
<figcaption><strong>Figure · {html.escape(model)} non-thinking broad-retrieval attention under cue removal.</strong> 横轴为 transformer layer，纵轴为 attention head；每个格子的颜色是最后一个 <code>Total:</code> query 对所有完整 active needle spans 的 <code>S<sub>broad</sub>=M×exp(H(p))/N</code>。左右图共享该模型 pooled raw score 的 p99.5 上限，超过上限的值只在显示时截断；因此同一模型内可以逐格比较有无开头提示，但颜色不能跨模型比较。鼠标悬停可读出 layer、head 与未截断分数。</figcaption>
</figure>
"""
        for model in ("Qwen3-8B", "Gemma4-E4B")
    )
    attention_rows = cue_attention_summary_rows(atlas)
    return f"""
<section id="cue-robustness">
<h2>4 · 开头提示的表征敏感性：拓扑保留不等于逐点不变</h2>
<h3>4.1 Hidden-state geometry</h3>
<p>V4.4.2 在相同 non-thinking flag 下，只删除开头两句 city-score 定义提示；每个 model × site × layer 使用 cue-present 与 cue-absent 的 pooled shared PCA。<code>centroid CKA</code> 比较 count 1–10 的两条 centroid geometry；<code>count×cue q</code> 在原始 full hidden space 检验 cue 是否以 count-dependent 方式改变状态；<code>Δ strength q</code> 检验 full-space count η² 是否改变。后二者均按 layer 做 BH-FDR。</p>
{table(["model", "counter", "layer", "centroid CKA", "count η² present → absent", "count×cue q", "Δ strength q"], rows)}
<p>表中 CKA 均为 0.981–0.997，说明 ordinal path 的整体 pairwise geometry 在删除提示后高度保留；与此同时，所有列出的 count×cue interaction 都显著，说明各 count 的向量并非只做同一个刚体平移。对 prompt counter，Qwen L29 和 Gemma L37/L39 的 count strength 也有显著但量级不同的改变；对 answer query，strength 差异未过 FDR。</p>
<div class="callout warning"><strong>如何解释“图形几乎不变”。</strong>高 CKA 回答的是“count 之间的相对几何是否相似”；interaction 回答的是“cue-induced delta 是否随 count 改变”。两者可以同时成立：提示可改变 gain、局部方向或 role offset，却不破坏 running-index 的排序拓扑。</div>

<h3>4.2 Attention map：同一 broad-retrieval score 的左右对照</h3>
<p>这里不再混用多种横轴或颜色定义。两模型都只显示 non-thinking 的 frozen broad-retrieval score；每个模型恰好两张大表，左边有提示、右边无提示。下表的 cosine、relative L1 与 top-10 overlap 只是 layer×head map 的描述性相似度，不是跨 seed 显著性检验；它们用于量化“亮区是否仍在同一批 heads”，不能替代后面的 frozen-set ablation。</p>
{attention_figures}
{table(["model", "map cosine", "relative L1 change", "top-10 overlap", "total S_broad present → absent"], attention_rows)}
<p>Qwen 的 map cosine 为 0.896，但 relative L1 change 为 0.422，top-10 仅重叠 5/10；也就是说，删除提示没有清空 broad-retrieval bank，却明显重新分配了 bank 内的读权重。Gemma 两图全零不是“所有 attention 都为零”，而是其 local/sliding attention architecture 使最后 answer query 在该 capture mask 下不能直接看见原始远端 needles；因此这个 direct raw-needle score 对 Gemma 是结构性不可用，不能据此否定经中间 residual/relay state 的读取。</p>
<div class="conclusion"><strong>本节结论</strong>开头提示不是 running-index geometry 的生成源；模型会从重复的 record 格式与累积上下文本身形成序结构。但提示仍调制 full-space representation，并在 Qwen 中重分配 broad-retrieval head map，因此不能声称 cue 完全没有机制影响。Gemma 的 direct raw-needle map 则受 attention window 限制；后续因果链路必须直接检验中间 state/relay，而不能把结构性零图误读为没有读取。V4.4.4 因果链路使用 cue-present 主设置，尚未完成逐环节的 cue-absent causal replication。</div>
</section>
"""


def build_causal_v2_intro(
    causal_v2: dict[str, Any], seed_confirmation: dict[str, Any]
) -> str:
    frozen_sets = {
        ("Qwen3-8B", "2"): "L27H18, L28H19",
        ("Qwen3-8B", "4"): "L27H18, L28H19, L23H29, L23H13",
        ("Gemma4-E4B", "1"): "L29H4",
        ("Gemma4-E4B", "2"): "L29H4, L35H2",
    }
    baseline_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for split in ("discovery", "confirmation"):
            row = causal_v2["baseline"][model][split]
            baseline_rows.append(
                [
                    model,
                    split,
                    str(row["examples"]),
                    f"{100 * row['accuracy']:.1f}%",
                    fmt(row["mean_absolute_error"], 3),
                    fmt(row["mean_signed_error"], 3),
                ]
            )
    patch_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for family in ("prompt_patching", "answer_patching"):
            item = causal_v2["correct_interventions"]["patch_pooled"][
                f"{model}::{family}"
            ]
            patch_rows.append(
                [
                    model,
                    "prompt full/multi-token"
                    if family == "prompt_patching"
                    else "answer query",
                    str(item["groups"]),
                    str(item["pair_instances"]),
                    f"{100 * item['pooled_average_patching_acc']:.1f}%",
                    f"{100 * item['group_min_average_patching_acc']:.1f}%–{100 * item['group_max_average_patching_acc']:.1f}%",
                ]
            )
    ablation_rows: list[list[str]] = []
    for key, item in causal_v2["correct_interventions"]["ablation_candidates"].items():
        ablation_rows.append(
            [
                item["model_label"],
                item["analysis_population"],
                f"top-{item['candidate_top_n']}",
                fmt(item["primary_effect"], 4),
                f"[{fmt(item['ci95_low'], 4)}, {fmt(item['ci95_high'], 4)}]",
                "unfrozen n=1…5 discovery",
            ]
        )
    comparison_order = [
        (model, k_text)
        for model in ("Qwen3-8B", "Gemma4-E4B")
        for k_text in sorted(
            seed_confirmation["models"][model], key=lambda value: int(value)
        )
    ]
    companion_ps = [
        float(seed_confirmation["models"][model][k_text]["absolute_error"]["exact_p"])
        for model, k_text in comparison_order
    ]
    companion_holm = dict(
        zip(comparison_order, holm_adjusted_pvalues(companion_ps), strict=True)
    )
    confirmation_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for k_text, metrics in sorted(
            seed_confirmation["models"][model].items(),
            key=lambda item: int(item[0]),
        ):
            all_shift = metrics["all_absolute_shift"]
            correct = metrics["clean_correct_to_wrong"]
            error = metrics["absolute_error"]
            all_pass = float(all_shift["ci95_low"]) > 0
            correct_pass = float(correct["ci95_low"]) > 0
            correct_family_pass = (
                float(correct["holm_p_across_four_frozen_sets"]) <= 0.05
            )
            status = (
                "clean-correct familywise 支持"
                if all_pass and correct_pass and correct_family_pass
                else (
                    "clean-correct pointwise；Holm 未过"
                    if all_pass and correct_pass
                    else "仅 all-example 主 CI；correct-only 未确认"
                )
            )
            confirmation_rows.append(
                [
                    model,
                    f"K={k_text}",
                    frozen_sets[(model, str(k_text))],
                    f"{fmt(all_shift['effect'], 4)} [{fmt(all_shift['ci95_low'], 4)}, {fmt(all_shift['ci95_high'], 4)}]",
                    f'{fmt(correct["effect"], 4)} [{fmt(correct["ci95_low"], 4)}, {fmt(correct["ci95_high"], 4)}]<br><span class="small">p/Holm={fmt_p(correct["two_sided_exact_seed_sign_flip_p"])}/{fmt_p(correct["holm_p_across_four_frozen_sets"])}</span>',
                    f"{fmt(error['effect'], 4)} [{fmt(error['ci95_low'], 4)}, {fmt(error['ci95_high'], 4)}]",
                    f"{fmt_p(error['exact_p'])} / {fmt_p(companion_holm[(model, str(k_text))])}",
                    status,
                ]
            )
    return f"""
<div class="callout evidence-note"><strong>更新后的 audit-grade V4.4 causal-v2。</strong>下方旧 V4.4 图保留原始 panel-restricted onset 与 steering 结果；本段补充重新跑完并通过 302/302 checks per model 的 causal-v2，以及 clean-correct supplement。两批实验的 estimand 不完全相同，数值不应直接合并成一个 meta-effect。</div>
<h3>7.2 Baseline 与 correct-only transport</h3>
{table(["model", "split", "examples", "accuracy", "MAE", "signed error"], baseline_rows)}
<p>causal-v2 在原 N=1…10 nested family 上增加 N=0，共 20 discovery seeds 与 10 confirmation seeds。两模型 confirmation 的 signed error 均为负，说明主要失败模式是 high-count undercount，而不是格式失败（valid rate=1）。</p>
{table(["model", "patch site", "groups", "eligible pair instances", "pooled donor-target adoption", "group range"], patch_rows)}
<div class="equation">donor-target adoption = mean I[patched receiver prediction = donor gold] &nbsp; | &nbsp; donor clean-correct ∧ receiver clean-correct.</div>
<p>clean-correct donor/receiver 条件下，answer-query patch 的 pooled donor-target adoption 在 Qwen/Gemma 分别为 96.6%/96.0%；prompt-side full/multi-token patch 为 81.5%/91.9%。这与“单个 needle endpoint patch 接近零”并不矛盾：前者协调搬运已筛选的 full-span/multi-token state，后者只搬一个 endpoint。</p>
{details_table("Ablation candidate effects（探索性 n 扫描）", ["model", "population", "candidate", "effect", "95% CI", "status"], ablation_rows)}
<p>ablation supplement 在 fresh seeds 上找到正向功能信号，但 top-n 是 n=1…5 同 seed 扫描后选出的候选；因此它支持“ranked attention bank 有可重复功能贡献”，不支持把某个 bank/top-n 写成冻结的独立确认。对大量单独 selected conditions 做 Holm 后没有条件 p≤.05，论文正文应报告 pooled/family-level 结论与这一 multiplicity 边界。</p>
<h3>7.3 冻结 top-k 的独立 seed 外推</h3>
{table(["model", "frozen set size", "frozen heads", "all-example |count shift| [95% CI]", "clean-correct c→w [95% CI]; exact/Holm", "companion ΔMAE [95% CI]", "exact p / Holm p (ΔMAE)", "primary interpretation"], confirmation_rows)}
<figure>{ablation_topk_svg(seed_confirmation)}<figcaption><strong>Figure · Frozen top-k ablation on fresh seeds.</strong> 左图横轴为事先冻结的 head-set size K，纵轴为 ranked−random 的绝对 generated-count shift；右图横轴同样为 K，纵轴为 clean-correct correct-to-wrong rate excess。圆点是 20 个 seed-cluster 的 pooled effect，竖线是 10,000 次 seed-cluster bootstrap 95% CI；CI 跨过 0 表示该主 estimand 未确认。两模型的 K 网格不同，连线只帮助看同模型剂量变化，不假设连续函数，也不用于跨模型比较绝对效应。</figcaption></figure>
<div class="equation">D<sub>abs</sub> = |ŷ<sub>ranked</sub>−ŷ<sub>clean</sub>| − mean<sub>r=1..3</sub>|ŷ<sub>random,r</sub>−ŷ<sub>clean</sub>|; &nbsp;&nbsp; D<sub>cw</sub> = I[ranked wrong] − mean<sub>r</sub>I[random<sub>r</sub> wrong] &nbsp; | &nbsp; clean correct.</div>
<p>这一轮在查看结果前冻结 Qwen K=2/4、Gemma K=1/2；具体前缀为 Qwen K2={frozen_sets[("Qwen3-8B", "2")]}、K4={frozen_sets[("Qwen3-8B", "4")]}，Gemma K1={frozen_sets[("Gemma4-E4B", "1")]}、K2={frozen_sets[("Gemma4-E4B", "2")]}。实验使用 20 个全新 seeds（1296–1315）、count 1–5、每模型 100 个 examples，并对每个 ranked set 配置 3 个 layer-matched random replicates。主 all-example estimand 是 ranked−random 的绝对 generated-count shift；主 clean-correct estimand 是原本答对样本中 ranked−random 的 correct-to-wrong rate。两套注册主分析都用 seed-cluster bootstrap 95% CI 判定；为让“显著性”完全可审计，报告另外对 clean-correct 的 20 个 seed-cluster contributions 做双侧 exact sign flip，并将四个 model×K 作为一个 Holm family。它是 multiplicity sensitivity analysis，不悄悄替换注册的 pointwise bootstrap 判据。</p>
<p>原 causal-v2 helper 在 n&gt;16 时实际切换为 100,000 次 Monte-Carlo sign flip，却保留了 exact 字段名；本报告从保存的 20 个 seed effects 重新枚举全部 2<sup>20</sup>=1,048,576 个符号组合，重算 clean-correct 与 companion ΔMAE 的真正双侧 exact p。Qwen K=2 的 clean-correct CI 跨 0（exact/Holm p=0.5/0.5），不能写成稳定 necessity；Qwen K=4 的 clean-correct excess 为 0.0650 [0.0238, 0.1124]，pointwise exact p=0.03125，但跨四比较 Holm p=0.0625，因此是“点估计与注册 CI 支持、family-wise sensitivity 未过”。它的 all-example ΔMAE=0.0500 [0.0200, 0.0833] 则 exact/Holm p=0.015625/0.03125。Gemma K=1/K=2 的 clean-correct failure excess 为 0.1231 [0.0595, 0.1857] 与 0.1282 [0.0690, 0.1882]，clean-correct exact/Holm p=0.0078125/0.0234375 与 0.0019531/0.0078125；相应 ΔMAE exact/Holm p=0.001595/0.004784 与 0.000944/0.003777。四个冻结比较应作为一个 family 阅读，而不是再从中选择最有利的 K。</p>
<div class="callout warning"><strong>不要混合两个 Qwen early set。</strong>本节 correct-only Qwen K4 是 L27H18/L28H19/L23H29/L23H13，用来确认 clean-run bank-level ablation；第 10.2 节 fresh-seed serial source 则冻结自更早 V4.4.2 路径筛选，为 L23H28/L23H29/L26H20/L27H18。二者共享部分成员但不是同一 set，不能把 clean necessity 与 serial mediation 逐头拼接。Gemma 第 10.4 节则有意直接复用本节 K2=L29H4/L35H2，以检验这一冻结 bank 是否接到 L37。</div>
<div class="conclusion"><strong>本段结论</strong>完整 count state 可由 prompt full/multi-token representation 与 late answer query 搬运。冻结 broad-retrieval ablation 提供 bank-level 功能必要性证据：Qwen K=4 的 all-example harm 通过四比较 Holm，clean-correct 仅达到注册 CI/pointwise exact、未过该附加 Holm sensitivity；Gemma K=1/K=2 的 clean-correct 与 ΔMAE 均通过 Holm。它仍定位到 ranked bank 而非唯一 head；更细的自然读写通路由后续 pre-O removal 与 serial mediation 决定。</div>
"""


def build_natural_ov_section(ov: dict[str, Any]) -> str:
    families = ov["primary_decision"]["families"]

    def component(family: str, endpoint: str) -> dict[str, Any]:
        hits = [
            item
            for item in families[family]["components"]
            if item["endpoint"] == endpoint
        ]
        if len(hits) != 1:
            raise RuntimeError(f"Missing OV component {family}/{endpoint}")
        return hits[0]

    natural = component("natural_signal", "natural_carrier_count_slope")
    injection = component("pre_o_injection", "injection_dose_slope")
    removal_error = component("centered_removal", "removal_error_axis_minus_control")
    removal_margin = component("centered_removal", "removal_margin_axis_minus_control")
    donor = component("path_mediation", "donor_patch_transport")
    mediation = component("path_mediation", "mediation_control_minus_axis_block")
    metric_rows = [
        [
            "1 · 自然信号",
            "clean forward 中测 H16/H19 centered z-output 对 count 的斜率",
            "斜率≤0",
            ci_text(natural),
            fmt_p(natural["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "2 · pre-O 充分性",
            "只在真实 pre-O z slice 加 β·d；让 heads 自身 W_O 写出",
            "dose slope≤0",
            ci_text(injection),
            fmt_p(injection["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "3a · centered 必要性",
            "删自然轴后增加的 |error|，减去 same-span 等范数正交删除",
            "额外误差≤0",
            ci_text(removal_error),
            fmt_p(removal_error["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "3b · centered 必要性",
            "同一 removal 对 correct-count margin 的额外影响",
            "margin 下降≥0",
            ci_text(removal_margin),
            fmt_p(removal_margin["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "4a · 路径前提",
            "把 donor z state patch 到 receiver，测 donor-count transport",
            "transport≤0",
            ci_text(donor),
            fmt_p(donor["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
        [
            "4b · 路径 mediation",
            "正交 block 保留的 donor effect − 自然轴 block 保留的 effect",
            "specificity≤0",
            ci_text(mediation),
            fmt_p(mediation["p"]),
            '<span class="sig-yes">显著，支持</span>',
        ],
    ]
    gate = evidence_gate_svg(
        [
            {
                "title": "Natural signal",
                "main": f"carrier slope {ci_text(natural)}",
                "sub": "candidate also exceeds matched-set mean",
                "p": f"family IUT p = {fmt_p(families['natural_signal']['intersection_union_p'])}",
            },
            {
                "title": "True pre-O sufficiency",
                "main": f"dose slope {ci_text(injection)}",
                "sub": "V-path z injection; no answer-axis injection",
                "p": f"family IUT p = {fmt_p(families['pre_o_injection']['intersection_union_p'])}",
            },
            {
                "title": "Centered necessity",
                "main": f"extra |error| {ci_text(removal_error)}",
                "sub": f"correct margin {ci_text(removal_margin)}",
                "p": f"family IUT p = {fmt_p(families['centered_removal']['intersection_union_p'])}",
            },
            {
                "title": "Path mediation",
                "main": f"donor transport {ci_text(donor)}",
                "sub": f"specific block {ci_text(mediation)}",
                "p": f"family IUT p = {fmt_p(families['path_mediation']['intersection_union_p'])}",
            },
        ]
    )
    nested_rows = [
        [
            str(row["k"]),
            ",".join(str(h) for h in row["heads"]),
            fmt_p(row["families"]["natural_signal"]["holm_p_across_k"]),
            fmt_p(row["families"]["pre_o_injection"]["holm_p_across_k"]),
            fmt_p(row["families"]["centered_removal"]["holm_p_across_k"]),
        ]
        for row in ov["nested_k"]
    ]
    return f"""
<section id="natural-ov">
<h2>8 · Qwen L28 natural OV transporter：从“可以推动”到“模型自然使用”</h2>
<h3>8.1 要验证的具体假说</h3>
<div class="test-card"><h4>假说：L28 H16/H19 是自然使用的 count-to-answer OV transporter</h4><dl>
<dt>候选从哪里来</dt><dd>在 discovery 数据上冻结 Qwen L28 H16/H19；confirmation 使用 seeds 1274–1293。</dd>
<dt>这里的 OV 是什么</dt><dd>head 先得到 pre-O state <code>z</code>，再由该 head 自己的 <code>W<sub>O</sub></code> 写回 residual。干预发生在 z-space，不把答案方向直接加到 residual。</dd>
<dt>不要求什么</dt><dd>不要求 H16/H19 自己从原始 needle 做 QK 定位；earlier heads 可以先构造或汇集 count-bearing source state。</dd>
<dt>通过标准</dt><dd>自然信号、pre-O 充分性、centered 必要性、path mediation 四门都必须显著；global IUT p 是四个 family 中最大的 p。</dd>
</dl></div>
<p>四个 K=2 matched control sets（H28/H31、H20/H23、H0/H3、H8/H11）在不查看 causal outcome 时，按 natural-step norm、answer-axis cosine、W<sub>O</sub>-span reachability 与 baseline output norm 匹配。这样，“删掉任意同范数方向都会伤模型”不能冒充自然 count channel 的特异作用。</p>
<div class="conclusion"><strong>本段结论</strong>本节验证的是“模型是否自然使用这个下游写入通道”，不是“同一组 heads 是否完成从原始 needle 定位到答案的全部工作”。</div>

<h3>8.2 四步实验分别做了什么</h3>
<figure>{gate}<figcaption><strong>Figure · Natural-OV evidence gates.</strong> 这是证据流程图而非坐标图，因此没有数值坐标轴。每个框给出一个预先规定的必要证据族、seed-cluster mean 与 95% bootstrap CI；family IUT p 取该族最弱组成检验。四框同时通过才判定 natural transporter；global IUT p={fmt_p(ov["primary_decision"]["global_intersection_union_p"])}。</figcaption></figure>
{table(["步骤", "具体操作", "零假设/失败边界", "effect [95% CI]", "exact p", "p<0.05?"], metric_rows)}
<div class="step-result"><strong>如何把六行折成一个确认结论。</strong>每个 family 内还要求 candidate 优于 matched-set mean；family IUT p 分别为 natural={fmt_p(families["natural_signal"]["intersection_union_p"])}、injection={fmt_p(families["pre_o_injection"]["intersection_union_p"])}、removal={fmt_p(families["centered_removal"]["intersection_union_p"])}、mediation={fmt_p(families["path_mediation"]["intersection_union_p"])}。global IUT 取最大值 {fmt_p(ov["primary_decision"]["global_intersection_union_p"])}，小于 0.05，故四门联合结论显著。</div>
<p><strong>逐步解释：</strong>步骤 1 排除“这个 span 完全没有自然 count signal”；步骤 2 排除“它只能在 post-O 被人工 steering”；步骤 3 排除“它只是一个可达但自然运行不需要的方向”；步骤 4 排除“它虽重要，却不介导 donor state transport”。mediation specificity 0.0136 相当于约 18.2% 的 donor-z transport，因此作用真实但只是部分路径。</p>
<div class="conclusion"><strong>本段结论</strong>H16/H19 不只是一个可操纵子空间：自然信号、真实 pre-O 充分性、matched-control 必要性和路径 mediation 同时成立（global IUT p={fmt_p(ov["primary_decision"]["global_intersection_union_p"])}）。这支持“自然使用的部分 OV transporter”，不支持“完整 count circuit 已全部找到”。</div>

<h3>8.3 Set size 与成员边界</h3>
{details_table("Nested-K secondary analysis", ["K", "heads", "natural Holm p", "injection Holm p", "removal Holm p"], nested_rows)}
<p>K=2/3/4/6/8 的 natural signal 与 injection 均通过 Holm；centered removal 只有 K=2 和 K=4 通过。扩大 K 同时扩大可干预 span，并没有“越多头越显著”的单调模式。这里显著性的统一阈值仍是 Holm p&lt;0.05。H16/H19 的 injection 近似可加，旧 factorial analysis 未确认超加性 synergy。</p>
<div class="conclusion"><strong>本节结论</strong>最稳健的 natural-OV 主张来自冻结的 K=2 H16/H19 matched-set test；更大 set 只作为二级稳健性结果，不能据此宣称更大的 head bank 更真实。</div>
</section>
"""


def ov_candidate_label(ov: dict[str, Any]) -> str:
    cfg = ov["config"]
    if cfg.get("candidate_sites"):
        return " + ".join(
            f"L{int(layer)}H{int(head)}" for layer, head in cfg["candidate_sites"]
        )
    return f"L{int(cfg['layer'])} " + "/".join(
        f"H{int(head)}" for head in cfg["candidate_heads"]
    )


def residual_variant_label(residual: dict[str, Any]) -> str:
    cfg = residual["config"]
    variant = str(cfg.get("mechanism_variant", "k2")).upper()
    return f"{variant} {{{ov_candidate_label(residual)}}}"


def ov_matched_set_labels(ov: dict[str, Any]) -> list[str]:
    cfg = ov["config"]
    if cfg.get("matched_control_sets"):
        return [
            " + ".join(f"L{int(layer)}H{int(head)}" for layer, head in site_set)
            for site_set in cfg["matched_control_sets"]
        ]
    labels = []
    for set_id, role in sorted(
        {(str(row["set_id"]), str(row["set_role"])) for row in ov["summary"]}
    ):
        if role != "matched_control":
            continue
        match = re.search(r"matched_control_L(\d+)_H(\d+(?:_H\d+)*)$", set_id)
        if match:
            layer = int(match.group(1))
            heads = "/".join(f"H{int(value)}" for value in match.group(2).split("_H"))
            labels.append(f"L{layer} {heads}")
        else:
            labels.append(set_id)
    return labels


def resolve_gemma_story(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    read_writes: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Choose the strongest *completed* Gemma mechanism without hiding failures.

    The order is part of the registered evidence ladder: a localized natural-OV
    result is stronger than a cross-layer set, which is stronger than a
    distributed residual-mediation result.  Read/write extensions are attached
    only to their exact parent candidate and never upgrade a failed parent.
    """
    supported_single = next(
        (
            (name, doc)
            for name, doc in singles.items()
            if bool(
                doc.get("primary_decision", {}).get(
                    "full_natural_ov_transporter_support", False
                )
            )
        ),
        None,
    )
    if supported_single is not None:
        name, doc = supported_single
        rw = read_writes.get(name)
        return {
            "kind": "single",
            "label": ov_candidate_label(doc),
            "natural": doc,
            "read_write": rw,
            "support": True,
            "global_p": float(doc["primary_decision"]["global_intersection_union_p"]),
            "alpha": float(doc["primary_decision"]["alpha"]),
            "summary": (
                f"{ov_candidate_label(doc)} 在独立 confirmation seeds 上通过四族 "
                "natural-OV IUT；其 α/V 分解只在对应 extension 完成后解释。"
            ),
        }
    if cross_layer is not None and bool(
        cross_layer.get("full_cross_layer_mechanism_support", False)
    ):
        return {
            "kind": "cross_layer",
            "label": ov_candidate_label(cross_layer),
            "natural": cross_layer,
            "read_write": None,
            "support": True,
            "global_p": max(
                float(cross_layer["primary_decision"]["global_intersection_union_p"]),
                float(cross_layer["relay_decision"]["intersection_union_p"]),
            ),
            "alpha": float(cross_layer["primary_decision"]["alpha"]),
            "summary": (
                "冻结的 L29H4+L35H2 跨层 set 同时通过 joint natural-OV "
                "与 L29→L35 relay 两道门。"
            ),
        }
    supported_residual = next(
        (
            (name, document)
            for name, document in residuals.items()
            if bool(
                document.get("primary_decision", {}).get(
                    "full_residual_count_path_support", False
                )
            )
        ),
        None,
    )
    if supported_residual is not None:
        residual_name, residual = supported_residual
        layer = int(residual["selected_mediator_layer"])
        clean_clause = (
            "、clean-run bank necessity"
            if residual["config"].get("require_clean_necessity", False)
            else ""
        )
        return {
            "kind": "residual",
            "residual_variant": residual_name,
            "label": (f"{residual_variant_label(residual)} → L{layer} residual → L41"),
            "natural": None,
            "read_write": None,
            "support": True,
            "global_p": float(
                residual["primary_decision"]["global_intersection_union_p"]
            ),
            "alpha": float(residual["primary_decision"]["alpha"]),
            "summary": (
                f"冻结 {residual_variant_label(residual)} 对 L{layer} count-aligned "
                f"residual 的写入{clean_clause}、精确阻断、count-axis 阻断与 "
                "L41 adoption 同时通过；该结论不定位唯一 downstream head。"
            ),
        }
    completed_local = [
        "L37 H1/H2",
        *(ov_candidate_label(doc) for doc in singles.values()),
    ]
    completed_fallbacks = []
    if cross_layer is not None:
        completed_fallbacks.append("cross-layer set")
    completed_fallbacks.extend(f"{name} residual path" for name in residuals)
    pending_fallbacks = [
        label
        for label, document in (
            ("cross-layer set", cross_layer),
            ("K2 residual path", residuals.get("k2")),
            ("K6 residual contingency", residuals.get("k6")),
        )
        if document is None
    ]
    completed_text = "、".join(completed_local + completed_fallbacks)
    pending_text = (
        "；尚未完成或未触发的后备分支为 " + "、".join(pending_fallbacks)
        if pending_fallbacks
        else ""
    )
    return {
        "kind": "partial",
        "label": "Gemma distributed counting evidence",
        "natural": None,
        "read_write": None,
        "support": False,
        "global_p": None,
        "alpha": 0.025,
        "summary": (
            f"已完成的冻结分支（{completed_text}）没有闭合完整机制 gate{pending_text}。"
            "当前只保留 independently confirmed ablation、prompt/answer patching 与各单项"
            "正效应，不把尚未完成的分支写成负结果，也不写成完整 head-level circuit。"
        ),
    }


def build_gemma_natural_ov_appendix(
    ov: dict[str, Any],
    *,
    heading: str = "8.4",
    context_label: str = "冻结 L37 假说",
) -> str:
    families = ov["primary_decision"]["families"]

    def component(family: str, endpoint: str) -> dict[str, Any]:
        hits = [
            item
            for item in families[family]["components"]
            if item["endpoint"] == endpoint
        ]
        if len(hits) != 1:
            raise RuntimeError(f"Missing Gemma OV component {family}/{endpoint}")
        return hits[0]

    natural = component("natural_signal", "natural_carrier_count_slope")
    injection = component("pre_o_injection", "injection_dose_slope")
    removal_error = component("centered_removal", "removal_error_axis_minus_control")
    removal_margin = component("centered_removal", "removal_margin_axis_minus_control")
    donor = component("path_mediation", "donor_patch_transport")
    mediation = component("path_mediation", "mediation_control_minus_axis_block")
    supported = bool(ov["primary_decision"]["full_natural_ov_transporter_support"])
    global_p = float(ov["primary_decision"]["global_intersection_union_p"])
    cfg = ov["config"]
    candidate_label = ov_candidate_label(ov)
    matched_labels = ov_matched_set_labels(ov)
    alpha = float(ov["primary_decision"]["alpha"])
    family_order = [
        "natural_signal",
        "pre_o_injection",
        "centered_removal",
        "path_mediation",
    ]
    failed = [name for name in family_order if not bool(families[name]["passes_alpha"])]
    gate_id = "gemma-gate-" + re.sub(r"[^a-z0-9]+", "-", candidate_label.lower()).strip(
        "-"
    )
    gate = evidence_gate_svg(
        [
            {
                "title": "Natural signal",
                "main": f"carrier slope {ci_text(natural)}",
                "sub": f"candidate must also exceed {len(matched_labels)} matched sets",
                "p": f"family IUT p = {fmt_p(families['natural_signal']['intersection_union_p'])}",
                "passed": families["natural_signal"]["passes_alpha"],
            },
            {
                "title": "True pre-O sufficiency",
                "main": f"dose slope {ci_text(injection)}",
                "sub": "real Gemma V projection + value normalization",
                "p": f"family IUT p = {fmt_p(families['pre_o_injection']['intersection_union_p'])}",
                "passed": families["pre_o_injection"]["passes_alpha"],
            },
            {
                "title": "Centered necessity",
                "main": f"extra |error| {ci_text(removal_error)}",
                "sub": f"correct margin {ci_text(removal_margin)}",
                "p": f"family IUT p = {fmt_p(families['centered_removal']['intersection_union_p'])}",
                "passed": families["centered_removal"]["passes_alpha"],
            },
            {
                "title": "Path mediation",
                "main": f"donor transport {ci_text(donor)}",
                "sub": f"specific block {ci_text(mediation)}",
                "p": f"family IUT p = {fmt_p(families['path_mediation']['intersection_union_p'])}",
                "passed": families["path_mediation"]["passes_alpha"],
            },
        ],
        id_prefix=gate_id,
    )
    metric_rows = [
        [
            "natural carrier",
            ci_text(natural),
            fmt_p(natural["p"]),
            sig_badge(natural["p"], alpha=alpha),
        ],
        [
            "pre-O dose response",
            ci_text(injection),
            fmt_p(injection["p"]),
            sig_badge(injection["p"], alpha=alpha),
        ],
        [
            "removal: extra |error|",
            ci_text(removal_error),
            fmt_p(removal_error["p"]),
            sig_badge(removal_error["p"], alpha=alpha),
        ],
        [
            "removal: correct-margin effect",
            ci_text(removal_margin),
            fmt_p(removal_margin["p"]),
            sig_badge(removal_margin["p"], alpha=alpha),
        ],
        [
            "donor-z transport",
            ci_text(donor),
            fmt_p(donor["p"]),
            sig_badge(donor["p"], alpha=alpha),
        ],
        [
            "mediation specificity",
            ci_text(mediation),
            fmt_p(mediation["p"]),
            sig_badge(mediation["p"], alpha=alpha),
        ],
    ]
    nested_rows = [
        [
            str(row["k"]),
            ",".join(str(head) for head in row["heads"]),
            fmt_p(row["families"]["natural_signal"]["holm_p_across_k"]),
            fmt_p(row["families"]["pre_o_injection"]["holm_p_across_k"]),
            fmt_p(row["families"]["centered_removal"]["holm_p_across_k"]),
        ]
        for row in ov.get("nested_k", [])
    ]
    conclusion = (
        f"Gemma {candidate_label} 通过四族联合标准，可称为本实验确认的自然 OV transporter；这使 terminal natural-write 结论获得跨模型支持。"
        if supported
        else f"Gemma {candidate_label} 没有通过四族联合标准；因此不能把局部正效应升级为完整 natural transporter。失败的必要证据族为："
        + "、".join(failed)
        + "。"
    )
    return f"""
<h3>{heading} Gemma natural-OV 检验：{candidate_label}（{html.escape(context_label)}）</h3>
<p>Gemma 使用与 Qwen 同一四族判定逻辑，但不复用 Qwen 的线性 value 近似：候选 set 为事先冻结的 {candidate_label}；direction seeds 为 {seed_span(cfg["direction_discovery_seeds"])}，center/control seeds 为 {seed_span(cfg["center_seeds"])}，confirmation seeds 为 {seed_span(cfg["confirmation_seeds"])}。{len(matched_labels)} 个未看本轮 causal outcome 即冻结的 matched sets 为 {html.escape("、".join(matched_labels))}；主候选必须分别通过方向检验并优于 matched-set mean，因此单个 endpoint 的 p&lt;{fmt(alpha, 3)} 仍不足以让整个 family 通过。</p>
<div class="conclusion"><strong>设计结论</strong>selection status 为 <code>{html.escape(str(cfg.get("selection_status", "frozen_before_confirmation")))}</code>。本轮在独立 confirmation seeds 上检验冻结 set；Qwen 与 Gemma 共享因果定义和判定门，不强迫两者共享层号或 head identity。</div>
<figure>{gate}<figcaption><strong>Figure · Gemma natural-OV evidence gates.</strong> 四个框依次对应自然载荷、真实 pre-O 充分性、centered z-space 必要性与 donor-path mediation；绿色勾表示整个 family（含 matched-set superiority）通过，粉色叉表示失败。框内 effect 是 seed mean [95% bootstrap CI]，family IUT p 是该门最弱必要检验；全局判定取四门中最大的 p={fmt_p(global_p)}。</figcaption></figure>
{table(["endpoint", "effect [95% CI]", "directional p", f"endpoint p<{fmt(alpha, 3)}?"], metric_rows)}
<p>联合判定不对六个 endpoint 做简单多数票，也不把不同单位的 effect 相加。每个 family 先对 candidate direction 与 matched controls 做 intersection–union test；再令 global IUT p 等于四个 family p 的最大值。最终 <code>full_natural_ov_transporter_support={str(supported).lower()}</code>，global IUT p={fmt_p(global_p)}。</p>
<div class="conclusion"><strong>结果结论</strong>{conclusion}</div>
{details_table("Gemma nested-K secondary analysis", ["K", "heads", "natural Holm p", "injection Holm p", "removal Holm p"], nested_rows) if nested_rows else ""}
<p class="small">Nested-K 只用于 set-size 稳健性分析；Holm 校正后结果不能反过来替换预先冻结的 K=2 主检验，也不能据此声称单个 head 是完整 counter。</p>
<div class="conclusion"><strong>本小节边界</strong>即使四族全部通过，允许的表述仍是“{candidate_label} 构成自然使用的部分 transporter”；它不证明该 set 直接从原始 needle 读取，也不排除其他并行写入通道。</div>
"""


def build_gemma_evidence_ladder(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
    story: dict[str, Any],
) -> str:
    rows: list[list[str]] = []

    def natural_row(label: str, doc: dict[str, Any], role: str) -> None:
        decision = doc["primary_decision"]
        supported = bool(decision["full_natural_ov_transporter_support"])
        alpha = float(decision["alpha"])
        rows.append(
            [
                label,
                role,
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}; α={fmt(alpha, 3)}",
                evidence_badge(supported, "通过", "保留负结果"),
            ]
        )

    natural_row("L37 H1/H2", l37, "最初冻结的 terminal-OV 假说")
    for name, label in (("l29h4", "L29H4"), ("l35h2", "L35H2")):
        if name in singles:
            natural_row(label, singles[name], "independent ablation-ranked single")
        else:
            rows.append(
                [
                    label,
                    "gated single-head fallback",
                    "前一单头已通过，按序贯规则未运行",
                    '<span class="evidence qualified">预设跳过</span>',
                ]
            )
    if cross_layer is not None:
        cross_p = max(
            float(cross_layer["primary_decision"]["global_intersection_union_p"]),
            float(cross_layer["relay_decision"]["intersection_union_p"]),
        )
        rows.append(
            [
                "L29H4 + L35H2",
                "cross-layer joint OV + L29→L35 relay",
                f"joint max-IUT p={fmt_p(cross_p)}; α={fmt(float(cross_layer['primary_decision']['alpha']), 3)}",
                evidence_badge(
                    bool(cross_layer["full_cross_layer_mechanism_support"]),
                    "通过",
                    "保留负结果",
                ),
            ]
        )
    else:
        rows.append(
            [
                "L29H4 + L35H2",
                "cross-layer fallback",
                "单头已通过，按序贯规则未运行",
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    for residual_name, residual in residuals.items():
        decision = residual["primary_decision"]
        layer = int(residual["selected_mediator_layer"])
        rows.append(
            [
                f"{residual_variant_label(residual)} → L{layer} residual → L41",
                f"{residual_name.upper()} distributed residual mediation",
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}; α={fmt(float(decision['alpha']), 3)}",
                evidence_badge(
                    bool(decision["full_residual_count_path_support"]),
                    "通过",
                    "保留负结果",
                ),
            ]
        )
    if "k2" not in residuals:
        rows.append(
            [
                "K2 bank → residual → L41",
                "registered residual fallback",
                "较强分支已通过，按序贯规则未运行",
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    if "k6" not in residuals:
        k2_passed = bool(
            residuals.get("k2", {})
            .get("primary_decision", {})
            .get("full_residual_count_path_support", False)
        )
        rows.append(
            [
                "K6 bank → residual → L41",
                "last registered exploratory contingency",
                (
                    "K2 residual conjunction 已通过，按冻结序贯规则停止；K6 未运行"
                    if k2_passed
                    else "只有 K2 residual 完整失败后才触发；当前未运行"
                ),
                '<span class="evidence qualified">预设跳过</span>',
            ]
        )
    return f"""
<h3>8.4 Gemma 证据阶梯：先冻结、后揭示、失败不删除</h3>
<p>Gemma 没有被要求复刻 Qwen 的 layer/head identity。顺序固定为：最初 L37 terminal set → independent-ablation 排名得到的 L29H4 → 条件式 L35H2 → 跨层 K2 → K2 residual mediation → K6 residual contingency。后一个分支只有在前一个完整 conjunction 失败时才启动；因此它是透明的 mechanism search，而不是在同一批 confirmation outcomes 上反复换定义。</p>
{table(["候选", "检验层级", "联合统计", "判定"], rows)}
<div class="callout warning"><strong>多重性边界。</strong>每个分支内部都要求 candidate core 与 matched-control superiority 同时成立，并在独立 confirmation seeds 上用 IUT；fallback 分支把阈值收紧到 α=0.025。整个跨分支搜索树没有再做一个全局 family-wise 校正，所以 Gemma 的最终机制应标为“顺序探索后独立 seed 确认”，不能写成一次性预注册的唯一候选验证。</div>
<div class="conclusion"><strong>本段结论</strong>{html.escape(str(story["summary"]))} 最强允许层级是 <code>{html.escape(str(story["kind"]))}</code>；所有更强但失败的定位仍在下文逐项展示。</div>
"""


def build_gemma_cross_layer_appendix(cross: dict[str, Any]) -> str:
    relay = cross["relay_decision"]
    components = relay["components"]
    forest_rows = [
        {
            "label": (
                "L29 donor gain"
                if item["endpoint"].startswith("l29_donor_gain")
                else "L35 exact-block mediation"
            )
            + (
                " · candidate−control"
                if item["role"] == "candidate_specificity"
                else " · candidate"
            ),
            "mean": item["mean"],
            "low": item["ci95_low"],
            "high": item["ci95_high"],
            "value": f"{ci_text(item)} · p={fmt_p(item['p'])}",
        }
        for item in components
    ]
    figure = forest_svg(
        forest_rows,
        title="Gemma L29 to L35 frozen cross-layer relay",
        description=(
            "Candidate and candidate-minus-matched-control effects are shown for "
            "the early L29 donor gain and the L35 exact-block mediation effect."
        ),
        x_label="normalized donor-count transport (positive supports the registered relay)",
    )
    alpha = float(cross["primary_decision"]["alpha"])
    supported = bool(cross["full_cross_layer_mechanism_support"])
    return f"""
<h3>10.3 Gemma 跨层 relay：L29 output 是否经 L35 传到 answer</h3>
<p><strong>操作定义。</strong>先把 donor 的 L29H4 answer-query pre-O <code>z</code> patch 到 receiver，测量 donor-count transport；随后在 L35H2 精确删除由该 L29 patch 诱发的自然 <code>z</code> 增量，并与同一 <code>W<sub>O</sub></code> span、相同 post-O norm、但与该增量正交的 block 比较。前者检验 L29 是否能推动 state，后者检验这部分推动是否确实经过 L35，而非仅仅在别处平行传播。</p>
<div class="equation">relay mediation = transport(L29 patch + orthogonal L35 block) − transport(L29 patch + exact induced-Δz<sub>L35</sub> block).</div>
<figure>{figure}<figcaption><strong>Figure · Gemma cross-layer relay.</strong> 横轴是归一化 donor-count transport；0 表示无 donor shift 或 exact block 不比正交 block 多消除 transport。点是 {len(cross["config"]["confirmation_seeds"])} 个 confirmation seeds 的均值，横线是 seed-cluster bootstrap 95% CI；每个 endpoint 分 candidate core 与 candidate−3 matched-set mean 两行。</figcaption></figure>
<p>relay IUT p={fmt_p(relay["intersection_union_p"])}，阈值 α={fmt(alpha, 3)}；joint natural-OV global IUT p={fmt_p(cross["primary_decision"]["global_intersection_union_p"])}。完整跨层机制要求两者都通过，因此 <code>full_cross_layer_mechanism_support={str(supported).lower()}</code>。</p>
<div class="conclusion"><strong>本段结论</strong>{"冻结 K2 同时满足 joint OV 与 L29→L35 relay，支持一条局部跨层 transporter。" if supported else "至少一个必要门失败；不能把 L29H4 与 L35H2 串成自然跨层 transporter，即使某个单项 effect 为正。"}</div>
"""


def build_gemma_residual_appendix(residual: dict[str, Any]) -> str:
    summary = residual["summary"]
    decision = residual["primary_decision"]
    cfg = residual["config"]
    selected = int(residual["selected_mediator_layer"])
    alpha = float(decision["alpha"])
    bank_label = residual_variant_label(residual)
    labels = {
        "clean_correct_failure_rate": "clean-correct failure-rate increase",
        "clean_delta_absolute_error": "clean expected-count absolute-error increase",
        "source_donor_transport": "source-bank donor transport",
        "exact_residual_mediation": "exact induced-Δ residual mediation",
        "count_axis_mediation": "frozen count-axis mediation",
        "terminal_count_adoption": "L41 terminal count adoption",
    }
    # The table retains the full estimand names; the figure uses compact labels
    # so its left margin remains readable at ordinary laptop viewport widths.
    plot_labels = {
        "source_donor_transport": "source transport",
        "exact_residual_mediation": "induced-Δ mediation",
        "count_axis_mediation": "count-axis mediation",
        "terminal_count_adoption": "L41 adoption",
    }
    clean_endpoints = (
        (
            (
                "clean_correct_failure_rate",
                "failure-rate increase under zero-z ablation",
            ),
            ("clean_delta_absolute_error", "Δ expected-count absolute error"),
        )
        if cfg.get("require_clean_necessity", False)
        else ()
    )
    path_endpoints = (
        "source_donor_transport",
        "exact_residual_mediation",
        "count_axis_mediation",
        "terminal_count_adoption",
    )
    forest_rows = []
    table_rows = []
    endpoint_rows: dict[str, list[dict[str, Any]]] = {}
    for endpoint in (*[item[0] for item in clean_endpoints], *path_endpoints):
        endpoint_rows[endpoint] = []
        for role in ("candidate_core", "candidate_specificity"):
            hits = [
                row
                for row in summary
                if row["endpoint"] == endpoint and row["set_role"] == role
            ]
            if len(hits) != 1:
                raise RuntimeError(f"Missing residual summary {endpoint}/{role}")
            row = hits[0]
            suffix = (
                "candidate" if role == "candidate_core" else "candidate−control mean"
            )
            label = f"{plot_labels.get(endpoint, labels[endpoint])} · {suffix}"
            p_value = float(row["one_sided_exact_sign_flip_p"])
            plotted = {
                "label": label,
                "mean": row["mean"],
                "low": row["ci95_low"],
                "high": row["ci95_high"],
                "value": f"{ci_text(row)} · p={fmt_p(p_value)}",
            }
            endpoint_rows[endpoint].append(plotted)
            if endpoint in path_endpoints:
                forest_rows.append(plotted)
            table_rows.append(
                [
                    labels[endpoint],
                    suffix,
                    ci_text(row),
                    fmt_p(p_value),
                    sig_badge(p_value, alpha=alpha),
                ]
            )
    figure = forest_svg(
        forest_rows,
        title=f"Gemma {bank_label} distributed residual mediation",
        description=(
            "Four registered endpoints are shown for the candidate source bank "
            "and its difference from the mean of three matched source banks."
        ),
        x_label="normalized donor-count effect (positive supports the registered path)",
        width=1260,
        left=390,
        right=320,
    )
    clean_figures = "".join(
        "<figure>"
        + forest_svg(
            endpoint_rows[endpoint],
            title=f"Gemma {bank_label} clean necessity: {labels[endpoint]}",
            description=(
                "Candidate source-bank damage and candidate-minus-matched-control "
                "damage on held-out clean runs."
            ),
            x_label=x_label,
            width=1260,
        )
        + (
            f"<figcaption><strong>Figure · {labels[endpoint]}.</strong> 横轴单位为"
            f" {html.escape(x_label)}；点是 confirmation-seed mean，横线是 95% "
            "seed-cluster bootstrap CI。candidate 与 candidate−3 matched-set mean "
            "两行都必须为正。</figcaption></figure>"
        )
        for endpoint, x_label in clean_endpoints
    )
    clean_block = (
        '<div class="test-card"><h4>K6 clean-run natural necessity</h4>'
        "<p>在每个正确 baseline prompt 的 answer query，把冻结 K6 bank 的 pre-O "
        "z slices 置零；与三组同层组成、同 set size 的 K6 controls 比较。第一个 "
        "estimand 只在 baseline clean-correct cases 中计算转错率，第二个在全部 counts "
        "中计算 expected-count absolute error 增量。该门防止把 donor-patch sufficiency "
        "误写成模型自然使用。</p></div>" + clean_figures
        if clean_endpoints
        else ""
    )
    discovery_rows = sorted(
        residual["discovery_layer_scores"], key=lambda row: int(row["layer"])
    )
    discovery_table = [
        [
            f"L{int(row['layer'])}",
            fmt(row["mean_aligned_induced_norm"], 4),
            fmt(row["positive_fraction"], 3),
            str(int(row["samples"])),
            '<span class="evidence confirmed">selected</span>'
            if int(row["layer"]) == selected
            else "",
        ]
        for row in discovery_rows
    ]
    supported = bool(decision["full_residual_count_path_support"])
    return f"""
<h3>10.4 Gemma 分布式 residual relay：不强迫唯一 downstream head</h3>
<p>当前定义把 independently frozen 的 source bank <code>{html.escape(bank_label)}</code> 当作一个整体。在 discovery seeds 上，只用 source patch 引发的 residual change，从 L36–L40 选择 mean count-aligned induced change 最大的一个边界；选定 L{selected} 后锁定该 layer、count axis、matched banks 与所有 endpoints，再对不重叠的 {seed_span(cfg["confirmation_seeds"])} confirmation seeds 评估。</p>
{table(["discovery layer", "mean aligned induced Δ", "positive fraction", "samples", "selection"], discovery_table)}
{clean_block}
<div class="test-card"><h4>五个 forward conditions 如何折成四个 endpoints</h4><dl>
<dt>source patch</dt><dd>把 donor 的 {html.escape(bank_label)} pre-O states 写到 receiver，测 donor transport。</dd>
<dt>exact block / exact orthogonal</dt><dd>在 L{selected} 删除这次 source patch 实际诱发的 residual Δ；对照删除等范数正交方向。二者差为 exact mediation。</dd>
<dt>count-axis block / count-axis orthogonal</dt><dd>删除 discovery-frozen natural count-axis 分量；对照删除等范数 axis-orthogonal 分量。二者差为 count-axis mediation。</dd>
<dt>L41 adoption</dt><dd>source patch 后 L41 residual 在 frozen count step 上是否朝 donor count 移动。</dd>
</dl></div>
<figure>{figure}<figcaption><strong>Figure · Gemma distributed residual-path gates.</strong> 横轴统一为正向 donor-count effect；0 是相应零假设。每个 endpoint 的第一行是冻结 source bank 本身，第二行是它减去 3 个 layer-matched banks 的 seed-wise mean；点为 {len(cfg["confirmation_seeds"])} 个 confirmation seed means，横线为 95% bootstrap CI。四个 path endpoints 的八行必须同时 CI&gt;0 且 exact sign-flip p≤{fmt(alpha, 3)}；若上方存在 clean-necessity 图，它们也属于同一全局 conjunction。</figcaption></figure>
{table(["endpoint", "contrast", "effect [95% CI]", "one-sided exact p", f"p≤{fmt(alpha, 3)}?"], table_rows)}
<p>global IUT p={fmt_p(decision["global_intersection_union_p"])}，取 {2 * len(decision["families"])} 个必要组成检验中最大的 p；<code>full_residual_count_path_support={str(supported).lower()}</code>。</p>
<div class="callout warning"><strong>解释边界。</strong>该实验可证明 frozen source bank 的 causal effect 经过一个 count-aligned residual channel 到达 L41；它不定位唯一 downstream attention head，也不排除 MLP 或其他 heads 在 L{selected} 前后共同实现 relay。它比 localized natural-OV transporter 是更弱、但仍可反驳的机制主张。</div>
<div class="conclusion"><strong>本段结论</strong>{"冻结 " + bank_label + "→L" + str(selected) + " residual→L41 的完整 conjunction 通过，支持一条分布式 counting relay。" if supported else "至少一个必要 endpoint 或 matched-specificity 门失败；该 residual 边界不能被写成完整 causal relay。"}</div>
"""


def build_read_write_section(read_write: dict[str, Any]) -> str:
    summary = read_write["summary"]
    read_metrics = [
        find_summary(summary, "read_value_behavior_transport"),
        find_summary(summary, "read_routing_behavior_transport"),
        find_summary(summary, "read_full_behavior_transport"),
        find_summary(summary, "read_value_minus_routing_transport"),
    ]
    read_labels = [
        "V/content component",
        "α/routing component",
        "full donor-z patch",
        "value − routing",
    ]
    read_rows = []
    for label, row in zip(read_labels, read_metrics):
        read_rows.append(
            {
                "label": label,
                "mean": row["mean"],
                "low": row["ci95_low"],
                "high": row["ci95_high"],
                "value": f"{ci_text(row)} · p={fmt_p(row['exact_sign_flip_p'])}",
            }
        )
    read_forest = forest_svg(
        read_rows,
        title="Factorized read contributions at Qwen L28 H16/H19",
        description="Horizontal axis is normalized donor behavioral transport. Value and routing components are both positive and indistinguishable in magnitude; the full patch is larger.",
        x_label="normalized donor behavioral transport (positive = donor count gains probability)",
    )
    write_rows = sorted(
        [
            row
            for row in summary
            if row.get("metric") == "write_residual_specificity"
            and row.get("stratum") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    write_svg = write_trace_svg(write_rows)
    mediation_rows = []
    for metric, label in (
        ("read_value_ov_mediation_specificity", "V/content component"),
        ("read_routing_ov_mediation_specificity", "α/routing component"),
    ):
        row = find_summary(summary, metric)
        mediation_rows.append(
            [
                label,
                ci_text(row),
                fmt_p(row["exact_sign_flip_p"]),
                f"{100 * row['positive_seed_fraction']:.0f}%",
            ]
        )
    value_transport, routing_transport, full_transport, value_minus_routing = (
        read_metrics
    )
    routing_mediation = find_summary(summary, "read_routing_ov_mediation_specificity")
    value_mediation = find_summary(summary, "read_value_ov_mediation_specificity")
    read_test_rows = [
        [
            "1 · routing-only",
            "用 donor α、receiver V（DR−RR，并在 donor-V background 复算）",
            ci_text(routing_transport),
            fmt_p(routing_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "2 · value-only",
            "用 receiver α、donor V（RD−RR，并在 donor-α background 复算）",
            ci_text(value_transport),
            fmt_p(value_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "3 · full donor",
            "同时换成 donor α 与 donor V（DD−RR）",
            ci_text(full_transport),
            fmt_p(full_transport["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "4 · value−routing",
            "直接比较两部分的 transport 大小",
            ci_text(value_minus_routing),
            fmt_p(value_minus_routing["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "5a · routing 经 OV",
            "自然轴 block 比正交 block 额外削弱 routing-only effect",
            ci_text(routing_mediation),
            fmt_p(routing_mediation["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "5b · value 经 OV",
            "自然轴 block 比正交 block 额外削弱 value-only effect",
            ci_text(value_mediation),
            fmt_p(value_mediation["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
    ]
    write_behavior = find_summary(summary, "write_behavior_specificity")
    write_table_rows = [
        [
            f"L{int(row['layer'])}",
            ci_text(row),
            fmt_p(row["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ]
        for row in write_rows
    ]
    write_table_rows.append(
        [
            "answer distribution",
            ci_text(write_behavior),
            fmt_p(write_behavior["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ]
    )
    return f"""
<section id="read-write">
<h2>9 · State 如何被读出并写回：mixed α/V read 与下游传播</h2>
<h3>9.1 第一步：把“看哪里”和“读到什么”拆开</h3>
<div class="test-card"><h4>Crossed α/V intervention 的四个 endpoint</h4><dl>
<dt>RR</dt><dd>receiver 的 attention routing α + receiver 的 value content V；这是 reference。</dd>
<dt>RD</dt><dd>receiver α + donor V；只改变“读到什么”。</dd>
<dt>DR</dt><dd>donor α + receiver V；只改变“看哪里/以什么权重读”。</dd>
<dt>DD</dt><dd>donor α + donor V；完整 donor pre-O endpoint。</dd>
</dl></div>
<p>如果 routing-only 能推动 donor count，说明选择 source 的 α 参与读取；如果 value-only 能推动，说明 source residual 中已经有可用 content。只有 component 的行为效应还会被第 8 节冻结的 natural OV axis 特异阻断，才把它视作当前自然通路的一部分。</p>
<figure>{read_forest}<figcaption><strong>Figure · L28 read decomposition.</strong> 横轴是 normalized donor behavioral transport；0 表示 component 没有把 answer distribution 推向 donor count，正值表示 donor count 获得概率。点为 20 个 evaluation seed 的 paired mean，横线为 seed bootstrap 95% CI。前三行是 component/full transport；最后一行是 value−routing 差，因此其区间跨 0 表示两者量级无法区分。</figcaption></figure>
{table(["检验", "具体替换/阻断", "effect [95% CI]", "exact p", "p<0.05?"], read_test_rows)}
<div class="step-result"><strong>显著性判读。</strong>routing=0.0517、value=0.0524、full=0.1140 的 exact p 都是 9.54×10<sup>−7</sup>，均显著；value−routing=0.0008 [−0.0105, 0.0129], p=0.451，不显著。因此证据支持“两部分都有”，但不能说哪一部分更大。routing 与 value 的 natural-OV mediation p 分别为 9.54×10<sup>−7</sup> 和 5.63×10<sup>−5</sup>，也都显著。</div>
<div class="conclusion"><strong>本段结论</strong>L28 H16/H19 采用 mixed read：既依赖 α 决定从哪些 source states 取信息，也依赖那些 source states 的 V content。当前精度下两者大小无显著差异（p=0.451）。</div>

<h3>9.2 第二步：检查 L28 写入是否一路存活到输出</h3>
<p><strong>具体操作：</strong>在 L28 真实 pre-O 边界施加 +β 与 −β natural z-step，经过 H16/H19 自身 W<sub>O</sub> 后继续正常 forward；matched control 位于相同 W<sub>O</sub> span、post-O norm 相同但与自然方向正交。每一层的 coefficient 是中心差分 residual change 在 discovery-frozen answer-count step <code>s<sub>l</sub></code> 上的归一化投影。</p>
<div class="equation">coefficient<sub>l</sub> = ⟨[h<sub>l</sub>(+β)−h<sub>l</sub>(−β)]/(2β), s<sub>l</sub>⟩ / ||s<sub>l</sub>||².</div>
<figure>{write_svg}<figcaption><strong>Figure · Downstream survival of the L28 OV write.</strong> 横轴为 decoder layer L28–L35；纵轴是 natural intervention coefficient 减 same-span orthogonal-control coefficient，单位为该层自然 answer-count step。点为 seed mean，竖线为 95% CI，0 表示 natural 与 orthogonal directions 的传播相同。所有 layer 的 Holm p≤2.29×10<sup>−5</sup>。</figcaption></figure>
{table(["readout site", "natural−orth specificity [95% CI]", "exact p", "p<0.05?"], write_table_rows)}
<p><strong>零假设：</strong>natural 与同 span 的正交方向传播相同，即 specificity=0。L28–L35 每层的区间都在 0 以上；layer family 校正后的最大 p 出现在 L35，为 2.29×10<sup>−5</sup>，仍小于 0.05。answer distribution specificity=0.0685 [0.0478, 0.0912], p=9.54×10<sup>−7</sup>，也显著。</p>
<div class="callout warning"><strong>证据边界。</strong>read/write extension 复用了 parent V4.4.4 的 evaluation seeds；它是冻结候选后的机制扩展，但不是全新 seed 的独立复制。axes 在 outcome 分层前冻结，因此 correct/wrong sensitivity 不会通过重新拟合改变 geometry。</div>
<div class="conclusion"><strong>本节结论</strong>当前可定位的 terminal chain 是：输入 L28 的 state → H16/H19 mixed α/V read → natural OV write → L29–L35 count-aligned residual → count distribution。这里已经验证“如何读、如何写”；上游是谁把可读 state 送到 L28，由第 10 节单独检验。</div>
</section>
"""


def build_gemma_read_write_appendix(
    read_write: dict[str, Any],
    natural_ov: dict[str, Any],
    *,
    heading: str = "9.3",
    natural_heading: str = "8.4",
) -> str:
    summary = read_write["summary"]
    cfg = read_write["config"]
    mediator_layer = int(cfg["mediator_layer"])
    heads = tuple(int(head) for head in cfg["heads"])
    candidate_label = f"L{mediator_layer} " + "/".join(f"H{head}" for head in heads)
    if candidate_label != ov_candidate_label(natural_ov):
        raise RuntimeError(
            f"Gemma read/write parent mismatch: {candidate_label} vs "
            f"{ov_candidate_label(natural_ov)}"
        )
    alpha = float(cfg.get("primary_alpha", 0.05))
    routing = find_summary(summary, "read_routing_behavior_transport")
    value = find_summary(summary, "read_value_behavior_transport")
    full = find_summary(summary, "read_full_behavior_transport")
    difference = find_summary(summary, "read_value_minus_routing_transport")
    routing_mediation = find_summary(summary, "read_routing_ov_mediation_specificity")
    value_mediation = find_summary(summary, "read_value_ov_mediation_specificity")
    write_behavior = find_summary(summary, "write_behavior_specificity")
    write_rows = sorted(
        [
            row
            for row in summary
            if row.get("metric") == "write_residual_specificity"
            and row.get("stratum") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    read_forest = forest_svg(
        [
            {
                "label": "α/routing component",
                "mean": routing["mean"],
                "low": routing["ci95_low"],
                "high": routing["ci95_high"],
                "value": f"{ci_text(routing)} · p={fmt_p(routing['exact_sign_flip_p'])}",
            },
            {
                "label": "V/content component",
                "mean": value["mean"],
                "low": value["ci95_low"],
                "high": value["ci95_high"],
                "value": f"{ci_text(value)} · p={fmt_p(value['exact_sign_flip_p'])}",
            },
            {
                "label": "full donor-z patch",
                "mean": full["mean"],
                "low": full["ci95_low"],
                "high": full["ci95_high"],
                "value": f"{ci_text(full)} · p={fmt_p(full['exact_sign_flip_p'])}",
            },
            {
                "label": "value − routing",
                "mean": difference["mean"],
                "low": difference["ci95_low"],
                "high": difference["ci95_high"],
                "value": f"{ci_text(difference)} · p={fmt_p(difference['exact_sign_flip_p'])}",
            },
        ],
        title=f"Factorized read contributions at Gemma {candidate_label}",
        description="Routing, value, full-patch and value-minus-routing effects on the frozen Gemma evaluation seeds.",
        x_label="normalized donor behavioral transport (positive = donor count gains probability)",
    )
    write_svg = write_trace_svg(
        write_rows,
        id_prefix="gemma-write-"
        + re.sub(r"[^a-z0-9]+", "-", candidate_label.lower()).strip("-"),
        title=f"Gemma {candidate_label} natural OV write propagation",
        description="Layer is on the horizontal axis. Natural-minus-orthogonal count-axis coefficient is on the vertical axis. Points are seed means and bars are 95 percent bootstrap confidence intervals.",
    )

    def positive_badge(row: dict[str, Any], p_key: str) -> str:
        supported = float(row["mean"]) > 0 and float(row[p_key]) < alpha
        return sig_badge(
            0.0 if supported else 1.0, label="支持" if supported else "不支持"
        )

    read_rows = [
        [
            "routing-only",
            "donor α + receiver V",
            ci_text(routing),
            fmt_p(routing["exact_sign_flip_p"]),
            positive_badge(routing, "exact_sign_flip_p"),
        ],
        [
            "value-only",
            "receiver α + donor V",
            ci_text(value),
            fmt_p(value["exact_sign_flip_p"]),
            positive_badge(value, "exact_sign_flip_p"),
        ],
        [
            "full donor",
            "donor α + donor V",
            ci_text(full),
            fmt_p(full["exact_sign_flip_p"]),
            positive_badge(full, "exact_sign_flip_p"),
        ],
        [
            "value−routing",
            "两种 component 量级之差",
            ci_text(difference),
            fmt_p(difference["exact_sign_flip_p"]),
            sig_badge(difference["exact_sign_flip_p"], alpha=alpha),
        ],
        [
            "routing 经 natural OV",
            "orthogonal block − natural-axis block",
            ci_text(routing_mediation),
            fmt_p(routing_mediation["exact_sign_flip_p"]),
            positive_badge(routing_mediation, "exact_sign_flip_p"),
        ],
        [
            "value 经 natural OV",
            "orthogonal block − natural-axis block",
            ci_text(value_mediation),
            fmt_p(value_mediation["exact_sign_flip_p"]),
            positive_badge(value_mediation, "exact_sign_flip_p"),
        ],
    ]
    write_table_rows = [
        [
            f"L{int(row['layer'])}",
            ci_text(row),
            fmt_p(row["exact_sign_flip_p"]),
            fmt_p(row.get("holm_p_within_family_metric")),
            positive_badge(row, "holm_p_within_family_metric"),
        ]
        for row in write_rows
    ]
    write_table_rows.append(
        [
            "answer distribution",
            ci_text(write_behavior),
            fmt_p(write_behavior["exact_sign_flip_p"]),
            "N/A",
            positive_badge(write_behavior, "exact_sign_flip_p"),
        ]
    )
    decision = read_write["primary_decision"]
    read_mode = decision["read_mode"]
    write_decision = decision["write_propagation"]
    full_supported = bool(decision["serial_read_write_supported"])
    natural_supported = bool(
        natural_ov["primary_decision"]["full_natural_ov_transporter_support"]
    )
    read_description = {
        "mixed": "routing 与 value/content 两部分都通过其 transport+OV-mediation family，属于 mixed read",
        "routing_only": "只确认 routing component",
        "value_only": "只确认 value/content component",
        "none": "未确认 routing 或 value component",
    }.get(str(read_mode["classification"]), str(read_mode["classification"]))
    if full_supported and natural_supported:
        result_conclusion = "Gemma 复现了 terminal mixed-read → natural-OV-write → downstream count-aligned state 这一段路径。"
    elif full_supported:
        result_conclusion = (
            f"Gemma 的候选 {candidate_label} channel 通过了 factorized read 与 intervention-induced downstream propagation；"
            "但其 parent natural-OV 四族检验未通过，所以这里只能称为候选轴的机械 read/write coherence，"
            "不能升级为模型 clean forward 中自然使用的 read/write replication。"
        )
    else:
        result_conclusion = f"Gemma 没有通过完整 read/write 联合判定：read classification={read_mode['classification']}，write supported={str(bool(write_decision['supported'])).lower()}。"
    return f"""
<h3>{heading} Gemma {candidate_label}：可访问 state 的 α/V 读取与写入传播</h3>
<p>Gemma 的 α/V 分解复用 natural-OV confirmation seeds {seed_span(cfg["evaluation_seeds"])}，因此是冻结候选后的机制分解，不是第二次独立复制。计算构造 RR、RD、DR、DD 四个 pre-O endpoint；value content 必须经过 Gemma 自己的 V projection 与 value normalization，不能套用 Qwen 的线性 <em>W</em><sub>V</sub> 近似。</p>
<div class="callout warning"><strong>可见窗口定义。</strong>{candidate_label} 的 <code>all_positions</code> 定义为该层实际 capture 到的全部 keys；slot、early non-slot、tail non-slot 和 query-self 都先与该窗口取交集，完全落在窗外的组记为 0，而不是当作缺失。因而本实验检验的是“该 set 如何读取当前可访问 state”，<strong>不检验也不声称它直接注意原始 needles</strong>。</div>
<figure>{read_forest}<figcaption><strong>Figure · Gemma {candidate_label} factorized read.</strong> 横轴是 normalized donor behavioral transport；正值表示答案分布向 donor count 移动。点为 {len(cfg["evaluation_seeds"])} 个 seed means，横线为 seed-cluster bootstrap 95% CI。最后一行是 value−routing 差值，而不是第三种 transport component。</figcaption></figure>
{table(["检验", "替换", "effect [95% CI]", "exact p", f"p<{fmt(alpha, 3)}?"], read_rows)}
<p>联合 read classification 为 <code>{html.escape(str(read_mode["classification"]))}</code>：{read_description}。component 是否属于自然通路，不只看 behavioral transport，还要求相应 effect 被第 {html.escape(natural_heading)} 节冻结的 natural OV axis 特异阻断。</p>
<div class="conclusion"><strong>读取结论</strong>{read_description}；value−routing 的差异是否显著必须按其 own p={fmt_p(difference["exact_sign_flip_p"])} 判断，不能仅凭两个点估计的高低排序。</div>
<figure>{write_svg}<figcaption><strong>Figure · Gemma L{mediator_layer}→L{int(write_rows[-1]["layer"])} write propagation.</strong> 横轴是 decoder layer；纵轴是在 frozen layer-specific answer-count step 上，natural pre-O intervention coefficient 减 same-span equal-post-O-norm orthogonal control coefficient。点为 seed mean，竖线为 95% CI，0 表示 natural 与正交方向传播相同。</figcaption></figure>
{table(["readout site", "natural−orth specificity [95% CI]", "exact p", "Holm p", f"校正后 p<{fmt(alpha, 3)}?"], write_table_rows)}
<p>write family 的最终层为 L{int(write_decision["final_layer"])}：specificity={fmt(write_decision["final_residual_specificity_mean"], 4)}，Holm p={fmt_p(write_decision["final_residual_specificity_holm_p"])}；answer-distribution specificity={fmt(write_decision["behavior_specificity_mean"], 4)}，p={fmt_p(write_decision["behavior_specificity_p"])}。完整 read/write 判定为 <code>serial_read_write_supported={str(full_supported).lower()}</code>。</p>
<div class="conclusion"><strong>本小节结论</strong>{result_conclusion}无论结果正负，它都只描述 {candidate_label} 可访问的 state 与后续传播；其上游来源必须由独立 serial/relay intervention 判定。</div>
"""


def build_upstream_section(relay: dict[str, Any], upstream: dict[str, Any]) -> str:
    metric_by_name = {row["metric"]: row for row in relay["metric_summary"]}
    relay_svg = relay_gate_svg(
        [
            {
                "label": "carrier",
                "value": fmt(metric_by_name["natural_relay_slope"]["mean"], 4),
                "p": f"p={fmt_p(metric_by_name['natural_relay_slope']['exact_sign_flip_p'])}",
                "passed": True,
            },
            {
                "label": "V-only first stage",
                "value": fmt(
                    metric_by_name["edge_patch_first_stage_transport"]["mean"], 4
                ),
                "p": f"p={fmt_p(metric_by_name['edge_patch_first_stage_transport']['exact_sign_flip_p'])}",
                "passed": True,
            },
            {
                "label": "behavior",
                "value": fmt(
                    metric_by_name["edge_patch_behavior_transport"]["mean"], 4
                ),
                "p": f"p={fmt_p(metric_by_name['edge_patch_behavior_transport']['exact_sign_flip_p'])}",
                "passed": False,
            },
            {
                "label": "OV mediation",
                "value": fmt(metric_by_name["ov_mediation_specificity"]["mean"], 4),
                "p": f"p={fmt_p(metric_by_name['ov_mediation_specificity']['exact_sign_flip_p'])}",
                "passed": False,
            },
            {
                "label": "natural removal",
                "value": "wrong direction",
                "p": "family p=0.9981",
                "passed": False,
            },
        ]
    )
    relay_table_rows = [
        [
            "1 · carrier",
            "clean forward 中 tail-64 natural contribution 对 count 的斜率",
            ci_text(metric_by_name["natural_relay_slope"]),
            fmt_p(metric_by_name["natural_relay_slope"]["exact_sign_flip_p"]),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "2 · V-only first stage",
            "固定 receiver Q/K/α，只 patch tail-64 value content",
            ci_text(metric_by_name["edge_patch_first_stage_transport"]),
            fmt_p(
                metric_by_name["edge_patch_first_stage_transport"]["exact_sign_flip_p"]
            ),
            '<span class="sig-yes">显著</span>',
        ],
        [
            "3 · answer behavior",
            "检查 V-only patch 是否把最终分布推向 donor count",
            ci_text(metric_by_name["edge_patch_behavior_transport"]),
            fmt_p(metric_by_name["edge_patch_behavior_transport"]["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "4 · L28 OV mediation",
            "自然 L28 block 是否比正交 block 多消除 patch effect",
            ci_text(metric_by_name["ov_mediation_specificity"]),
            fmt_p(metric_by_name["ov_mediation_specificity"]["exact_sign_flip_p"]),
            '<span class="sig-no">不显著</span>',
        ],
        [
            "5a · removal error",
            "删除 tail-64 natural axis 是否比正交删除增加更多误差",
            ci_text(metric_by_name["relay_removal_error_specificity"]),
            fmt_p(
                metric_by_name["relay_removal_error_specificity"]["exact_sign_flip_p"]
            ),
            '<span class="sig-no">反方向</span>',
        ],
        [
            "5b · removal margin",
            "删除 tail-64 natural axis 是否比正交删除降低更多正确 margin",
            ci_text(metric_by_name["relay_removal_margin_specificity"]),
            fmt_p(
                metric_by_name["relay_removal_margin_specificity"]["exact_sign_flip_p"]
            ),
            '<span class="sig-no">反方向</span>',
        ],
    ]
    primary = upstream["primary_decision"]
    early = primary["early_effect"]
    mediation = primary["mediation"]
    path_forest = forest_svg(
        [
            {
                "label": "early top-4 slot-state patch",
                "mean": early["mean"],
                "low": early["ci_low"],
                "high": early["ci_high"],
                "value": f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}] · p={fmt_p(early['exact_two_sided_p'])}",
            },
            {
                "label": "L28 mediation specificity",
                "mean": mediation["mean"],
                "low": mediation["ci_low"],
                "high": mediation["ci_high"],
                "value": f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}] · p={fmt_p(mediation['exact_two_sided_p'])}",
            },
        ],
        title="Independent serial-path confirmation",
        description="Early donor log-odds gain and the orthogonal-control minus exact-L28-block mediation specificity are both positive on fresh seeds.",
        x_label="donor-vs-receiver candidate-sequence log-odds units",
    )
    loo_rows = []
    loo_table_rows = []
    for item in upstream["leave_one_out"]:
        dec = item["decrement"]
        supported = bool(item["incremental_contribution_supported"])
        loo_rows.append(
            {
                "label": f"remove {item['removed_head']}",
                "mean": dec["mean"],
                "low": dec["ci_low"],
                "high": dec["ci_high"],
                "color": "#D94B86" if supported else "#718096",
                "value": f"{fmt(dec['mean'], 4)} · Holm p={fmt_p(item['decrement_holm_p'])}",
            }
        )
        loo_table_rows.append(
            [
                item["removed_head"],
                fmt(item["loo_mediation"]["mean"], 4),
                f"{fmt(dec['mean'], 4)} [{fmt(dec['ci_low'], 4)}, {fmt(dec['ci_high'], 4)}]",
                fmt_p(dec["exact_two_sided_p"]),
                fmt_p(item["decrement_holm_p"]),
                "necessary within tested set"
                if supported
                else "no unique decrement resolved",
            ]
        )
    loo_forest = forest_svg(
        loo_rows,
        title="Leave-one-out membership in the L28 H16-H19 mediator set",
        description="Positive full-minus-LOO decrement means removing the named head weakens mediation. Only H19 survives Holm correction.",
        x_label="full-set mediation − leave-one-out mediation (log-odds units)",
    )
    return f"""
<section id="upstream">
<h2>10 · 上游 relay 与独立 serial-path confirmation</h2>
<h3>10.1 候选 relay：为什么“有信息”仍然不够</h3>
<div class="test-card"><h4>被检验的链路</h4><dl>
<dt>候选位置</dt><dd><code>pre_query_non_slot_tail_64</code>：answer query 前最后 64 个非 slot tokens；在 discovery 上冻结。</dd>
<dt>假说</dt><dd>这些 late positions 保存 count content，receiver attention 自然读取它，再经 L28 H16/H19 写到答案。</dd>
<dt>判定规则</dt><dd>carrier、V-only first stage、answer behavior、L28 mediation、natural removal 必须全部沿预定方向显著；任一门失败就不能称为自然 relay。</dd>
</dl></div>
<figure>{
        relay_svg
    }<figcaption><strong>Figure · tail-64 relay gate.</strong> 这是串行证据门图，没有共同数值坐标轴；每个框显示该阶段的 seed mean 与 exact p。绿色框通过预定方向，粉色框失败。carrier 与机械 first stage 成立，但 answer-level transport 区间跨 0，OV mediation 为 0，removal 方向相反，因此 global IUT p=0.9981。</figcaption></figure>
{
        table(
            [
                "步骤",
                "具体操作/问题",
                "effect [95% CI]",
                "exact directional p",
                "p<0.05 且方向正确?",
            ],
            relay_table_rows,
        )
    }
<p><strong>逐门判读：</strong>carrier p=9.54×10<sup>−7</sup>、V-only first-stage p=0.000405，均显著，说明 tail-64 中确有可读 content；但 answer behavior p=0.0693、OV mediation p=0.508，均不显著。两个 removal endpoint 还朝预注册方向的反面变化，directional p 分别为 0.9948 和 0.9981。global IUT p=0.9981，不显著。</p>
<div class="conclusion"><strong>本段结论</strong>tail-64 position set “可解码且可机械访问”，但没有证据表明模型自然依赖它把 count 送到 answer。否定仅针对这个冻结 position set，不否定其他 token set 或 MLP relay。</div>

<h3>10.2 真正得到支持的上游路径：fresh-seed serial mediation</h3>
<div class="plain-protocol"><h4>三次 forward 比较</h4><ol>
<li><strong>Source patch：</strong>把 donor 在 slot-query positions 上的 early top-4 set-output state patch 给 receiver。top-4 冻结为 L23H28、L23H29、L26H20、L27H18。</li>
<li><strong>Exact L28 block：</strong>在 L28 H16–H19 的 pre-O z 中，精确删除 source patch 诱发的自然 change。</li>
<li><strong>Matched control：</strong>删除同一 W<sub>O</sub> span、相同 post-O norm、但与自然 change 正交的方向。若 natural block 比 control 多消除 donor effect，才叫 mediation specificity。</li>
</ol></div>
<p>确认实验使用全新 seeds 1294–1313、六个 directed donor pairs；route、head set、endpoint 与 control construction 均在看这些 seeds 前冻结。零假设有两个：early source patch 不产生 donor gain；或 natural L28 block 不比正交 control 多消除 gain。</p>
<figure>{
        path_forest
    }<figcaption><strong>Figure · Independent serial mediation.</strong> 横轴统一为 donor-vs-receiver candidate-sequence log-odds units。第一行是 early slot-state patch 相对 clean 的 donor log-odds gain；第二行是 orthogonal control 保留的 gain 减 exact natural block 保留的 gain，即 L28 mediation specificity。点为 20 个 fresh-seed paired mean，线为 95% bootstrap CI，0 是无效应边界。</figcaption></figure>
{
        table(
            ["必要门", "effect [95% CI]", "exact two-sided p", "p<0.05?", "验证内容"],
            [
                [
                    "early source effect",
                    f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}]",
                    fmt_p(early["exact_two_sided_p"]),
                    '<span class="sig-yes">显著</span>',
                    "early top-4 slot-state patch 能推动 donor count",
                ],
                [
                    "L28 mediation specificity",
                    f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}]",
                    fmt_p(mediation["exact_two_sided_p"]),
                    '<span class="sig-yes">显著</span>',
                    "该 donor effect 特异经过 L28 H16–H19 natural change",
                ],
            ],
        )
    }
<p>两门 conjunction 的 IUT p 取较大值 0.005884，小于 0.05，故串行路径显著。120 primary rows、480 LOO rows以及 block closure、orthogonality、deterministic prefill audits 全部通过。</p>
<div class="conclusion"><strong>本段结论</strong>“early broad top-4 slot-state → L28 H16–H19 → answer”在独立 seeds 上复现。由于这是 donor-induced path mediation，它证明该通路能够并确实介导受控 source perturbation；它尚未单独证明 early top-4 在未干预 clean forward 中逐头必要。</div>

<h3>10.3 哪个 L28 head 对 set mediation 不可替代</h3>
<p><strong>具体操作：</strong>从完整 H16–H19 mediator set 中每次去掉一个 head，重新计算 mediation；定义 decrement=完整 set mediation−leave-one-out mediation。正 decrement 表示被去掉的 head 贡献不能由剩余 heads 替代。四次比较使用 Holm 校正，因此显著阈值看 Holm p&lt;0.05，而不是 raw p。</p>
<figure>{
        loo_forest
    }<figcaption><strong>Figure · Leave-one-out head-set membership.</strong> 横轴是 full H16–H19 mediation 减去移除指定 head 后的 mediation；正值表示该 head 对 set mediation 有不可由剩余成员替代的增量贡献。点与区间均为 20 个 fresh seeds 的 paired estimates；颜色仅区分 Holm-corrected support，统计意义不依赖颜色。</figcaption></figure>
{
        table(
            [
                "removed",
                "LOO mediation",
                "full−LOO [95% CI]",
                "exact p",
                "Holm p",
                "interpretation",
            ],
            loo_table_rows,
        )
    }
<p>移除 H19 的 decrement=0.1538 [0.0783, 0.2291]，raw p=0.00101、Holm p=0.00404，显著；剩余 H16–H18 的 mediation=0.0171，p=0.518，不显著。H16 decrement 的 raw p=0.0391，但 Holm p=0.117，不显著；H17/H18 Holm p=1，也不显著。</p>
<div class="conclusion"><strong>本节结论</strong>H19 是当前 H16–H19 mediator set 内的非冗余锚点；H16/H17/H18 更像冗余或支持性 companion subspace。这个结果不证明 H19 单头充分，也不把 counting 简化成 H19 的单头算法。</div>
</section>
"""


def build_gemma_serial_appendix(upstream: dict[str, Any]) -> str:
    cfg = upstream["config"]
    primary = upstream["primary_decision"]
    early = primary["early_effect"]
    mediation = primary["mediation"]
    confirmed = bool(primary["serial_chain_confirmed"])
    mediator_layer = int(cfg["mediator_layer"])
    early_heads = [f"L{int(row[0])}H{int(row[1])}" for row in cfg["early_candidates"]]
    primary_late = str(cfg["primary_late_set"])
    late_sets = {
        str(name): [int(head) for head in heads]
        for name, heads in cfg["late_head_sets"]
    }
    late_heads = late_sets[primary_late]
    late_label = "/".join(f"H{head}" for head in late_heads)
    path_forest = forest_svg(
        [
            {
                "label": "frozen broad-set slot-state patch",
                "mean": early["mean"],
                "low": early["ci_low"],
                "high": early["ci_high"],
                "value": f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}] · p={fmt_p(early['exact_two_sided_p'])}",
            },
            {
                "label": f"L{mediator_layer} natural-block specificity",
                "mean": mediation["mean"],
                "low": mediation["ci_low"],
                "high": mediation["ci_high"],
                "value": f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}] · p={fmt_p(mediation['exact_two_sided_p'])}",
            },
        ],
        title="Independent Gemma early-to-L37 serial mediation",
        description="The first row is the donor log-odds gain caused by the frozen broad-set slot-state patch. The second is orthogonal-control minus exact-natural-block retained gain at L37.",
        x_label="donor-vs-receiver candidate-sequence log-odds units",
    )
    loo_rows = []
    for item in upstream.get("leave_one_out", []):
        decrement = item["decrement"]
        loo_rows.append(
            [
                str(item["removed_head"]),
                fmt(item["loo_mediation"]["mean"], 4),
                f"{fmt(decrement['mean'], 4)} [{fmt(decrement['ci_low'], 4)}, {fmt(decrement['ci_high'], 4)}]",
                fmt_p(decrement["exact_two_sided_p"]),
                fmt_p(item["decrement_holm_p"]),
                "set 内非冗余"
                if bool(item["incremental_contribution_supported"])
                else "未解析出独立增量",
            ]
        )
    path_conclusion = (
        f"冻结 broad set（{', '.join(early_heads)}）产生 donor gain，且该 gain 被 L{mediator_layer} {late_label} 的自然 pre-O change 特异介导；Gemma 的受限 early→terminal 串联路径在 fresh seeds 上确认。"
        if confirmed
        else f"这组 frozen broad set → L{mediator_layer} {late_label} 没有通过两门 IUT；不能把它写成 Gemma 已确认的串联来源，即使某一单门或某个 LOO 对比为正。"
    )
    return f"""
<h3>10.4 Gemma fresh-seed 串联检验：frozen broad set → L{mediator_layer} {
        late_label
    }</h3>
<p>上游 set 没有根据本轮 outcome 重新排序：它冻结自 correct-only causal-v2 的 broad-aggregation K=2（{
        ", ".join(early_heads)
    }），晚端 mediator 冻结为 L{mediator_layer} {late_label}。确认数据使用 {
        seed_span(cfg["evaluation_seeds"])
    }、counts {min(cfg["counts"])}–{max(cfg["counts"])} 与 {
        len(cfg["donor_pairs"])
    } 个 directed donor pairs；每个 seed 而非 pair/token 是独立推断单位。</p>
<div class="plain-protocol"><h4>Gemma 串联对比的四个 forward</h4><ol>
<li>clean receiver；</li>
<li>只在 registered active-slot query positions patch frozen broad-set output；</li>
<li>同一 early patch，再精确恢复其诱发的 L{mediator_layer} {
        late_label
    } pre-O z change（natural block）；</li>
<li>同一 early patch，再删除相同 <em>W</em><sub>O</sub> span、相同 post-O norm、但与自然 change 正交的 control。</li>
</ol></div>
<p>第一门要求 early patch 的 donor-vs-receiver candidate-sequence log-odds gain&gt;0；第二门要求 <code>M=gain<sub>orthogonal</sub>−gain<sub>natural block</sub>&gt;0</code>。全局 IUT p 是两门 exact seed-level sign-flip p 的最大值；closure≤10<sup>−5</sup>、orthogonality≤10<sup>−4</sup> 与 deterministic-prefill≤10<sup>−5</sup> 还必须全部通过。</p>
<div class="conclusion"><strong>设计结论</strong>这个检验验证的是受控 source perturbation 是否沿 frozen late channel 传播；它足以建立一条受支持的逻辑通路，但不要求证明这是模型唯一 relay，也不把每个 early head 都写成 clean forward 中逐头必要。</div>
<figure>{
        path_forest
    }<figcaption><strong>Figure · Gemma independent serial mediation.</strong> 横轴统一为 donor-vs-receiver candidate-sequence log-odds。第一行是 early slot-state patch 相对 clean 的 gain；第二行是正交 control 保留的 gain 减 exact natural block 保留的 gain。点是 {
        len(cfg["evaluation_seeds"])
    } 个 fresh-seed means，横线是 seed-cluster bootstrap 95% CI，0 为无效应。</figcaption></figure>
{
        table(
            ["必要门", "effect [95% CI]", "exact two-sided p", "p<0.05?", "含义"],
            [
                [
                    "early source effect",
                    f"{fmt(early['mean'], 4)} [{fmt(early['ci_low'], 4)}, {fmt(early['ci_high'], 4)}]",
                    fmt_p(early["exact_two_sided_p"]),
                    sig_badge(
                        early["exact_two_sided_p"] if early["mean"] > 0 else 1.0,
                        label="支持"
                        if early["mean"] > 0 and early["exact_two_sided_p"] < 0.05
                        else "不支持",
                    ),
                    "frozen broad set 能否推动 donor count",
                ],
                [
                    f"L{mediator_layer} mediation specificity",
                    f"{fmt(mediation['mean'], 4)} [{fmt(mediation['ci_low'], 4)}, {fmt(mediation['ci_high'], 4)}]",
                    fmt_p(mediation["exact_two_sided_p"]),
                    sig_badge(
                        mediation["exact_two_sided_p"]
                        if mediation["mean"] > 0
                        else 1.0,
                        label="支持"
                        if mediation["mean"] > 0
                        and mediation["exact_two_sided_p"] < 0.05
                        else "不支持",
                    ),
                    "donor effect 是否特异经过 frozen late set",
                ],
            ],
        )
    }
<p>串联判定为 <code>serial_chain_confirmed={str(confirmed).lower()}</code>，IUT p={
        fmt_p(primary["intersection_union_p"])
    }。这里的 p 不是把 donor pairs 当独立样本得到的；exact sign-flip 只作用于 seed means，95% CI 也按 seed cluster bootstrap。</p>
<div class="conclusion"><strong>串联结果</strong>{path_conclusion}</div>
{
        details_table(
            "Gemma late-set leave-one-out",
            [
                "removed",
                "LOO mediation",
                "full−LOO [95% CI]",
                "exact p",
                "Holm p",
                "interpretation",
            ],
            loo_rows,
        )
        if loo_rows
        else ""
    }
<p class="small">LOO 只回答“某个晚端 head 在当前 frozen set 内是否有不可由另一成员替代的增量贡献”；它不检验单头充分性，也不允许把 counting 简化为一个 head 的算法。</p>
<div class="conclusion"><strong>本小节边界</strong>正结果建立一条可复现的 Gemma early→late 逻辑通路；负结果只否定这组冻结 broad set 作为已确认上游来源，不否定其他 early sets、MLP-mediated relay 或并行通道。</div>
"""


def build_synthesis_section() -> str:
    claim_rows = [
        [
            "模型在 prompt 读取阶段形成 running index",
            "跨 position/order/content 的 frozen-basis geometry；cue removal 高 CKA",
            "强表征证据；不是因果运算证明",
        ],
        [
            "检索是分布式而非严格单头",
            "attention atlas；ranked bank perturbation；early top-4 route",
            "支持；特定 early heads 的 clean necessity 未完成",
        ],
        [
            "late answer state 是可执行 count carrier",
            "answer-state patch、steering、late geometry",
            "跨 Qwen/Gemma 功能因果支持",
        ],
        [
            "Qwen L28 H16/H19 是自然 OV transporter",
            "四族 IUT：signal/injection/removal/mediation",
            "确认；部分 mediation",
        ],
        [
            "Qwen L28 read 同时依赖 α 与 V",
            "crossed α–V decomposition + OV block",
            "支持；parent seeds 机制扩展",
        ],
        [
            "early slot state 经 L28 H16–H19 到 answer",
            "fresh-seed exact-block serial mediation",
            "独立确认的受限链路",
        ],
        ["tail-64 是自然 relay", "registered four-family relay test", "不支持"],
        ["存在唯一逐 token +1 head", "当前没有直接测试支持", "不可声称"],
    ]
    step_rows = [
        [
            "1 · Prompt representation",
            "读取第 n 个 active record 后，在 needle-end residual 观察 ordered running-index state",
            "跨 seed 的 frozen-basis geometry；cue-present/absent shared-basis audit",
            "N/A：表征图不是单一因果检验",
            "存在可解码的累计进度；不等于模型已使用",
        ],
        [
            "2 · Early source read",
            "patch early top-4 在 slot-query positions 的 set-output state",
            "donor log-odds gain=0.1057 [0.0412, 0.1683]",
            "p=0.005884，显著",
            "early broad bank 可把 slot-state signal 送向答案",
        ],
        [
            "3 · L28 mixed read",
            "在 L28 H16–H19 分别替换 α routing 与 V content",
            "routing=0.0517；value=0.0524；value−routing=0.0008",
            "两 component p=9.54×10⁻⁷，显著；差值 p=0.451，不显著",
            "两种读取方式都参与，大小无法区分",
        ],
        [
            "4 · Natural OV write",
            "pre-O injection、centered removal、donor-path block 与 matched controls",
            "四个必要 evidence families 全部通过",
            "global IUT p=0.004541，显著",
            "H16/H19 是自然使用的部分 transporter",
        ],
        [
            "5 · Downstream survival",
            "比较 natural 与 same-span orthogonal step 在 L28–L35 的 count-axis projection",
            "L35 specificity=0.0156 [0.0108, 0.0201]",
            "layer-family 最大校正 p=2.29×10⁻⁵，显著",
            "L28 写入没有在后层立即消失",
        ],
        [
            "6 · Answer readout",
            "patch 完整 Total: query state，观察是否采用 donor prediction",
            "Qwen clean-correct pooled adoption=96.6%",
            "描述性 pooled rate；此 supplement 不提供单一确认 p",
            "late answer state 可执行地携带已算出的 prediction",
        ],
        [
            "Rejected branch",
            "对 tail-64 relay 做 carrier、edge、behavior、mediation、removal conjunction",
            "behavior/mediation 不通过，removal 反方向",
            "global IUT p=0.9981，不显著",
            "不能把 tail-64 写成自然 relay",
        ],
    ]
    return f"""
<section id="synthesis">
<h2>11 · Mechanism synthesis：把前面的实验翻译成一条可读的计数流程</h2>
<figure>{mechanism_svg()}<figcaption><strong>Figure · Supported non-thinking counting mechanism.</strong> 该图是因果结构图，没有数值坐标轴。实线箭头表示已有 transport/mediation 支持；框的粒度就是当前定位精度。灰色虚线分支标出被否定的 tail-64 relay。图中 early top-4→L28 链路已在 fresh seeds 复现；mixed read/write 分解仍是 parent-seed 机制扩展。</figcaption></figure>
<h3>11.1 六步机制与每一步的统计判定</h3>
<p>下面每一行只回答一个问题。先形成可解码 representation，再验证 source patch 能否到达答案；随后拆分 L28 的读取、验证其自然 OV 写入、检查写入能否存活，最后定位可执行的 answer state。不同 effect 量纲不相同，因此不跨行求平均。</p>
{table(["步骤", "具体做了什么", "主要观察", "显著性", "这一步允许的结论"], step_rows)}
<div class="step-result"><strong>最关键的 conjunction。</strong>early source effect 与 L28 mediation 的 fresh-seed IUT p=0.005884；natural OV 四证据族 global IUT p=0.004541；两者都小于 0.05。故可以把“early slot-state → L28 natural OV → late answer”写成受支持的串行机制。tail-64 的 p=0.9981，必须作为被否定的具体 relay 分支保留。</div>
<div class="conclusion"><strong>本段结论</strong>最小机制是：prompt 中形成分布式 running-index state；early bank 读取/汇集 slot-state signal；Qwen L28 H16/H19 以 mixed α/V 方式读取并通过自然 OV 方向写回；该写入存活至 late answer state。这里的 <em>W</em><sub>O</sub> 与后续 Jacobian 正是 prompt counter 和 answer counter 可以方向不同、但 count ordering 与因果信息保持的机制。我们没有证明唯一单头 +1 运算，也没有穷尽所有 relay。</div>

<h3>11.2 Claim matrix</h3>
{table(["可写入论文的命题", "证据", "允许的强度"], claim_rows)}
<div class="paper-wording"><strong>建议正文表述。</strong>“Across realistic 10k-token counting prompts, non-thinking models expressed an ordered prompt-side running-index geometry and a late answer-query count state. In Qwen3-8B, preregistered pre-output interventions identified a natural L28 OV transport channel: the H16/H19 set carried a count-correlated component, supported signed pre-O injection, was selectively necessary under centered z-space removal, and mediated donor-state transport. A factorized α–V intervention indicated mixed routing and value-content readout, while the induced OV write remained count-aligned through the final layer. Finally, an independent fresh-seed experiment confirmed serial mediation from a frozen early broad-retrieval slot-state set through L28 H16–H19 to the answer distribution; leave-one-out analysis identified H19 as nonredundant within this tested set.”</div>
<div class="conclusion"><strong>本节结论</strong>这套证据足以支持一个 set-level、分布式的 non-thinking read–write mechanism；不足以支持唯一单头、显式整数寄存器或完整无遗漏 circuit 的表述。论文正文应同时报告 p、effect、CI 与 seed 独立性边界。</div>
</section>
"""


def build_gemma_synthesis_appendix(
    gemma_ov: dict[str, Any],
    gemma_read_write: dict[str, Any],
    gemma_upstream: dict[str, Any],
) -> str:
    natural = bool(gemma_ov["primary_decision"]["full_natural_ov_transporter_support"])
    read_write = bool(
        gemma_read_write["primary_decision"]["serial_read_write_supported"]
    )
    serial = bool(gemma_upstream["primary_decision"]["serial_chain_confirmed"])
    full = natural and read_write and serial
    rows = [
        [
            "Prompt running-index representation",
            "frozen-basis PCA / full-space statistics / cue-paired audit",
            "支持；结构在 cue removal 后保留",
            '<span class="evidence descriptive">表征</span>',
        ],
        [
            "Frozen broad-retrieval function",
            "correct-only K=1/K=2 ablation vs 3 layer-matched random sets",
            "K1/K2 clean-correct failure CI 均排除 0",
            '<span class="evidence functional">独立功能支持</span>',
        ],
        [
            "L37 H1/H2 natural OV",
            "carrier + true pre-O injection + centered removal + mediation",
            f"global IUT p={fmt_p(gemma_ov['primary_decision']['global_intersection_union_p'])}",
            sig_badge(0.0 if natural else 1.0, label="通过" if natural else "未通过"),
        ],
        [
            "L37 terminal read/write",
            "sliding-window-aware crossed α/V + L37→L41 propagation",
            f"classification={html.escape(str(gemma_read_write['primary_decision']['read_mode']['classification']))}；supported={read_write}",
            sig_badge(
                0.0 if read_write else 1.0, label="通过" if read_write else "未通过"
            ),
        ],
        [
            "Frozen broad set → L37 → answer",
            "fresh-seed exact natural block vs same-span orthogonal control",
            f"IUT p={fmt_p(gemma_upstream['primary_decision']['intersection_union_p'])}",
            sig_badge(0.0 if serial else 1.0, label="通过" if serial else "未通过"),
        ],
        [
            "Late answer state",
            "clean-correct full-state patch",
            "pooled donor-target adoption 96.0%",
            '<span class="evidence functional">功能因果</span>',
        ],
    ]
    if full:
        claim = "Gemma 在冻结候选与 fresh seeds 上复现了与 Qwen 同构的分布式 read/write 路径；可做跨模型 mechanism replication，但 head/layer identity 不相同。"
        paper_sentence = (
            "In Gemma4-E4B, a separately frozen L37 H1/H2 set satisfied the same natural-OV "
            "signal, pre-output sufficiency, centered-necessity, and path-mediation criteria. "
            "A sliding-window-aware α–V decomposition supported terminal read/write, and a "
            "fresh-seed exact-block experiment confirmed serial mediation from the frozen "
            "L29H4/L35H2 broad set through L37. We therefore interpret Gemma as a tested-path "
            "replication of the distributed read/write mechanism, not a replication of Qwen head identity."
        )
    elif natural and read_write:
        claim = "Gemma 的 terminal natural transporter 与内部 read/write 已成立，但 frozen L29H4/L35H2 尚未被确认是该 transporter 的上游来源。"
        paper_sentence = (
            "Gemma4-E4B replicated the terminal natural-OV and sliding-window-aware read/write "
            "signatures at L37 H1/H2; however, the fresh-seed serial test did not confirm the "
            "frozen L29H4/L35H2 set as its upstream source."
        )
    elif natural:
        claim = "Gemma L37 的自然使用证据成立，但当前 α/V 分解或 downstream propagation 没有闭合，不能写成完整 terminal read/write chain。"
        paper_sentence = (
            "Gemma4-E4B showed natural causal use of the frozen L37 H1/H2 OV channel, but the "
            "factorized read/write or downstream-propagation criteria did not close; we therefore "
            "do not claim a full Gemma read/write-chain replication."
        )
    else:
        claim = "Gemma 尚未复制完整 natural-transporter 路径；positive representation、ablation 或 derivative effects 只能作为各自层级的局部证据。"
        paper_sentence = (
            "Gemma4-E4B retained prompt-side geometry, broad-bank ablation effects, and a late "
            "answer state, but the preregistered natural-transporter conjunction was not satisfied; "
            "we therefore restrict Gemma claims to the individually supported representational or functional links."
        )
    return f"""
<h3>11.3 Gemma 跨模型 synthesis：哪些 link 真正复制</h3>
{table(["link", "直接检验", "结果", "证据等级"], rows)}
<p>跨模型复制使用“同一因果问题、同一推断单位、同一 matched-control 逻辑”，而不是要求同层同头。Gemma 的 L37 滑窗还意味着 terminal read 的 source 是进入该层前已形成的可访问 state；因此即便整条路径通过，也应写成 broad-set/relay → terminal read/write，而不是 L37 直接从 10k-token prompt 原始 needles 做全局 QK。</p>
<div class="conclusion"><strong>跨模型结论</strong>{claim}</div>
<p>联合判断遵循预先写定的 interpretation matrix：natural OV、α/V read/write、fresh-seed early→L37 mediation 三项全部通过，才叫 full tested-path replication；任一环节失败，就只保留已单独通过的前缀或局部 link。没有把三个 p 值相乘，也没有在看到结果后更换 Gemma heads。</p>
<div class="paper-wording"><strong>Suggested cross-model wording.</strong> {html.escape(paper_sentence)}</div>
<div class="conclusion"><strong>论文写作边界</strong>这里建立的是一条冻结、受控、可复现的逻辑通路；它不要求否定所有其他 relay，也不支持“唯一 circuit”或“唯一单头 counter”。</div>
"""


def build_gemma_synthesis_ladder(
    *,
    l37: dict[str, Any],
    singles: dict[str, dict[str, Any]],
    read_writes: dict[str, dict[str, Any]],
    cross_layer: dict[str, Any] | None,
    residuals: dict[str, dict[str, Any]],
    story: dict[str, Any],
) -> str:
    rows: list[list[str]] = [
        [
            "Prompt running-index representation",
            "frozen-basis PCA / full-space cue-paired statistics",
            "有序 geometry 在 cue removal 后保留",
            '<span class="evidence descriptive">表征</span>',
        ],
        [
            "Frozen broad-retrieval function",
            "correct-only K=1/K=2 ablation vs 3 layer-matched controls",
            "fresh-seed clean-correct failure 与 ΔMAE 显著",
            '<span class="evidence functional">独立功能支持</span>',
        ],
        [
            "L37 H1/H2 localized natural OV",
            "四族 true-pre-O conjunction",
            f"global IUT p={fmt_p(l37['primary_decision']['global_intersection_union_p'])}",
            evidence_badge(
                bool(l37["primary_decision"]["full_natural_ov_transporter_support"]),
                "通过",
                "否定该候选",
            ),
        ],
    ]
    for name, document in singles.items():
        decision = document["primary_decision"]
        rows.append(
            [
                f"{ov_candidate_label(document)} localized natural OV",
                "independent-ablation-ranked single; four-family IUT",
                f"global IUT p={fmt_p(decision['global_intersection_union_p'])}",
                evidence_badge(
                    bool(decision["full_natural_ov_transporter_support"]),
                    "通过",
                    "否定该候选",
                ),
            ]
        )
        if name in read_writes:
            rw = read_writes[name]
            rows.append(
                [
                    f"{ov_candidate_label(document)} α/V read/write",
                    "crossed α/V + downstream trace",
                    f"classification={html.escape(str(rw['primary_decision']['read_mode']['classification']))}; serial={rw['primary_decision']['serial_read_write_supported']}",
                    evidence_badge(
                        bool(rw["primary_decision"]["serial_read_write_supported"]),
                        "机制分解支持",
                        "未完整支持",
                    ),
                ]
            )
    if cross_layer is not None:
        rows.append(
            [
                "L29H4+L35H2 cross-layer set",
                "joint natural OV + exact L29→L35 relay",
                f"OV p={fmt_p(cross_layer['primary_decision']['global_intersection_union_p'])}; relay p={fmt_p(cross_layer['relay_decision']['intersection_union_p'])}",
                evidence_badge(
                    bool(cross_layer["full_cross_layer_mechanism_support"]),
                    "通过",
                    "未闭合",
                ),
            ]
        )
    for residual_name, residual in residuals.items():
        rows.append(
            [
                f"{residual_variant_label(residual)}→L{int(residual['selected_mediator_layer'])} residual→L41",
                (
                    "clean necessity + "
                    if residual["config"].get("require_clean_necessity", False)
                    else ""
                )
                + "source patch + exact/count-axis mediation + terminal adoption",
                f"global IUT p={fmt_p(residual['primary_decision']['global_intersection_union_p'])}",
                evidence_badge(
                    bool(
                        residual["primary_decision"]["full_residual_count_path_support"]
                    ),
                    "通过",
                    "未闭合",
                ),
            ]
        )
    if story["kind"] == "single":
        rw = story.get("read_write")
        rw_clause = (
            "；factorized α/V 与 downstream propagation 也满足其 extension 判据"
            if rw is not None
            and bool(rw["primary_decision"]["serial_read_write_supported"])
            else "；α/V extension 不作为独立复制"
        )
        claim = (
            f"Gemma 的最强定位是 {story['label']} natural OV transporter"
            f"（global IUT p={fmt_p(story['global_p'])}）{rw_clause}。"
        )
        paper = (
            f"In Gemma4-E4B, the independently ranked {story['label']} candidate "
            "satisfied the frozen natural-signal, true pre-output sufficiency, "
            "centered-necessity, and donor-path mediation conjunction on held-out seeds."
        )
    elif story["kind"] == "cross_layer":
        claim = (
            f"Gemma 的最强定位是 {story['label']} 跨层 set：joint natural OV 与 "
            f"L29→L35 exact-block relay 同时通过（joint p={fmt_p(story['global_p'])}）。"
        )
        paper = (
            "In Gemma4-E4B, a frozen cross-layer L29H4/L35H2 set satisfied both "
            "the joint natural-OV conjunction and an exact-block L29-to-L35 relay test "
            "on held-out seeds."
        )
    elif story["kind"] == "residual":
        claim = (
            f"Gemma 未定位到通过全部门的单头/局部 OV set，但 {story['label']} 的 "
            f"分布式 residual relay 通过（global IUT p={fmt_p(story['global_p'])}）。"
        )
        paper = (
            "In Gemma4-E4B, localized natural-OV hypotheses did not satisfy their full "
            "conjunctions. A separately confirmed fallback nevertheless showed that the "
            f"frozen {story['label']} path causally wrote a count-aligned distributed "
            "residual state whose registered removal reduced donor-count transport to "
            "the terminal layer."
        )
    else:
        claim = (
            "Gemma 的 representation、answer-state patching 与 broad-bank necessity 成立，"
            "但当前冻结的 localized/cross-layer/residual conjunction 均未闭合。"
        )
        paper = (
            "Gemma4-E4B retained an ordered running-index representation, a causally "
            "effective answer state, and broad-bank ablation effects, but none of the "
            "frozen localized or distributed transport conjunctions closed; we therefore "
            "do not claim a complete Gemma head-level circuit."
        )
    return f"""
<h3>11.3 Gemma synthesis：由实际通过的最强门决定表述</h3>
{table(["link", "直接检验", "结果", "证据等级"], rows)}
<div class="conclusion"><strong>跨模型结论</strong>{html.escape(claim)}</div>
<p>跨模型复制指相同的计算问题与 matched-control 逻辑，不要求相同 layer/head identity。若 Gemma 最强层级是 residual，就只能主张“同构的分布式 state transport”，不能称为 Qwen L28 局部 OV circuit 的逐头复制。</p>
<div class="paper-wording"><strong>Suggested cross-model wording.</strong> {html.escape(paper)}</div>
<div class="conclusion"><strong>论文写作边界</strong>这里寻找并验证一条可复现的逻辑通路，不需要否定所有并行 relay；但任何失败的更强定位都必须与较弱正结果并列保留。</div>
"""


def build_correct_state_boundary(
    analysis: dict[str, Any], geometry_rows: list[dict[str, str]]
) -> str:
    if not analysis.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only state-route audit did not pass")
    geometry_table_rows: list[list[str]] = []
    location_labels = {
        "prompt_running_counter_source_bank_z": "prompt endpoint · frozen source-bank z",
        "answer_query_read_aggregate_source_bank_z": "answer query · source-bank aggregate z",
    }
    for row in sorted(
        geometry_rows,
        key=lambda item: (str(item["model_label"]), str(item["location"])),
    ):
        supported = str(row["geometry_supported_beyond_position"]).lower() == "true"
        geometry_table_rows.append(
            [
                html.escape(str(row["model_label"])),
                html.escape(location_labels.get(str(row["location"]), str(row["location"]))),
                fmt(float(row["oof_rounded_accuracy"]), 3),
                fmt_p(float(row["position_adjusted_iut_p"])),
                fmt_p(float(row["position_adjusted_iut_holm_p"])),
                evidence_badge(supported, "超过 position control", "未超过 position control"),
            ]
        )

    route_labels = {
        "answer_query_aggregate": "answer-query aggregate patch",
        "slot_endpoint_state": "single prompt endpoint patch",
    }
    route_table_rows: list[list[str]] = []
    for row in sorted(
        analysis["route_results"],
        key=lambda item: (str(item["model_label"]), str(item["route"])),
    ):
        route_table_rows.append(
            [
                html.escape(str(row["model_label"])),
                html.escape(route_labels.get(str(row["route"]), str(row["route"]))),
                f"{fmt(float(row['source_donor_log_odds_gain_mean']), 4, signed=True)} "
                f"[{fmt(float(row['source_donor_log_odds_gain_ci95_low']), 4)}, {fmt(float(row['source_donor_log_odds_gain_ci95_high']), 4)}]"
                f"; p={fmt_p(float(row['source_donor_log_odds_gain_p']))}",
                f"{fmt(float(row['writer_log_odds_mediation_specificity_mean']), 4, signed=True)} "
                f"[{fmt(float(row['writer_log_odds_mediation_specificity_ci95_low']), 4)}, {fmt(float(row['writer_log_odds_mediation_specificity_ci95_high']), 4)}]"
                f"; p={fmt_p(float(row['writer_log_odds_mediation_specificity_p']))}",
                evidence_badge(bool(row["route_supported"]), "完整 route 通过", "完整 route 未通过"),
            ]
        )

    return f"""
<h3>11.4 Correct-only low-count boundary：读到 state 不等于旧 writer set 完成写入</h3>
<p>这项补充实验只保留<strong>两个冻结模型在 clean forward 都正确回答 count 1–3</strong>的样本。20 个全新 seeds 内，每个 seed 平均六个有向 donor→receiver count pairs；推断单位仍是 seed。source patch 与 writer-specific block 在同一 forward family 内评估，所有 480 条 effect rows 的复现、norm、正交与 closure audits 均通过。</p>
{table(
    ["模型", "state location", "OOF rounded acc.", "position-adjusted IUT p", "Holm p", "判定"],
    geometry_table_rows,
)}
<p>answer-query source-bank aggregate 在两模型中都显著超过 cubic normalized-position control（Holm p=3.81×10<sup>−6</sup>）；Gemma 的 prompt source-bank z 也通过（Holm p=0.001427），而 Qwen 这一个<strong>特定 frozen bank 的 pre-O z</strong>未通过（Holm p=0.3768）。后者不否定前文在 full residual 上得到的 Qwen running-index geometry，因为 location、subspace 与 estimand 不同。</p>
{table(
    ["模型", "route", "source donor log-odds gain [95% CI]", "old writer-set mediation [95% CI]", "联合判定"],
    route_table_rows,
)}
<p>在 answer query 直接搬运 aggregate state 时，Qwen source gain 为 +0.6776 [0.4564, 0.8978]（p=1.43×10<sup>−5</sup>），Gemma 为 +12.1734 [11.3239, 12.9635]（p=9.54×10<sup>−7</sup>）；说明正确低-count 运行中确有可执行 readout state。但旧冻结 writer set 的 axis-specific mediation 不成立：Qwen −0.0176（p=0.7092），Gemma −0.0594（p=0.9899，且方向相反）。single prompt endpoint patch 对 Gemma 为严格 0，对 Qwen source gain 为负，因此也没有闭合 prompt-endpoint→writer route。</p>
<div class="conclusion"><strong>本小节结论</strong>这轮 correct-only 实验强化“两个模型都形成可执行 answer-query count state”，同时否定“旧低-count writer set 普遍介导该 state”的更强说法。它不推翻 Qwen 主实验中 H16/H19 的 all-count natural-OV conjunction，因为两者的冻结 set、被干预位置与 count regime 不同；更合适的解释是 Qwen 已有一个确认的 OV 通路，但其支配性可能随 regime/route 改变。对 Gemma，它进一步支持只写成 distributed effective residual write，而不指定 localized OV heads。</div>
"""


def build_limits_dynamic(
    *,
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_l37: dict[str, Any],
    gemma_singles: dict[str, dict[str, Any]],
    gemma_read_writes: dict[str, dict[str, Any]],
    gemma_cross_layer: dict[str, Any] | None,
    gemma_residuals: dict[str, dict[str, Any]],
    gemma_story: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    provenance_rows = [
        [
            "Representation + macro mechanism",
            "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
            "self-contained V4.4 interactive report",
        ],
        [
            "causal-v2",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
            f"schema {causal_v2['schema_version']}",
        ],
        [
            "correct-only seed extrapolation",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
            f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "Qwen natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
            f"schema {ov['schema_version']}",
        ],
        [
            "Qwen read/write",
            "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {read_write['schema_version']}",
        ],
        [
            "Qwen relay",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
            f"schema {relay['schema_version']}",
        ],
        [
            "Qwen upstream",
            "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {upstream['schema_version']}",
        ],
        [
            "Gemma L37 retained negative",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
            f"schema {gemma_l37['schema_version']}",
        ],
        [
            "correct-only low-count state routes",
            "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json",
            f"schema {correct_state['schema_version']} · 480/480 effect rows audited",
        ],
    ]
    for name, document in gemma_singles.items():
        provenance_rows.append(
            [
                f"Gemma {name} natural OV",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/search/{name}/realistic_niah_v4_4_4_analysis.json",
                f"schema {document['schema_version']}",
            ]
        )
    for name, document in gemma_read_writes.items():
        provenance_rows.append(
            [
                f"Gemma {name} read/write",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/search/{name}/realistic_niah_v4_4_4_read_write_analysis.json",
                f"schema {document['schema_version']}",
            ]
        )
    if gemma_cross_layer is not None:
        provenance_rows.append(
            [
                "Gemma cross-layer K2",
                "reports/v4_non-thinking_causal/v4_4_4/gemma/cross_layer/realistic_niah_v4_4_4_cross_layer_analysis.json",
                f"schema {gemma_cross_layer['schema_version']}",
            ]
        )
    for residual_name, gemma_residual in gemma_residuals.items():
        provenance_rows.append(
            [
                f"Gemma {residual_name.upper()} residual relay",
                f"reports/v4_non-thinking_causal/v4_4_4/gemma/residual/{residual_name}/realistic_niah_v4_4_4_residual_analysis.json",
                f"schema {gemma_residual['schema_version']}",
            ]
        )
    limit_rows = [
        [
            "跨模型身份",
            f"Gemma strongest layer={gemma_story['kind']}；Qwen/Gemma 使用不同 layer/head set",
            "只主张相同计算问题上的路径同构，不主张 head identity 或架构普适性",
        ],
        [
            "搜索树多重性",
            "各 Gemma 分支有独立 confirmation seeds 与内部 IUT；跨分支无单一全局 FWER",
            "写作中标明 sequential exploratory search + held-out confirmation，保留所有负分支",
        ],
        [
            "read/write 独立性",
            "factorized α/V extension 复用 parent candidate seeds",
            "只解释已冻结通道如何工作，不当作第二个独立 replication",
        ],
        [
            "并行机制",
            "验证一条 causal path 不穷尽网络中的其他 heads、MLPs 或 token relays",
            "不写唯一 circuit；负结果只否定对应冻结候选",
        ],
        [
            "PCA 推断",
            "三维仅显示冻结 basis 的前三方向；显著性来自 full-space / causal endpoints",
            "不从视觉间距直接推断 effect size 或 p 值",
        ],
    ]
    return f"""
<section id="limits">
<h2>12 · 证据边界与可复现性</h2>
{table(["边界", "当前事实", "写作约束"], limit_rows)}
<p>Gemma 证据阶梯的目的不是“试到显著为止”，而是把越来越宽松的机制定位逐层分开：localized head、cross-layer set、distributed residual。更弱分支通过不会抹去更强分支的失败，也不会获得更强分支的术语。</p>
<div class="conclusion"><strong>本段结论</strong>Qwen 可写成一条闭合的受限 OV mechanism；Gemma 只能写到 <code>{
        html.escape(str(gemma_story["kind"]))
    }</code>，其具体表述为：{html.escape(str(gemma_story["summary"]))}</div>
{
        details_table(
            "Source ledger",
            ["component", "relative path", "audit/schema note"],
            provenance_rows,
        )
    }
{
        details_table(
            "FileStream raw/derived roots",
            ["campaign", "FileStream path", "role"],
            [
                [
                    "V4.4 representation source",
                    "runs/run_20260731_v4_numeric_presentation_v3",
                    "raw generations/captures and frozen stimuli",
                ],
                [
                    "Qwen V4.4.4",
                    "runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1",
                    "natural OV / read-write / upstream derivatives",
                ],
                [
                    "correct-only frozen top-k",
                    "runs/run_20260804_v4_4_ablation_seed_extrapolation_qwen_n2_n4_gemma_n1_n2",
                    "Qwen/Gemma ranked-vs-random necessity",
                ],
                [
                    "Gemma retained L37",
                    "runs/v4_4_4_gemma_replication/v4_4_4_natural_ov/run_20260805_gemma_l37_h1_h2_frozen_v1",
                    "retained negative localized hypothesis",
                ],
                [
                    "Gemma evidence ladder",
                    "runs/v4_4_4_gemma_mechanism_search",
                    "single-head, cross-layer and residual gated branches",
                ],
                [
                    "correct-only low-count routes",
                    "runs/v4_4_4_correct_state_routes/run_20260805_dual_model_lowcount_correct",
                    "answer-query aggregate / prompt-endpoint patches and writer mediation",
                ],
            ],
        )
    }
<p class="small">报告只嵌入聚合统计与可视化，不复制 raw hidden states、full V tensors 或 raw attention rows；原始数据保留在 FileStream。builder 对每个实际存在的 analysis JSON 强制 audit PASS；causal-v2 与 correct-only audits 也必须通过。</p>
<div class="conclusion"><strong>最终结论</strong>non-thinking counting 最符合“分布式 running-index representation → broad retrieval → causal write/relay → 可执行 answer count state”。Qwen 已解析到 localized OV set；Gemma 的定位粒度严格由冻结证据阶梯的实际通过层级决定。</div>
</section>
"""


def build_limits_section(
    repo_root: Path,
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    relay: dict[str, Any],
    upstream: dict[str, Any],
    gemma_ov: dict[str, Any],
    gemma_read_write: dict[str, Any],
    gemma_upstream: dict[str, Any],
) -> str:
    gemma_full = (
        bool(gemma_ov["primary_decision"]["full_natural_ov_transporter_support"])
        and bool(gemma_read_write["primary_decision"]["serial_read_write_supported"])
        and bool(gemma_upstream["primary_decision"]["serial_chain_confirmed"])
    )
    provenance_rows = [
        [
            "Representation + macro mechanism",
            "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
            "self-contained V4.4 interactive report",
        ],
        [
            "causal-v2",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
            f"schema {causal_v2['schema_version']}",
        ],
        [
            "correct-only seed extrapolation",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
            f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS",
        ],
        [
            "20-seed exact sign-flip reanalysis",
            "reports/v4_non-thinking_causal/v4_4_causal_v2/exact_sign_flip_reanalysis.json",
            "full 2^20 enumeration; Holm across four frozen sets",
        ],
        [
            "natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
            f"schema {ov['schema_version']}",
        ],
        [
            "read/write",
            "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {read_write['schema_version']}",
        ],
        [
            "relay",
            "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
            f"schema {relay['schema_version']}",
        ],
        [
            "upstream confirmation",
            "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {upstream['schema_version']}",
        ],
        [
            "Gemma natural OV",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
            f"schema {gemma_ov['schema_version']}",
        ],
        [
            "Gemma read/write",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
            f"schema {gemma_read_write['schema_version']}",
        ],
        [
            "Gemma upstream confirmation",
            "reports/v4_non-thinking_causal/v4_4_4/gemma/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
            f"schema {gemma_upstream['schema_version']}",
        ],
    ]
    limit_rows = [
        [
            "跨模型身份",
            f"Gemma full tested-path replication={gemma_full}；无论联合结果如何，Qwen 与 Gemma 使用不同 layer/head set",
            "只主张同构计算路径，不主张相同 head identity 或架构普适性",
        ],
        [
            "Gemma L37 可见窗口",
            "L37 的 semantic source groups 与实际 capture key window 取交集；窗外组显式为 0",
            "把 L37 写成 terminal accessible-state reader，不写成直接回看原始 10k-token needles",
        ],
        [
            "early-set 因果范围",
            "frozen top-k clean-correct ablation确认 bank-level function；serial mediation确认 donor-induced path",
            "不把每个 early head 写成逐头 clean-run 必要，也不要求排除所有其他 relay",
        ],
    ]
    return f"""
<section id="limits">
<h2>12 · 证据边界与可复现性</h2>
{table(["边界", "当前事实", "如何处理"], limit_rows)}
<p>Gemma 等价实验与 correct-only frozen top-k ablation 已经并入，不再作为“待补实验”。tail-64 仍作为一个被严格否定的具体候选保留，但“穷尽并否定所有其他 relay”不属于当前论文主张的必要条件；我们的正面结论是一条冻结通路获得多层证据收敛，而不是全网络唯一性证明。</p>
<div class="conclusion"><strong>本段结论</strong>现有数据已经足以对 Qwen 给出闭合的受限机制链，并对 Gemma 给出由其三项联合判定所允许的最强跨模型结论。尚存边界主要是外部模型/架构普适性与未测试并行路径，而不是报告中仍缺一块已承诺的数据。</div>
{
        details_table(
            "Source ledger",
            ["component", "relative path", "audit/schema note"],
            provenance_rows,
        )
    }
{
        details_table(
            "FileStream raw/derived roots",
            ["campaign", "FileStream path", "role"],
            [
                [
                    "V4.4 representation source",
                    "runs/run_20260731_v4_numeric_presentation_v3",
                    "raw generations/captures and frozen stimuli",
                ],
                [
                    "Qwen V4.4.4 natural/read-write",
                    "runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1",
                    "Qwen natural-OV plus derivative analyses",
                ],
                [
                    "correct-only frozen top-k",
                    "runs/run_20260804_v4_4_ablation_seed_extrapolation_qwen_n2_n4_gemma_n1_n2",
                    "Qwen/Gemma clean baseline, ranked/random ablation and audit",
                ],
                [
                    "Gemma natural/read-write",
                    "runs/v4_4_4_gemma_replication/v4_4_4_natural_ov/run_20260805_gemma_l37_h1_h2_frozen_v1",
                    "Gemma natural-OV and derivative α/V read-write",
                ],
                [
                    "Gemma fresh serial path",
                    "runs/v4_4_4_gemma_replication/v4_4_4_serial_path/run_20260805_gemma_l29h4_l35h2_to_l37h1h2_fresh_v1",
                    "independent early→L37 mediation",
                ],
            ],
        )
    }
<p class="small">本报告只嵌入聚合统计与可视化，不复制 raw hidden states、full V tensors 或 raw attention rows；原始数据保留在 FileStream。Qwen/Gemma natural-OV、read/write 与 upstream audits 均须 PASS，builder 才会生成报告；Qwen relay audit 亦为 PASS。causal-v2 每模型 302/302 checks 通过，correct-only seed extrapolation audit 为 {
        seed_confirmation["audit"]["passed"]
    }/{seed_confirmation["audit"]["checks"]} PASS。</p>
<div class="conclusion"><strong>最终结论</strong>当前 non-thinking counting 最符合“分布式 running-index representation，经 broad retrieval 汇集，由 late set-level OV channel 写入可执行 answer count state”的机制。Qwen 的 early top-4→L28 H16–H19→answer 已独立复现；Gemma 的最强结论严格由 natural-OV、read/write 与 fresh-seed serial 三门联合结果决定。任何跨模型正结论都是 set-level 路径复制，而不是单头 counter 的证明。</div>
</section>
"""


EXTRA_CSS = r"""
.abstract{font-size:18px;line-height:1.72;max-width:96ch}.paper-table{font-size:13px}.paper-table td:first-child{font-weight:650;color:var(--indigo)}
.baseline-strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:24px 0}.baseline-strip>div{background:var(--surface);border:1px solid var(--line);padding:14px 16px}.baseline-strip span,.baseline-strip small{display:block;color:var(--muted);font-size:12px}.baseline-strip strong{display:block;font:700 22px/1.3 Consolas,monospace;color:var(--indigo);margin:4px 0}
.evidence{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700;white-space:nowrap}.descriptive{background:#E7E2F7;color:#34257B}.functional{background:#E3F4F7;color:#075C6E}.confirmed{background:#DDF3E8;color:#155C41}.supported{background:#F3EDCF;color:#685613}.rejected{background:#F6DCE8;color:#7D204D}
.prompt-block{white-space:pre-wrap;overflow:auto;background:#15112B;color:#F8F4EA;padding:18px 20px;border-radius:5px;border:1px solid #312B4A;max-width:94ch}.prompt-block code{background:transparent;color:inherit;padding:0}.evidence-note{border-left-color:var(--violet)}
.paper-wording{background:#E9E3D8;border:1px solid var(--line);padding:18px 20px;margin:22px 0;line-height:1.7}.paper-wording strong{color:var(--indigo)}
.integrated-forest text,.write-trace text,.gate-svg text,.relay-svg text,.mechanism-svg text{font-family:"Segoe UI",Arial,sans-serif}.integrated-forest .grid,.write-trace .grid{stroke:#D7D0C5;stroke-width:1}.integrated-forest .zero{stroke:#20242D;stroke-width:1.5}.integrated-forest .ci{stroke-width:3}.integrated-forest .cap{stroke-width:2}.integrated-forest .dot{stroke:#FFFDF8;stroke-width:2}.integrated-forest .tick,.write-trace .tick{fill:#69717B;font-size:12px}.integrated-forest .row-label{fill:#20242D;font-size:14px;font-weight:650}.integrated-forest .value-label,.write-trace .value-label{fill:#4D5560;font:12px Consolas,monospace}.integrated-forest .axis-label,.write-trace .axis-label{fill:#303744;font-size:13px;font-weight:650}
.write-trace .trace-line{fill:none;stroke:#6750E8;stroke-width:3}.write-trace .trace-ci{stroke:#6750E8;stroke-width:2}.write-trace .trace-dot{fill:#6750E8;stroke:#FFFDF8;stroke-width:2}
.gate-svg .gate-box{fill:#F8F5EE;stroke:#BDB4A7;stroke-width:1.5}.gate-svg .gate-box.gate-fail{fill:#FFF4F8;stroke:#C96B91}.gate-svg .gate-check{fill:#00A88F}.gate-svg .gate-check.gate-fail{fill:#B73B70}.gate-svg .gate-check-text{fill:white;font-weight:700;font-size:18px}.gate-svg .gate-heading{fill:#23165C;font-weight:700;font-size:18px}.gate-svg .gate-main{fill:#20242D;font:15px Consolas,monospace}.gate-svg .gate-sub{fill:#5E6672;font-size:13px}.gate-svg .gate-p{fill:#08705E;font:13px Consolas,monospace;font-weight:700}
.relay-svg .relay-box{stroke-width:2}.relay-svg .relay-pass{fill:#E4F4EC;stroke:#00A88F}.relay-svg .relay-fail{fill:#F8E6EE;stroke:#D94B86}.relay-svg .relay-mark{font-size:24px;font-weight:700;fill:#20242D}.relay-svg .relay-heading{font-size:13px;font-weight:700;fill:#23165C}.relay-svg .relay-value,.relay-svg .relay-p{font:12px Consolas,monospace;fill:#303744}.relay-svg .relay-arrow{stroke:#718096;stroke-width:2}.relay-svg .relay-summary{font:14px Consolas,monospace;fill:#7D204D;font-weight:700}
.mechanism-svg .mech-node{fill:#F8F5EE;stroke:#6750E8;stroke-width:2}.mechanism-svg .mech-2,.mechanism-svg .mech-3{fill:#E9E4FA}.mechanism-svg .mech-4{fill:#E4F4EC;stroke:#00A88F}.mechanism-svg .mech-heading{font-size:15px;font-weight:700;fill:#23165C}.mechanism-svg .mech-sub{font-size:12px;fill:#4F5863}.mechanism-svg .mech-arrow{stroke:#6750E8;stroke-width:3}.mechanism-svg .mech-evidence,.mechanism-svg .mech-boundary{font-size:12px;fill:#5E6672}.mechanism-svg .mech-negative{fill:#EFECE6;stroke:#718096;stroke-width:1.5}.mechanism-svg .mech-dashed{stroke:#718096;stroke-width:2;stroke-dasharray:7 6}
.mechanism-main{background:linear-gradient(145deg,#F4F0E8 0%,#ECE8F8 58%,#E5F3EE 100%);border:1px solid #CFC6BA;padding:28px;margin-top:24px}.main-figure-kicker{font:700 11px/1.4 Consolas,monospace;letter-spacing:.13em;color:#6750E8;margin-bottom:8px}.mechanism-walkthrough{margin-top:18px}.mechanism-canvas-wrap{background:#15112B;border:1px solid #302A49;overflow:auto}.mechanism-canvas-wrap svg{display:block;min-width:980px;width:100%;height:auto;color:#6750E8}.mechanism-canvas-wrap text{font-family:"Segoe UI",Arial,sans-serif}.walk-input rect,.walk-node rect{fill:#211B3D;stroke:#5D557B;stroke-width:2;transition:fill .28s ease,stroke .28s ease,filter .28s ease}.walk-input circle,.walk-node circle{fill:#6750E8;stroke:#F8F5EE;stroke-width:1.5}.walk-title{fill:#F6F2E8;font-size:15px;font-weight:700}.walk-token,.walk-head,.walk-formula{fill:#CFC8E7;font-size:13px}.walk-sub,.walk-boundary{fill:#9DA2B4;font-size:11px}.mini-manifold,.fan-line{fill:none;stroke:#70688E;stroke-width:3}.walk-edge{fill:none;stroke:#58516F;stroke-width:4;color:#58516F;transition:stroke .28s ease,color .28s ease}.walk-input.is-complete rect,.walk-node.is-complete rect{fill:#29224D;stroke:#8D7FF1}.walk-input.is-active rect,.walk-node.is-active rect{fill:#3B2E74;stroke:#D6B52C;filter:drop-shadow(0 0 12px rgba(214,181,44,.42))}.walk-node.is-active .mini-manifold,.walk-node.is-active .fan-line{stroke:#D6B52C}.walk-node.walk-output.is-active rect{fill:#124A42;stroke:#2DBE77}.walk-edge.is-complete{stroke:#6750E8;color:#6750E8}.walk-edge.is-active{stroke:#D6B52C;color:#D6B52C;stroke-dasharray:9 7;animation:walk-dash .8s linear infinite}.walk-answer{fill:#F8F5EE;font:700 18px Consolas,monospace}.walk-answer-number{fill:#D6B52C;font:800 36px Consolas,monospace}.mechanism-controls,.running-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:14px 0}.mechanism-controls button,.running-controls button,.step-dots button{border:1px solid #9188A6;background:#FFFDF8;color:#23165C;padding:8px 12px;border-radius:4px;font-weight:650;cursor:pointer}.mechanism-controls button:hover,.running-controls button:hover,.step-dots button:hover{background:#EEE9FA}.step-dots{display:flex;gap:5px;margin-left:auto}.step-dots button{width:34px;height:34px;padding:0;border-radius:50%}.step-dots button[aria-current="step"]{background:#6750E8;color:white;border-color:#6750E8}.mechanism-live{min-height:76px;background:#FFFDF8;border-left:4px solid #6750E8;padding:12px 16px;line-height:1.55}.mechanism-live strong{display:block;color:#23165C;margin-bottom:3px}.mechanism-live .live-evidence{font:12px Consolas,monospace;color:#08705E}
.running-index-block{margin-top:30px}.running-controls{background:#F4F0E8;border:1px solid #D7D0C5;padding:12px 14px}.running-controls label{display:flex;align-items:center;gap:8px;font-size:13px;color:#4F5863}.running-controls select{padding:7px 9px}.running-slider{flex:1;min-width:240px}.running-slider input{width:100%}.running-shell{height:560px}.running-shell canvas{width:100%;height:100%;display:block;touch-action:none}.running-status{font:13px Consolas,monospace;color:#4F5863;background:#F4F0E8;border:1px solid #D7D0C5;border-top:0;padding:9px 12px}
.plain-protocol{background:#F8F5EE;border:1px solid #CEC5B8;padding:16px 20px;margin:20px 0}.plain-protocol h4{margin:0 0 10px;color:#23165C}.plain-protocol li{margin:.55em 0;line-height:1.55}.sig-yes,.sig-no{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:750;white-space:nowrap}.sig-yes{background:#DDF3E8;color:#155C41}.sig-no{background:#F6DCE8;color:#7D204D}.test-card{border:1px solid #CFC7BA;background:#FFFDF8;padding:16px 18px;margin:16px 0}.test-card h4{margin:0 0 10px;color:#23165C}.test-card dl{display:grid;grid-template-columns:130px 1fr;gap:8px 14px;margin:0}.test-card dt{font-weight:700;color:#4E5661}.test-card dd{margin:0;line-height:1.55}.step-result{border-left:4px solid #6750E8;background:#F5F1FB;padding:12px 15px;margin:14px 0}.step-result strong{color:#23165C}.ablation-topk text{font-family:"Segoe UI",Arial,sans-serif}.ablation-topk .grid{stroke:#D7D0C5;stroke-width:1}.ablation-topk .x-guide{stroke:#D7D0C5;stroke-width:1;stroke-dasharray:4 5}.ablation-topk .axis{stroke:#2F3540;stroke-width:1.5}.ablation-topk .series-line{fill:none;stroke-width:3}.ablation-topk .series-dot{stroke:#FFFDF8;stroke-width:2}.ablation-topk .ci{stroke-width:2.5}.ablation-topk .cap{stroke-width:2}.ablation-topk .tick{fill:#68717C;font-size:12px}.ablation-topk .axis-label{fill:#303744;font-size:13px;font-weight:700}.ablation-topk .point-label{font-size:12px;font-weight:750}.ablation-topk .panel-title{fill:#23165C;font-size:15px;font-weight:750}.ablation-topk .legend-label{font-size:12px;font-weight:700}
@keyframes walk-dash{to{stroke-dashoffset:-32}}
@media(prefers-reduced-motion:reduce){.walk-input rect,.walk-node rect,.walk-edge{transition:none}.walk-edge.is-active{animation:none}}
@media(max-width:760px){.baseline-strip{grid-template-columns:1fr}.paper-table{min-width:760px}.integrated-forest,.write-trace,.gate-svg,.relay-svg,.mechanism-svg,.ablation-topk{min-width:760px}.figure-block,figure{overflow:auto}.mechanism-main{padding:18px}.step-dots{margin-left:0}.running-shell{height:460px}.test-card dl{grid-template-columns:1fr}.test-card dd{margin-bottom:8px}}
"""


EXTRA_JS = r"""
function makeMechanismWalkthrough(){
 const root=document.getElementById('mechanism-overview');if(!root)return;
 const nodes=[...root.querySelectorAll('[data-walk-step]')],edges=[...root.querySelectorAll('[data-walk-edge]')];
 const live=document.getElementById('mechanism-live'),prev=document.getElementById('mechanism-prev'),next=document.getElementById('mechanism-next'),play=document.getElementById('mechanism-play');
 const dots=[...root.querySelectorAll('[data-mechanism-step]')];
 const stages=[
  {title:'步骤 1/5 · 顺序读取重复 record',body:'模型以 non-thinking 模式读完整个 10k-token prompt。V4.4 随机化 needle 的位置、内容和顺序，因此后续稳定结构不能只由固定位置解释。',evidence:'设计事实；此步不单独进行显著性检验。'},
  {title:'步骤 2/5 · 形成 prompt running-index state',body:'每个 active needle 末端 residual 随 occurrence index n=1…10 沿有序 manifold 移动。删除开头定义提示后，相对拓扑仍高度保留，但 full-space state 会被调制。',evidence:'表征证据：PCA 只显示结构；显著性由 full-space tests 承担。'},
  {title:'步骤 3/5 · early broad bank 汇集 slot states',body:'冻结的 L23H28、L23H29、L26H20、L27H18 set-output donor patch 把 answer distribution 推向 donor count；随后阻断 L28 通道会特异削弱该效应。',evidence:'fresh seeds 1294–1313；serial-path conjunction IUT p=0.005884（显著）。'},
  {title:'步骤 4/5 · mixed read 经 OV 改换坐标并写回',body:'Qwen L28 H16/H19 的 pre-O z 同时依赖 routing α 与 value content V；W_O 把 head-space count state 写入 residual，所以 prompt count axis 与 answer count axis 不需要平行。H19 是测试 set 内非冗余锚点。',evidence:'Qwen routing/value p=9.54×10⁻⁷；natural-OV global IUT p=0.004541。Gemma localized OV 未闭合，只确认 distributed residual write。'},
  {title:'步骤 5/5 · late answer state 形成并驱动数字输出',body:'Qwen L28 的自然写入沿 L29–L35 的冻结 answer-count axes 保留；Gemma 的 K2 source bank 则写入 L37 count-aligned residual 并传到 L41。最终 Total: query residual 可搬运 donor prediction。',evidence:'Qwen L35 最大校正 p=2.29×10⁻⁵；Gemma residual-path IUT p=9.54×10⁻⁷；两模型 correct-only answer aggregate patch 均显著。'}
 ];
 let step=0,timer=null;
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function render(){
  nodes.forEach(node=>{const i=+node.dataset.walkStep;node.classList.toggle('is-active',i===step);node.classList.toggle('is-complete',i<step)});
  edges.forEach(edge=>{const i=+edge.dataset.walkEdge;edge.classList.toggle('is-active',i===step);edge.classList.toggle('is-complete',i<step)});
  dots.forEach(dot=>{const active=+dot.dataset.mechanismStep===step;dot.setAttribute('aria-current',active?'step':'false')});
  prev.disabled=step===0;next.disabled=step===stages.length-1;
  live.innerHTML=`<strong>${stages[step].title}</strong><span>${stages[step].body}</span><span class="live-evidence">${stages[step].evidence}</span>`;
 }
 prev.addEventListener('click',()=>{stop();step=Math.max(0,step-1);render()});
 next.addEventListener('click',()=>{stop();step=Math.min(stages.length-1,step+1);render()});
 dots.forEach(dot=>dot.addEventListener('click',()=>{stop();step=+dot.dataset.mechanismStep;render()}));
 play.addEventListener('click',()=>{
  if(timer){stop();return}if(step===stages.length-1)step=0;render();play.textContent='❚❚ 暂停';play.setAttribute('aria-pressed','true');
  timer=setInterval(()=>{if(step>=stages.length-1){stop();return}step+=1;render()},1200);
 });
 render();
}

function makeRunningIndex(){
 const canvas=document.getElementById('running-index-canvas');if(!canvas)return;
 const ctx=canvas.getContext('2d'),model=document.getElementById('running-model'),slider=document.getElementById('running-step');
 const prev=document.getElementById('running-prev'),next=document.getElementById('running-next'),play=document.getElementById('running-play'),status=document.getElementById('running-status');
 let step=1,yaw=-.72,pitch=.43,zoom=1,drag=false,lastX=0,lastY=0,timer=null;
 function active(){const options=Object.values(PROMPT_DATA).filter(item=>item.model===model.value);return options.find(item=>item.manifold_display)||options[Math.floor(options.length/2)]}
 function centroids(item){const out=[];for(let n=1;n<=10;n++){const rows=item.rows.filter(row=>row[5]===n);const p=[0,1,2].map(pc=>rows.reduce((sum,row)=>sum+row[6+pc],0)/rows.length);out.push({n,p,rows})}return out}
 function geometry(item,w,h){
  const values=[0,1,2].map(pc=>item.rows.map(row=>row[6+pc])),mins=values.map(v=>Math.min(...v)),maxs=values.map(v=>Math.max(...v));
  const center=mins.map((v,i)=>(v+maxs[i])/2),common=Math.max(...mins.map((v,i)=>maxs[i]-v),1e-8),radius=Math.min(w,h)*.34*zoom;
  const project=p=>{let x=(p[0]-center[0])*2/common,y=(p[1]-center[1])*2/common,z=(p[2]-center[2])*2/common;const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;return{x:w/2+x1*radius,y:h/2-y1*radius,z:z2}};
  return{project,center,common};
 }
 function line(a,b,stroke,width=1,dash=[]){ctx.beginPath();ctx.setLineDash(dash);ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke();ctx.setLineDash([])}
 function draw(){
  const rect=canvas.getBoundingClientRect(),w=rect.width,h=rect.height,item=active();ctx.clearRect(0,0,w,h);ctx.fillStyle='#15112B';ctx.fillRect(0,0,w,h);if(!item)return;
  const cs=centroids(item),g=geometry(item,w,h),project=g.project,half=g.common*.47;
  const axisColors=['#8D7FF1','#52C4B1','#D6B52C'];
  for(let axis=0;axis<3;axis++){const a=[...g.center],b=[...g.center];a[axis]-=half;b[axis]+=half;const qa=project(a),qb=project(b);line(qa,qb,axisColors[axis],1.2,[4,6]);ctx.fillStyle=axisColors[axis];ctx.font='12px Consolas';ctx.fillText(`PC${axis+1}`,qb.x+5,qb.y-4)}
  const qcs=cs.map(c=>({...c,q:project(c.p)}));
  for(let i=1;i<qcs.length;i++)line(qcs[i-1].q,qcs[i].q,'rgba(244,240,232,.18)',2,[5,7]);
  for(let i=1;i<step;i++)line(qcs[i-1].q,qcs[i].q,'rgba(255,253,248,.88)',4);
  const current=cs[step-1],cloud=current.rows.map(row=>({row,q:project([row[6],row[7],row[8]])})).sort((a,b)=>a.q.z-b.q.z);
  for(const point of cloud){ctx.globalAlpha=.28;ctx.fillStyle=COUNT_COLORS[step-1];ctx.beginPath();ctx.arc(point.q.x,point.q.y,3.2,0,Math.PI*2);ctx.fill()}
  ctx.globalAlpha=1;
  for(const c of qcs){const reached=c.n<=step;ctx.globalAlpha=reached?1:.23;ctx.fillStyle=COUNT_COLORS[c.n-1];ctx.strokeStyle=reached?'#FFFDF8':'#746D87';ctx.lineWidth=c.n===step?3:1.4;ctx.beginPath();ctx.arc(c.q.x,c.q.y,c.n===step?10:6.5,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle=reached?'#FFFDF8':'#9CA0AE';ctx.font=c.n===step?'700 13px Segoe UI':'11px Segoe UI';ctx.fillText(String(c.n),c.q.x+10,c.q.y-9)}
  ctx.globalAlpha=1;ctx.fillStyle='#F8F5EE';ctx.font='700 16px Segoe UI';ctx.fillText(`running index n=${step}`,18,28);ctx.fillStyle='#A8ACB8';ctx.font='12px Consolas';ctx.fillText(`${model.value} · display L${item.layer} · actual V4.4 states`,18,48);
  const seeds=new Set(current.rows.map(row=>row[0])).size;status.textContent=`n=${step}/10 · ${model.value} · prompt manifold-display L${item.layer} · ${seeds} seeds · basis=${item.fit_variant} ${item.fit_split}`;
 }
 function resize(){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);draw()}
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function setStep(value){step=Math.max(1,Math.min(10,value));slider.value=String(step);prev.disabled=step===1;next.disabled=step===10;draw()}
 prev.addEventListener('click',()=>{stop();setStep(step-1)});next.addEventListener('click',()=>{stop();setStep(step+1)});slider.addEventListener('input',()=>{stop();setStep(+slider.value)});
 model.addEventListener('change',()=>{stop();setStep(1)});
 play.addEventListener('click',()=>{if(timer){stop();return}if(step===10)setStep(1);play.textContent='❚❚ 暂停';play.setAttribute('aria-pressed','true');timer=setInterval(()=>{if(step===10){stop();return}setStep(step+1)},750)});
 canvas.addEventListener('pointerdown',event=>{drag=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId)});
 canvas.addEventListener('pointermove',event=>{if(!drag)return;yaw+=(event.clientX-lastX)*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(event.clientY-lastY)*.008));lastX=event.clientX;lastY=event.clientY;draw()});
 canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);
 canvas.addEventListener('wheel',event=>{event.preventDefault();zoom=Math.max(.55,Math.min(2.1,zoom*(event.deltaY>0?.92:1.08)));draw()},{passive:false});
 new ResizeObserver(resize).observe(canvas);setStep(1);
}
"""


CLEAR_CSS = r"""
.mechanism-clear{background:#F5F4EF;border-color:#C9C7BE;color:#252923}.mechanism-clear .main-figure-kicker{color:#27685F}.mechanism-clear h2{max-width:900px}.mechanism-clear-intro{max-width:980px;font-size:16px;line-height:1.7}.model-mechanism{margin:30px 0 38px;border-top:3px solid #27685F;padding-top:16px}.model-mechanism.gemma{border-top-color:#A66A45}.model-mechanism-header{display:grid;grid-template-columns:minmax(220px,.7fr) minmax(420px,1.8fr);gap:28px;align-items:start;margin-bottom:18px}.model-mechanism-header h3{margin:0;color:#202520;font-size:24px}.model-mechanism-header p{margin:0;line-height:1.65}.mechanism-step-list{list-style:none;margin:0;padding:0;border-top:1px solid #CBC9C0}.mechanism-step-list li{display:grid;grid-template-columns:52px minmax(180px,.65fr) minmax(340px,1.55fr) minmax(210px,.8fr);gap:18px;align-items:start;padding:18px 0;border-bottom:1px solid #D7D5CD}.mechanism-step-number{font:700 22px/1 Consolas,monospace;color:#27685F}.gemma .mechanism-step-number{color:#A66A45}.mechanism-step-title{font-weight:750;color:#202520}.mechanism-step-action,.mechanism-step-evidence{line-height:1.55}.mechanism-step-evidence{font:12px/1.55 Consolas,monospace;color:#3C625C}.mechanism-principle{border-top:1px solid #BDBBB2;padding-top:18px;margin-top:8px}.mechanism-principle .equation{margin:10px 0}.causal-roadmap{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0;border-top:1px solid #C8C6BD;border-bottom:1px solid #C8C6BD;margin:22px 0}.causal-roadmap>div{padding:16px 18px;border-right:1px solid #D5D3CB}.causal-roadmap>div:last-child{border-right:0}.causal-roadmap strong{display:block;margin-bottom:6px;color:#24302D}.causal-roadmap span{font-size:13px;line-height:1.5;color:#555E5A}.answer-patch-svg text{font-family:"Segoe UI",Arial,sans-serif}.answer-patch-svg .grid{stroke:#D7D5CD;stroke-width:1}.answer-patch-svg .axis{stroke:#343A37;stroke-width:1.5}.answer-patch-svg .tick{fill:#66706B;font-size:12px}.answer-patch-svg .panel-title{fill:#202520;font-size:15px;font-weight:700}.answer-patch-svg .axis-label{fill:#343A37;font-size:13px;font-weight:700}.answer-patch-svg .bar-label{fill:#202520;font:700 13px Consolas,monospace}.positive-mechanism-model{margin:26px 0 42px;border-top:3px solid #27685F;padding-top:16px}.positive-mechanism-model.gemma{border-top-color:#A66A45}.positive-mechanism-model h3{margin-top:0}.result-sentence{font-size:17px;line-height:1.7;max-width:980px}.compact-result-table td,.compact-result-table th{vertical-align:top}.scope-lines{border-top:1px solid #C8C6BD}.scope-line{display:grid;grid-template-columns:220px 1fr 190px;gap:20px;padding:15px 0;border-bottom:1px solid #D8D6CF}.scope-line strong{color:#202520}.scope-line .status{font:12px Consolas,monospace;color:#27685F}.provenance-note{font-size:11px;color:#727873;margin-top:24px}
@media(max-width:820px){.model-mechanism-header{grid-template-columns:1fr}.mechanism-step-list li{grid-template-columns:42px 1fr}.mechanism-step-action,.mechanism-step-evidence{grid-column:2}.causal-roadmap{grid-template-columns:1fr 1fr}.causal-roadmap>div:nth-child(2){border-right:0}.scope-line{grid-template-columns:1fr}.paper-table{min-width:720px}}
@media(max-width:520px){.causal-roadmap{grid-template-columns:1fr}.causal-roadmap>div{border-right:0;border-bottom:1px solid #D5D3CB}}
.mechanism-paper-figure{margin:26px 0 38px}.mechanism-paper-controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:0 0 12px}.mechanism-paper-controls button{border:1px solid #969A93;background:#FBFAF5;color:#252923;padding:7px 11px;border-radius:4px;cursor:pointer;font-weight:650}.mechanism-paper-controls button:disabled{opacity:.42;cursor:default}.mechanism-paper-dots{display:flex;gap:5px;margin-left:auto}.mechanism-paper-dots button{width:32px;height:32px;padding:0;border-radius:50%}.mechanism-paper-dots button[aria-current="step"]{background:#27685F;color:#FFF;border-color:#27685F}.mechanism-paper-svg{display:block;width:100%;height:auto;min-width:960px;background:#FBFAF5;border:1px solid #C9C7BE}.mechanism-paper-svg text{font-family:"Segoe UI",Arial,sans-serif}.mechanism-paper-svg .lane-label{font-size:20px;font-weight:750;fill:#202520}.mechanism-paper-svg .lane-sub{font-size:12px;fill:#66706B}.mechanism-paper-svg .paper-node rect{fill:#F0F1EC;stroke:#A9ADA5;stroke-width:1.5;transition:fill .24s ease,stroke .24s ease,filter .24s ease}.mechanism-paper-svg .paper-node .node-title{font-size:14px;font-weight:700;fill:#252923}.mechanism-paper-svg .paper-node .node-sub{font-size:11px;fill:#606862}.mechanism-paper-svg .paper-edge{stroke:#AEB2AB;stroke-width:2.5;fill:none;transition:stroke .24s ease,stroke-width .24s ease}.mechanism-paper-svg .paper-node.is-complete rect{fill:#E5F0EC;stroke:#6A958A}.mechanism-paper-svg .paper-edge.is-complete{stroke:#6A958A}.mechanism-paper-svg .gemma-node.is-complete rect{fill:#F3EAE3;stroke:#B78767}.mechanism-paper-svg .gemma-edge.is-complete{stroke:#B78767}.mechanism-paper-svg .paper-node.is-active rect{fill:#DCECE7;stroke:#27685F;stroke-width:3;filter:drop-shadow(0 3px 5px rgba(39,104,95,.2))}.mechanism-paper-svg .gemma-node.is-active rect{fill:#F3E2D7;stroke:#A66A45;filter:drop-shadow(0 3px 5px rgba(166,106,69,.2))}.mechanism-paper-svg .paper-edge.is-active{stroke:#27685F;stroke-width:4}.mechanism-paper-svg .gemma-edge.is-active{stroke:#A66A45}.mechanism-paper-svg .window-strip{font:11px Consolas,monospace;fill:#59615C}.mechanism-paper-svg .window-s{fill:#E7E7E1;stroke:#BFC1BA}.mechanism-paper-svg .window-f{fill:#D7E9E3;stroke:#579183}.mechanism-paper-svg .window-label{font:700 10px Consolas,monospace;fill:#303632}.mechanism-live-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;border:1px solid #C9C7BE;border-top:0}.mechanism-live-grid>div{padding:13px 16px;min-height:72px}.mechanism-live-grid>div+div{border-left:1px solid #D3D1C9}.mechanism-live-grid strong{display:block;margin-bottom:4px}.mechanism-live-grid span{font-size:13px;line-height:1.55;color:#4F5752}.mechanism-definitions{margin:30px 0}.mechanism-definitions h3{margin-bottom:8px}.mechanism-definition-grid{display:grid;grid-template-columns:190px 1fr;border-top:1px solid #C8C6BD}.mechanism-definition-grid>div{display:contents}.mechanism-definition-grid strong,.mechanism-definition-grid span{padding:13px 0;border-bottom:1px solid #D8D6CF;line-height:1.62}.mechanism-definition-grid strong{padding-right:22px;color:#24302D}.mechanism-definition-grid code{white-space:normal}.window-explainer{margin:20px 0 26px;padding:18px 0;border-top:2px solid #A66A45;border-bottom:1px solid #D1CEC5}.window-explainer h4{margin:0 0 10px;font-size:18px}.window-explainer-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:22px}.window-explainer-grid strong{display:block;margin-bottom:5px}.window-explainer-grid p{margin:0;line-height:1.62}.mechanism-step-action .formula-line{display:block;margin-top:6px;font:12px/1.55 Consolas,monospace;color:#38433E}.mechanism-index-note{font-size:12px;color:#68706B;margin-top:6px}
@media(max-width:820px){.mechanism-paper-figure{overflow:auto}.mechanism-live-grid{grid-template-columns:1fr}.mechanism-live-grid>div+div{border-left:0;border-top:1px solid #D3D1C9}.mechanism-definition-grid{grid-template-columns:1fr}.mechanism-definition-grid>div{display:block;border-bottom:1px solid #D8D6CF}.mechanism-definition-grid strong,.mechanism-definition-grid span{display:block;border-bottom:0;padding:8px 0}.window-explainer-grid{grid-template-columns:1fr}.mechanism-paper-dots{margin-left:0}}
"""


REPORT_REFINEMENT_CSS = r"""
.token-role-comparison{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;align-items:start;margin:22px 0}.token-role-comparison figure{min-width:0;margin:0}.token-role-comparison .stat-svg{display:block;width:100%;height:auto}
.study-preface{display:grid;grid-template-columns:150px 1fr;gap:10px 18px;margin:16px 0 22px;padding:16px 0;border-top:2px solid #7C8F88;border-bottom:1px solid #D1D2CC}.study-preface strong{color:#24302D}.study-preface span{line-height:1.65}
.noise-decomposition-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px 0}.noise-decomposition-grid>div{display:flex;flex-direction:column;gap:8px;padding:16px;border:1px solid #D7DEE8;border-top:4px solid #6750E8;border-radius:9px;background:#FFF;box-shadow:0 8px 22px rgba(35,22,92,.035)}.noise-decomposition-grid>div:nth-child(2){border-top-color:#00C2FF}.noise-decomposition-grid>div:nth-child(3){border-top-color:#FF5FA2}.noise-decomposition-grid strong{color:#23165C}.noise-decomposition-grid code{align-self:flex-start}.noise-decomposition-grid span{color:#566477;line-height:1.55}
.subspace-logic-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin:20px 0 28px}.subspace-logic-grid article{padding:20px;border:1px solid #D7DEE8;border-top:5px solid #FF5FA2;border-radius:10px;background:#FFF}.subspace-logic-grid article+article{border-top-color:#00D4B4}.subspace-logic-grid span{display:block;color:#7D204D;font-size:12px;font-weight:800;letter-spacing:.07em;text-transform:uppercase}.subspace-logic-grid article+article span{color:#087A67}.subspace-logic-grid strong{display:block;margin:7px 0 11px;color:#23165C;font-size:18px}.subspace-logic-grid p{max-width:none;margin:8px 0}.subspace-logic-grid b{color:#23165C}
.audit-group-title{margin-top:44px;padding-bottom:9px;border-bottom:2px solid #6750E8}.audit-question{margin:14px 0;border:1px solid #D7DEE8;border-left:5px solid #6750E8;border-radius:10px;background:#FFF;box-shadow:0 10px 28px rgba(35,22,92,.04);overflow:hidden}.audit-question:nth-of-type(3n+1){border-left-color:#00D4B4}.audit-question:nth-of-type(3n+2){border-left-color:#00C2FF}.audit-question-summary{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;padding:17px 20px;cursor:pointer;list-style:none}.audit-question-summary::-webkit-details-marker{display:none}.audit-question-summary::after{content:"＋";grid-column:4;color:#6750E8;font-size:19px;font-weight:800}.audit-question[open] .audit-question-summary::after{content:"−"}.audit-question[open] .audit-question-summary{background:#F8FBFF;border-bottom:1px solid #D7DEE8}.audit-question-title{font-family:Georgia,"Times New Roman",serif;color:#161923;font-size:18px;font-weight:700;line-height:1.35}.audit-question-body{padding:20px 22px 22px}.audit-index{display:inline-flex;align-items:center;justify-content:center;min-width:46px;padding:4px 9px;border-radius:999px;background:#23165C;color:#FFF;font-size:12px;font-weight:800;letter-spacing:.06em}.audit-location{color:#64748B;font-size:13px;white-space:nowrap}.audit-answer-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.audit-answer-grid>div{padding:14px;background:#F8FBFF;border:1px solid #E3E8F0;border-radius:7px}.audit-answer-grid>div:nth-child(2){background:#F3F0FF}.audit-answer-grid>div:nth-child(3){background:#EEFCF8}.audit-answer-grid strong{display:block;margin-bottom:6px;color:#23165C;font-size:13px;text-transform:uppercase;letter-spacing:.035em}.audit-answer-grid p{margin:0;max-width:none;line-height:1.58}.audit-boundary{max-width:none;margin:14px 2px 0;padding-top:12px;border-top:1px dashed #CBD5E1;color:#566477}.audit-boundary strong{color:#7D204D}
details.secondary-analysis,details.secondary-section{margin:14px 0 24px;border:1px solid #D7DEE8;border-radius:9px;background:#FFF;overflow:hidden}details.secondary-analysis>summary,details.secondary-section>summary{cursor:pointer;list-style:none;padding:13px 16px;color:#23165C;font-weight:750;background:#F6F4FF}details.secondary-analysis>summary::-webkit-details-marker,details.secondary-section>summary::-webkit-details-marker{display:none}details.secondary-analysis>summary::before,details.secondary-section>summary::before{content:"＋";display:inline-block;width:1.45em;color:#6750E8}details.secondary-analysis[open]>summary::before,details.secondary-section[open]>summary::before{content:"−"}.secondary-analysis-body,.secondary-section-body{padding:2px 18px 18px}
details.data-table{margin:16px 0;border:1px solid #D7DEE8;border-radius:8px;background:#FFF;overflow:hidden}details.data-table summary{cursor:pointer;list-style:none;padding:12px 15px;font-weight:700;color:#23165C;background:#F6F4FF}details.data-table summary::-webkit-details-marker{display:none}details.data-table summary::before{content:"＋";display:inline-block;width:1.4em;color:#6750E8}details.data-table[open] summary::before{content:"−"}details.data-table[open] summary{border-bottom:1px solid #D7DEE8}details.data-table .table-scroll{margin:0}
.mechanism-actor-table{margin:18px 0 24px}.mechanism-actor-table td:nth-child(1){font-weight:700;color:#24302D}.mechanism-actor-table td{line-height:1.55}
.causal-ledger{width:100%;min-width:1120px;table-layout:fixed}.causal-ledger th,.causal-ledger td{overflow-wrap:anywhere}.causal-ledger td{vertical-align:top;line-height:1.52}.causal-ledger td:nth-child(1){font-weight:700;color:#24302D}.causal-ledger code{white-space:normal}
.ov-short-flow{display:grid;grid-template-columns:1.15fr 34px 1.1fr 34px 1fr 34px 1.15fr;align-items:stretch;margin:20px 0 24px}.ov-short-flow .ov-box{padding:14px 15px;border-top:3px solid #27685F;border-bottom:1px solid #BFC5C0;background:#F4F5F1}.ov-short-flow .ov-box strong{display:block;margin-bottom:6px}.ov-short-flow .ov-box span{font-size:13px;line-height:1.5;color:#515A55}.ov-short-flow .ov-arrow{display:flex;align-items:center;justify-content:center;color:#27685F;font-size:24px;font-weight:700}.ov-operation-table td:nth-child(1){font-weight:700;white-space:nowrap}.ov-operation-table td{vertical-align:top;line-height:1.55}.ov-data-strip{margin:14px 0 20px;padding:12px 15px;border-left:4px solid #27685F;background:#F4F5F1;line-height:1.62}.ov-data-strip.gemma{border-left-color:#A66A45}
@media(max-width:980px){.ov-short-flow{grid-template-columns:1fr}.ov-short-flow .ov-arrow{height:30px;transform:rotate(90deg)}}
@media(max-width:820px){.token-role-comparison{grid-template-columns:1fr}}
@media(max-width:980px){.audit-answer-grid{grid-template-columns:1fr}.noise-decomposition-grid,.subspace-logic-grid{grid-template-columns:1fr}}
@media(max-width:820px){.study-preface{grid-template-columns:1fr}.mechanism-actor-table{min-width:980px}.causal-ledger{min-width:1120px}.audit-question-summary{grid-template-columns:auto 1fr}.audit-location{grid-column:2;white-space:normal}.audit-question-summary::after{grid-column:3;grid-row:1 / span 2}.audit-question-body{padding:16px}}
"""


AURORA_CSS = r"""
:root{--paper:#F8FBFF;--surface:#FFFFFF;--ink:#161923;--muted:#64748B;--line:#D7DEE8;--indigo:#23165C;--violet:#6750E8;--cyan:#00C2FF;--yellow:#F6E36A;--teal:#00D4B4;--green:#39E58C;--magenta:#C04DFF;--pink:#FF5FA2;--gray:#8190A5;--brown:#765347}
html{scroll-padding-top:64px}body{background:var(--paper);color:var(--ink);font-family:"Aptos","Segoe UI",Arial,sans-serif;font-size:16px;line-height:1.68}main{max-width:1240px;padding:44px 34px 110px}nav{background:rgba(248,251,255,.96);border-bottom:1px solid #D7DEE8;box-shadow:0 6px 20px rgba(35,22,92,.045)}nav a{color:#23165C;font-size:13px;letter-spacing:.01em}nav a:hover{color:#6750E8}header{max-width:1030px;padding:48px 0 34px;border-bottom:4px solid #23165C}h1,h2,h3,h4{font-family:Georgia,"Times New Roman",serif;color:#161923;letter-spacing:-.018em}h1{font-size:43px;line-height:1.1;max-width:940px}h2{font-size:31px;line-height:1.22;max-width:1020px}h3{font-size:22px;line-height:1.3;margin-top:36px}.eyebrow{color:#6750E8}.lead{color:#3F4A5A;max-width:82ch}.meta{color:#8190A5}section{padding:58px 0;border-bottom:1px solid #D7DEE8}p{max-width:92ch}.small,figcaption{color:#64748B}figure,.figure-block{background:#FFFFFF;border-color:#D7DEE8;border-radius:10px}figure{box-shadow:0 12px 32px rgba(35,22,92,.045)}figcaption{padding-top:4px;line-height:1.6}.callout,.conclusion{background:#FFFFFF;border-left-color:#00D4B4;box-shadow:0 8px 24px rgba(35,22,92,.035)}.conclusion strong:first-child{color:#23165C}.warning{border-left-color:#F6E36A}.study-preface{border-top-color:#6750E8;border-bottom-color:#D7DEE8}.study-preface strong{color:#23165C}.paper-table td:first-child,.paper-wording strong,.test-card h4,.plain-protocol h4{color:#23165C}details.data-table,.test-card,.plain-protocol,.paper-wording{background:#FFFFFF;border-color:#D7DEE8;border-radius:8px}details.data-table summary{color:#23165C}th{background:#EEF1FF;color:#23165C}th,td{border-bottom-color:#E3E8F0}tbody tr:hover{background:#F1FAFF}code{background:#EEF1FF;color:#23165C}.baseline-strip>div{background:#FFFFFF;border-color:#D7DEE8}.baseline-strip strong{color:#6750E8}.evidence.descriptive{background:#EEEAFE;color:#23165C}.evidence.functional{background:#E4F9FF;color:#07546E}.evidence.confirmed,.sig-yes{background:#DFFAF1;color:#075A48}.evidence.supported{background:#FFF9D6;color:#765A00}.evidence.rejected,.sig-no{background:#FFE7F1;color:#7D204D}.plot-shell,.mechanism-canvas-wrap{background:#23165C;border-color:#3D2E7A}.prompt-block{background:#23165C;border-color:#6750E8;color:#F8FBFF}.controls select,.controls button,.switcher button,.mechanism-paper-controls button,.mechanism-controls button,.running-controls button,.step-dots button{background:#FFFFFF;border-color:#B7C2D3;color:#23165C}.controls button:hover,.switcher button:hover,.mechanism-paper-controls button:hover,.mechanism-controls button:hover,.running-controls button:hover{background:#EEF1FF}.switcher button[aria-pressed="true"],.mechanism-paper-dots button[aria-current="step"],.step-dots button[aria-current="step"]{background:#6750E8;border-color:#6750E8;color:#FFFFFF}.mechanism-main,.mechanism-clear{background:#FFFFFF;border:1px solid #D7DEE8;border-top:5px solid #23165C;border-radius:12px;box-shadow:0 18px 44px rgba(35,22,92,.06)}.main-figure-kicker,.mechanism-clear .main-figure-kicker{color:#6750E8}.model-mechanism,.positive-mechanism-model{border-top-color:#6750E8}.model-mechanism.gemma,.positive-mechanism-model.gemma{border-top-color:#00D4B4}.mechanism-step-number,.scope-line .status{color:#6750E8}.gemma .mechanism-step-number{color:#00A98F}.mechanism-step-evidence{color:#087A67}.mechanism-paper-svg{background:#F8FBFF;border-color:#D7DEE8}.mechanism-paper-svg .paper-node rect{fill:#F3F0FF;stroke:#6750E8}.mechanism-paper-svg .paper-node.is-complete rect{fill:#E3FBF5;stroke:#00D4B4}.mechanism-paper-svg .paper-edge.is-complete{stroke:#00D4B4}.mechanism-paper-svg .paper-node.is-active rect{fill:#EAE5FF;stroke:#6750E8;filter:none}.mechanism-paper-svg .gemma-node.is-active rect{fill:#E1FCF6;stroke:#00D4B4;filter:none}.mechanism-paper-svg .paper-edge.is-active{stroke:#6750E8}.mechanism-paper-svg .gemma-edge.is-active{stroke:#00D4B4}.mechanism-live-grid{border-color:#D7DEE8}.mechanism-live-grid>div+div{border-color:#D7DEE8}.ov-short-flow .ov-box{border-top-color:#6750E8;background:#F8FBFF;border-bottom-color:#D7DEE8}.ov-short-flow .ov-arrow{color:#6750E8}.ov-data-strip{border-left-color:#6750E8;background:#F6F4FF}.ov-data-strip.gemma{border-left-color:#00D4B4;background:#EEFCF8}.step-result{border-left-color:#6750E8;background:#F3F0FF}.running-controls,.running-status{background:#F8FBFF;border-color:#D7DEE8}.paper-wording{background:#F6F4FF}.integrated-forest .grid,.write-trace .grid,.layer-curve-svg .grid{stroke:#E0E6EF}.integrated-forest .zero{stroke:#161923}.integrated-forest .dot{stroke:#FFFFFF}.integrated-forest .tick,.write-trace .tick,.layer-curve-svg .tick{fill:#64748B}.integrated-forest .row-label,.integrated-forest .axis-label,.write-trace .axis-label,.layer-curve-svg .axis-label,.layer-curve-svg .panel-title{fill:#161923}.write-trace .trace-line,.write-trace .trace-ci{stroke:#6750E8}.write-trace .trace-dot{fill:#6750E8;stroke:#FFFFFF}.gate-svg .gate-box{fill:#F8FBFF;stroke:#B9C5D5}.gate-svg .gate-heading{fill:#23165C}.gate-svg .gate-check{fill:#00D4B4}.relay-svg .relay-pass{fill:#E3FBF5;stroke:#00D4B4}.relay-svg .relay-fail{fill:#FFE7F1;stroke:#FF5FA2}.all-token-scatter rect{fill:#23165C}.provenance-note{color:#8190A5}
.gemma .mechanism-step-number{color:#00D4B4}.mechanism-step-evidence{color:#23165C}.plot-shell,.mechanism-canvas-wrap{border-color:#6750E8}.controls select,.controls button,.switcher button,.mechanism-paper-controls button,.mechanism-controls button,.running-controls button,.step-dots button{border-color:#8190A5}.evidence.functional,.evidence.confirmed,.evidence.supported,.evidence.rejected,.sig-yes,.sig-no{color:#23165C}.mechanism-paper-svg #mechanism-arrow-q path{fill:#6750E8}.mechanism-paper-svg #mechanism-arrow-g path{fill:#00D4B4}.mechanism-paper-svg .qwen-edge{stroke:#6750E8}.mechanism-paper-svg .gemma-edge{stroke:#00D4B4}.mechanism-paper-svg .window-s{fill:#F8FBFF;stroke:#8190A5}.mechanism-paper-svg .window-f{fill:#00D4B4;stroke:#23165C}.mechanism-paper-svg .window-label{fill:#161923}
@media(max-width:760px){main{padding:28px 16px 72px}header{padding-top:30px}h1{font-size:34px}h2{font-size:27px}section{padding:44px 0}}
"""


PAPER_TYPOGRAPHY_CSS = r"""
:root{--math-font:"Cambria Math","STIX Two Math","STIXGeneral","DejaVu Math TeX Gyre","Times New Roman",serif}
.math-inline,.equation,.formula-line{font-family:var(--math-font);font-weight:400;font-style:normal;letter-spacing:0;color:#111318;font-variant-numeric:lining-nums tabular-nums}
.math-inline{display:inline;padding:0 .06em;margin:0 .015em;background:transparent;border:0;border-radius:0;white-space:nowrap;font-size:1.045em}
.math-inline sub,.math-inline sup,.equation sub,.equation sup,.formula-line sub,.formula-line sup{font-size:.72em;line-height:0}
.equation,.formula-line{display:block;width:100%;max-width:100%;margin:14px auto 18px;padding:4px 18px;border:0;border-radius:0;background:transparent;text-align:center;font-size:1.08em;line-height:1.9;overflow-x:auto;overflow-y:hidden;white-space:nowrap}
.formula-line{margin:9px 0 12px}
.mechanism-step-action>.formula-line,.test-card dd>.formula-line{font-family:var(--math-font);font-size:1.035em;font-weight:400;font-style:normal;line-height:1.9;color:#111318;border-top:1px solid #E3E8F0;border-bottom:1px solid #E3E8F0;padding-top:8px;padding-bottom:8px}
.mechanism-definition-grid .math-inline{font-size:1.055em}
.mechanism-definition-grid code{font-family:"Aptos Mono","Cascadia Mono",Consolas,monospace;background:#F4F5FA;color:#31394A}
.model-mechanism-header{grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:36px;align-items:start;min-height:0}
.model-mechanism-header>h3,.model-mechanism-header>p{min-width:0}
.mechanism-clear>.conclusion{max-width:92ch}
@media(max-width:1180px){nav{position:relative;top:auto}html{scroll-padding-top:12px}}
@media(max-width:820px){.model-mechanism-header{grid-template-columns:1fr;gap:10px}.equation,.formula-line{text-align:left;padding-left:2px;padding-right:2px}}
@media print{nav{display:none}.equation,.formula-line{overflow:visible;white-space:normal;break-inside:avoid}.math-inline{color:#000}figure,.conclusion,.test-card{box-shadow:none}}
"""


def _route_row(
    analysis: dict[str, Any], model: str, route: str
) -> dict[str, Any]:
    hits = [
        row
        for row in analysis["route_results"]
        if row["model_label"] == model and row["route"] == route
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Missing route row for {model}/{route}")
    return hits[0]


def _fullspan_upstream_row(
    analysis: dict[str, Any], *, early_set: str, route: str
) -> dict[str, Any]:
    primary_late_set = str(analysis["config"]["primary_late_set"])
    hits = [
        row
        for row in analysis["summary"]
        if str(row["early_set"]) == early_set
        and str(row["route"]) == route
        and str(row["late_set"]) == primary_late_set
    ]
    if len(hits) != 1:
        raise RuntimeError(
            f"Missing full-span upstream row {early_set}/{route}/{primary_late_set}"
        )
    return hits[0]


def _head_labels(candidates: Sequence[Sequence[Any]], k: int) -> list[str]:
    return [
        f"L{int(item[0])}H{int(item[1])}" for item in list(candidates)[: int(k)]
    ]


def answer_patch_comparison_svg(causal_v2: dict[str, Any]) -> str:
    models = ["Qwen3-8B", "Gemma4-E4B"]
    colors = {"Qwen3-8B": "#6750E8", "Gemma4-E4B": "#00D4B4"}
    all_values = {
        model: float(
            causal_v2["primary_confirmation_family_summary"][
                f"{model}::answer_patching"
            ]["mean_effect"]
        )
        for model in models
    }
    correct_values = {
        model: float(
            causal_v2["correct_interventions"]["patch_pooled"][
                f"{model}::answer_patching"
            ]["pooled_average_patching_acc"]
        )
        for model in models
    }
    width, height = 1040, 430
    top, bottom = 70, 80
    panels = [
        (70, 430, "All samples", "control-adjusted donor transport", all_values),
        (610, 360, "Correct-only", "donor-target adoption rate", correct_values),
    ]
    parts = [
        f'<svg class="stat-svg answer-patch-svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="answer-patch-title answer-patch-desc">',
        '<title id="answer-patch-title">Answer-query patching under all-sample and correct-only estimands</title>',
        '<desc id="answer-patch-desc">The left panel shows mean control-adjusted donor transport for all samples. The right panel shows donor-target adoption probability when donor and receiver are both clean-correct.</desc>',
    ]
    for x0, panel_w, title, ylabel, values in panels:
        left, right = x0 + 62, x0 + panel_w - 18
        plot_h = height - top - bottom

        def y(value: float) -> float:
            return top + (1.0 - value) * plot_h

        parts.append(
            f'<text class="panel-title" x="{x0 + panel_w / 2:.1f}" y="30" text-anchor="middle">{html.escape(title)}</text>'
        )
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            yy = y(tick)
            parts.append(
                f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}"/>'
            )
            parts.append(
                f'<text class="tick" x="{left - 10}" y="{yy + 4:.1f}" text-anchor="end">{tick:.2f}</text>'
            )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{height-bottom}" x2="{right}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>'
        )
        centers = [left + (right-left)*0.32, left + (right-left)*0.72]
        bar_w = min(88.0, (right-left)*0.22)
        for model, xx in zip(models, centers, strict=True):
            value = values[model]
            yy = y(value)
            parts.append(
                f'<rect x="{xx-bar_w/2:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{height-bottom-yy:.1f}" rx="4" style="fill:{colors[model]}"/>'
            )
            parts.append(
                f'<text class="bar-label" x="{xx:.1f}" y="{yy-10:.1f}" text-anchor="middle">{value:.3f}</text>'
            )
            parts.append(
                f'<text class="tick" x="{xx:.1f}" y="{height-bottom+25}" text-anchor="middle">{html.escape(model)}</text>'
            )
        parts.append(
            f'<text class="axis-label" x="{(left+right)/2:.1f}" y="{height-20}" text-anchor="middle">{html.escape(ylabel)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_mechanism_overview_clear(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    causal_v2: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    q_patch = _route_row(correct_state, "Qwen3-8B", "answer_query_aggregate")
    g_patch = _route_row(correct_state, "Gemma4-E4B", "answer_query_aggregate")
    q_all = causal_v2["primary_confirmation_family_summary"]["Qwen3-8B::answer_patching"]
    g_all = causal_v2["primary_confirmation_family_summary"]["Gemma4-E4B::answer_patching"]
    q_correct = causal_v2["correct_interventions"]["patch_pooled"]["Qwen3-8B::answer_patching"]
    g_correct = causal_v2["correct_interventions"]["patch_pooled"]["Gemma4-E4B::answer_patching"]
    q_actor_rows = [
        ["Needle-end residual hᴾ(s,n,ℓ)", "第 n 个 active needle 的最后 token", "decoder block 把此前上下文与当前 record 更新进 residual；count step bᴾℓ 用 discovery OLS 提取", "分布在 prompt positions 的 running-index state", "frozen-basis geometry；不是单独的因果 head 定位"],
        ["Early bank: L23H28/L23H29/L26H20/L27H18", "注册 slot-query positions 上可访问的 prompt states", "各 head 计算 zₕ(q)=Σⱼαₕ(q,j)Vg(h)h(j)，再经自身 Wᴼʰ 写回；实验只确认 set-level source function，不虚构四个 head 的逐头分工", "送入 L28 之前的 donor-directed state change", f"fresh serial source+mediation IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}"],
        ["L28 H16/H19", "进入 L28 的可访问 state；α 选择 source，V 提供 content", "RR/RD/DR/DD 交叉替换后做 Shapley 分解；两部分形成 pre-O z，再由 ΣWᴼʰzₕ 写入 residual", "L28 post-attention count-related residual step", "routing 与 value families 均显著；natural-OV global IUT 显著"],
        ["H19 within the L28 set", "与 H16 共享的 L28 mediator input", "leave-one-out 删除单 head，测完整 set mediation 的下降", "set 内非冗余的主要锚点；不等同于 H19 单头充分", "H19 LOO decrement 显著；H16 更接近 companion/redundant role"],
        ["L29–L35 answer-query residual", "L28 Wᴼ 写回后的 residual state", "attention/MLP block 与 residual skip 共同施加下游 Jacobian Jℓ→A；沿每层冻结 bᴬℓ 测投影", "可由 LM head 读取的 terminal answer-count state", "natural−orthogonal propagation 在各层通过 Holm；answer patch 改变数字"],
    ]
    g_actor_rows = [
        ["Needle-end residual hᴾ(s,n,ℓ)", "第 n 个 active needle endpoint；S layer 仅见512-token窗口，F layer见完整 prefix", "局部更新、周期性 full-attention refresh 与 residual/MLP 变换共同形成有序 state", "prompt-side running-index representation", "frozen-basis geometry；不把 state 简化为窗口内单一整数 token"],
        ["L29H4 (full-attention)", "answer query 可访问的完整 causal prefix", "z₂₉,₄=Σⱼα₂₉,₄(q,j)Vg(h)h(j)，再由自身 Wᴼ²⁹,⁴ 写回", "第一次冻结 source-bank write；与 L35H2 联合作为 K2 source", "full-span rank #1；K1/K2 matched-control ablation 与 K2 path patch"],
        ["L35H2 (full-attention)", "经过 L29–L34 处理后的全 prefix state", "z₃₅,₂=ΣⱼαV 汇集，再经自身 Wᴼ³⁵,² 写回；实验确认 K2 联合路径，不声称它独自完成全部计数", "进入 L36–L40 sliding stack 前的全局刷新", "full-span rank #2；与 L29H4 共同构成确认 source bank"],
        ["L37 answer-query residual mediator", "K2 source patch 在前两层 Wᴼ 写回后诱发的 δ₃₇", "exact block 删除完整 δ₃₇；count-axis block 删除 projᵦ₃₇(δ₃₇)，均与等范数正交删除比较", "窗口内继续传播的分布式、部分 count-aligned state", "exact 与 count-axis mediation families 均通过 IUT"],
        ["L41 terminal full-attention state", "L37 经 L38–L40 residual/局部 blocks 传来的 state，并可再次访问完整 prefix", "测 Δh₄₁ 在冻结 b₄₁ 上的 donor-count adoption，再由 LM head 生成数字", "可执行 answer count distribution", "terminal adoption 与独立 answer-state patch 均显著"],
    ]
    return f"""
<section id="mechanism-overview" class="mechanism-main mechanism-clear">
<div class="main-figure-kicker">NON-THINKING COUNTING · MODEL-SPECIFIC MECHANISMS</div>
<h2>模型怎样把 prompt 中的累计状态变成最终数字？</h2>
<p class="mechanism-clear-intro">两模型共享同一个计算问题：先在 prompt 中形成随已读取 occurrence 数量变化的 state，再把这个 state 汇集到 answer query。区别在于目前的因果定位粒度：Qwen 已定位到具体 L28 OV head set；Gemma 已定位到 broad head bank 写入 L37 residual 的分布式路径。下面每一行依次写明“模型在做什么”以及“哪项实验支持这一步”。</p>

<article class="model-mechanism qwen">
<div class="model-mechanism-header"><h3>Qwen3-8B</h3><p><strong>一句话机制：</strong>prompt running counter → early broad retrieval → L28 H16/H19 读取 → W<sub>O</sub> 改换坐标并写回 → L29–L35 answer state → 输出 <code>Total:N</code>。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">形成 running counter</span><span class="mechanism-step-action">每读完一个 active needle，needle-end residual 就更新一次；第 n 个 endpoint 的 state 编码“已经读到第 n 个 occurrence”。</span><span class="mechanism-step-evidence">frozen-basis prompt geometry · n=1…10 ordered trajectory</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">汇集多个 slot state</span><span class="mechanism-step-action">L23H28、L23H29、L26H20、L27H18 组成 early broad bank，把分散在 prompt positions 的累计信息送向后面的 answer computation。</span><span class="mechanism-step-evidence">fresh-seed serial IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">L28 读取 state</span><span class="mechanism-step-action">H16/H19 在 answer query 处形成 pre-O head state <code>z</code>。attention routing α 决定从哪些可访问位置取信息，V content 携带被取出的 count-related content。</span><span class="mechanism-step-evidence">routing p={fmt_p(read_write['primary_decision']['read_mode']['routing_family_p'])} · value p={fmt_p(read_write['primary_decision']['read_mode']['value_family_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">OV 写成 answer direction</span><span class="mechanism-step-action"><code>w=ΣW<sub>O</sub><sup>h</sup>z<sub>h</sub></code> 把 head-space state 写回 residual。这里发生坐标变换，所以 prompt counter direction 与 answer counter direction 不必平行。</span><span class="mechanism-step-evidence">natural OV global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">形成并读取 answer state</span><span class="mechanism-step-action">写入后的 count component 沿 L29–L35 保留；最终 <code>Total:</code> query residual 驱动数字 token。把 donor answer state patch 给 receiver，会把输出推向 donor count。</span><span class="mechanism-step-evidence">all-sample transport={q_all['mean_effect']:.3f} · correct-only adoption={100*q_correct['pooled_average_patching_acc']:.1f}% · fresh low-count p={fmt_p(q_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<article class="model-mechanism gemma">
<div class="model-mechanism-header"><h3>Gemma4-E4B</h3><p><strong>一句话机制：</strong>prompt running counter → L29H4/L35H2 broad bank → 写入 L37 count-aligned residual → residual 传播至 L41 → answer query 输出 <code>Total:N</code>。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">形成 running counter</span><span class="mechanism-step-action">与 Qwen 相同，Gemma 在每个 active needle endpoint 更新一个有序 prompt-side state；它表示已读取 occurrence 的累计进度。</span><span class="mechanism-step-evidence">frozen-basis prompt geometry · n=1…10 ordered trajectory</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">broad bank 汇集可访问 state</span><span class="mechanism-step-action">L29H4 与 L35H2 组成冻结 K2 bank。由于 sliding/local attention，后层不必直接看到原始远端 needles；bank 读取的是进入当前窗口前已经形成的可访问 state。</span><span class="mechanism-step-evidence">fresh top-k ablation · K1/K2 clean-correct Holm-significant</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">写入 L37 residual</span><span class="mechanism-step-action">patch K2 source bank 会在 L37 产生 count-aligned residual change；精确删除这部分 change 会特异削弱 donor-count transport。</span><span class="mechanism-step-evidence">source + exact/count-axis mediation · global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">传播到 terminal layer</span><span class="mechanism-step-action">L37 的分布式 count state 沿 residual stream 传播至 L41，并提高 donor count 在 terminal answer distribution 中的采用程度。</span><span class="mechanism-step-evidence">terminal adoption p={fmt_p(gemma_residual['primary_decision']['families']['terminal_count_adoption']['intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">answer query 读出数字</span><span class="mechanism-step-action">最终 answer-query state 已包含可执行的 count prediction；把 donor aggregate state 搬到 receiver，会显著推动 receiver 采用 donor count。</span><span class="mechanism-step-evidence">all-sample transport={g_all['mean_effect']:.3f} · correct-only adoption={100*g_correct['pooled_average_patching_acc']:.1f}% · fresh low-count p={fmt_p(g_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<div class="mechanism-principle"><strong>两模型共享的表示原则。</strong><div class="equation">prompt count direction u<sub>P</sub> → head state z → residual write w=ΣW<sub>O</sub>z → downstream answer direction u<sub>A</sub>∝Jw</div><p>语义上保持的是 count ordering 与 causal transport，不是同一欧氏方向。因此 3D PCA 中 prompt counter 与 answer counter 可以旋转、缩放或压缩，而仍然实现同一个计数变量。</p></div>
</section>
    """


def build_mechanism_overview_detailed(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    causal_v2: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    q_patch = _route_row(correct_state, "Qwen3-8B", "answer_query_aggregate")
    g_patch = _route_row(correct_state, "Gemma4-E4B", "answer_query_aggregate")
    q_all = causal_v2["primary_confirmation_family_summary"]["Qwen3-8B::answer_patching"]
    g_all = causal_v2["primary_confirmation_family_summary"]["Gemma4-E4B::answer_patching"]
    q_correct = causal_v2["correct_interventions"]["patch_pooled"]["Qwen3-8B::answer_patching"]
    g_correct = causal_v2["correct_interventions"]["patch_pooled"]["Gemma4-E4B::answer_patching"]
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    q_fs_top4_heads = _head_labels(
        fullspan_upstream["config"]["early_candidates"], 4
    )
    q_fs_top4_text = ", ".join(q_fs_top4_heads)
    q_actor_rows = [
        ["Needle-end residual hᴾ(s,n,ℓ)", "同一 N=10 prompt 中第 n 个 active needle 的最后 token", "decoder block 把此前 causal prefix 与当前 record 更新进 residual；在 discovery seeds 上逐维 OLS 得到单位 count step bᴾℓ", "分布在 prompt positions 的 running-index state", "冻结 basis 的 ordered geometry；这里只定位 state，不把它虚构成某个单头"],
        [f"Full-span early top-4：{q_fs_top4_text}", "注册的 slot-state positions 上已经形成的 prompt-side states", "每个 head 先算 zₕ(q)=Σⱼαₕ(q,j)Vg(h)h(j)，再由自身 Wᴼʰ 写回；把 donor pre-O z patch 到 receiver 后，测 donor log-odds gain，并在 L28 H16/H19 精确阻断 induced natural-OV component", "送入 L28 之前的 donor-directed state change", f"10 seeds、6个K×3 routes 内 Holm；top-4 source与mediation均p={fmt_p(q_fs_top4_p)}。这是set-level证据，不臆造四个head的逐头专职"],
        ["L28 H16/H19", "进入 L28 的可访问 state；α 选择 source positions，V 提供被读取的 content", "构造 RR/RD/DR/DD 四个 α×V 组合并做 Shapley 分账，形成 pre-O z；随后计算 ΣₕWᴼʰzₕ", "L28 post-attention residual 中的 count-related write", "routing 与 value families 均显著；natural-OV global IUT 显著"],
        ["H19（H16/H19 set 内）", "与 H16 共用的 L28 mediator input", "leave-one-out 删除单 head，测完整 set mediation 相对下降", "set 内非冗余的主要锚点；不等同于 H19 单头充分", "H19 LOO decrement 显著；H16 是联合路径成员，但单独贡献更接近 companion/redundant role"],
        ["L29–L35 answer-query residual", "L28 Wᴼ 写回后的 residual state", "residual skip、后续 attention 与 MLP 共同施加局部 Jacobian Jℓ→A；每层用冻结 bᴬℓ 测差分投影", "可被 LM head 读出的 terminal answer-count state", "natural-vs-orthogonal propagation 通过校正；answer-state patch 改变数字输出"],
    ]
    g_actor_rows = [
        ["Needle-end residual hᴾ(s,n,ℓ)", "第 n 个 active needle endpoint；S layer 只见512-token窗口，F layer可见完整 prefix", "局部更新、周期性 full-attention refresh 与 residual/MLP 变换共同形成有序 state", "prompt-side running-index representation", "冻结 basis 的 ordered geometry；不把窗口内任意单 token 等同于完整整数寄存器"],
        ["L29H4（full-attention）", "answer query 可访问的完整 causal prefix", "z₂₉,₄=Σⱼα₂₉,₄(q,j)Vg(h)h(j)，再由该 head 自身 Wᴼ²⁹,⁴ 写回", "第一次冻结 source-bank write；与 L35H2 联合形成 K2 source", "full-span rank #1；K1 all-sample matched-control ablation 与 K2 path experiment"],
        ["L35H2（full-attention）", "经 L29–L34 更新后的全-prefix state", "z₃₅,₂=Σⱼα₃₅,₂(q,j)Vg(h)h(j)，再经自身 Wᴼ³⁵,² 写回；实验确认 K2 联合路径，不声称它独自完成全部计数", "进入 L36–L40 sliding stack 前的全局 refresh", "full-span rank #2；与 L29H4 共同构成确认 source bank"],
        ["L37 answer-query residual mediator", "K2 source patch 经两组 Wᴼ 写回后诱发的 δ₃₇", "exact block 删除完整 δ₃₇；count-axis block 删除 projᵦ₃₇(δ₃₇)，并与等范数正交删除比较", "窗口内继续传播的分布式、部分 count-aligned state", "exact 与 count-axis mediation families 均通过 IUT"],
        ["L41 terminal full-attention state", "L37 经 L38–L40 residual/局部 blocks 传来的 state，并可在 L41 再访问完整 prefix", "计算 Δh₄₁ 在冻结 b₄₁ 上的 donor-count adoption，随后由 LM head 映射到数字 logits", "可执行 answer count distribution", "terminal adoption 与独立 answer-state patch 均显著"],
    ]
    return f"""
<section id="mechanism-overview" class="mechanism-main mechanism-clear">
<div class="main-figure-kicker">NON-THINKING COUNTING · MODEL-SPECIFIC MECHANISMS</div>
<h2>模型如何从分散的 needle 形成、读取并写出 count state？</h2>
<p class="mechanism-clear-intro">这里把“mechanism”拆成四件可测量的事：状态在什么 token/layer 提取、count direction 怎样拟合、attention head 怎样读取并经 <em>W</em><sub>O</sub> 写回、以及替换或阻断该状态是否改变数字输出。Qwen 与 Gemma 的数学对象相同，但 Gemma 的 512-token sliding window 与周期性 full-attention layers 使信息流具有明显的“全局刷新—局部传递”节奏。</p>

<figure class="mechanism-paper-figure" id="mechanism-paper-figure" aria-labelledby="mechanism-paper-caption">
<div class="mechanism-paper-controls" aria-label="Mechanism step controls">
<button type="button" id="mechanism-prev">← 上一步</button>
<button type="button" id="mechanism-play" aria-pressed="false">▶ 播放一次</button>
<button type="button" id="mechanism-next">下一步 →</button>
<div class="mechanism-paper-dots" role="group" aria-label="直接选择机制步骤">
<button type="button" data-mechanism-step="0" aria-label="步骤 1">1</button><button type="button" data-mechanism-step="1" aria-label="步骤 2">2</button><button type="button" data-mechanism-step="2" aria-label="步骤 3">3</button><button type="button" data-mechanism-step="3" aria-label="步骤 4">4</button><button type="button" data-mechanism-step="4" aria-label="步骤 5">5</button>
</div></div>
<svg class="mechanism-paper-svg" viewBox="0 0 1220 610" role="img" aria-labelledby="mechanism-svg-title mechanism-svg-desc">
<title id="mechanism-svg-title">Qwen and Gemma non-thinking counting mechanisms</title>
<desc id="mechanism-svg-desc">Two aligned lanes show prompt running-index extraction, source-head retrieval, answer-query pre-output states, residual writing and propagation, and numerical output. Gemma additionally shows periodic full-attention layers separated by 512-token sliding-attention layers.</desc>
<defs><marker id="mechanism-arrow-q" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#6750E8"/></marker><marker id="mechanism-arrow-g" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><path d="M0,0 L9,3.5 L0,7 Z" fill="#00D4B4"/></marker></defs>
<text class="lane-label" x="22" y="48">Qwen3-8B</text><text class="lane-sub" x="22" y="68">all layers can address the full causal prefix</text>
<g class="paper-node qwen-node" data-mechanism-stage="0"><rect x="22" y="92" width="190" height="112" rx="8"/><text class="node-title" x="117" y="122" text-anchor="middle">Prompt running index</text><text class="node-sub" x="117" y="148" text-anchor="middle">needle-end hᴾ(s,n,ℓ)</text><text class="node-sub" x="117" y="169" text-anchor="middle">n = 1 … 10</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="1" d="M216 148 L260 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="1"><rect x="268" y="92" width="190" height="112" rx="8"/><text class="node-title" x="363" y="119" text-anchor="middle">Full-span early top-4</text><text class="node-sub" x="363" y="145" text-anchor="middle">L27H18 · L23H29</text><text class="node-sub" x="363" y="166" text-anchor="middle">L23H13 · L23H28</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="2" d="M462 148 L506 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="2"><rect x="514" y="92" width="190" height="112" rx="8"/><text class="node-title" x="609" y="119" text-anchor="middle">L28 H16/H19 read</text><text class="node-sub" x="609" y="145" text-anchor="middle">z = Σⱼ α(q,j)V(j)</text><text class="node-sub" x="609" y="166" text-anchor="middle">routing + value content</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="3" d="M708 148 L752 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="3"><rect x="760" y="92" width="190" height="112" rx="8"/><text class="node-title" x="855" y="119" text-anchor="middle">OV residual write</text><text class="node-sub" x="855" y="145" text-anchor="middle">w = Σₕ Wᴼʰ zₕ</text><text class="node-sub" x="855" y="166" text-anchor="middle">propagate through L29–L35</text></g>
<path class="paper-edge qwen-edge" data-mechanism-edge="4" d="M954 148 L998 148" marker-end="url(#mechanism-arrow-q)"/>
<g class="paper-node qwen-node" data-mechanism-stage="4"><rect x="1006" y="92" width="190" height="112" rx="8"/><text class="node-title" x="1101" y="122" text-anchor="middle">Executable answer state</text><text class="node-sub" x="1101" y="148" text-anchor="middle">Total: query → LM head</text><text class="node-sub" x="1101" y="169" text-anchor="middle">greedy digit N</text></g>

<text class="lane-label" x="22" y="272">Gemma4-E4B</text><text class="lane-sub" x="22" y="292">five sliding layers, then one full-attention layer; window W = 512</text>
<g class="paper-node gemma-node" data-mechanism-stage="0"><rect x="22" y="316" width="190" height="112" rx="8"/><text class="node-title" x="117" y="346" text-anchor="middle">Prompt running index</text><text class="node-sub" x="117" y="372" text-anchor="middle">needle-end hᴾ(s,n,ℓ)</text><text class="node-sub" x="117" y="393" text-anchor="middle">local state + periodic refresh</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="1" d="M216 372 L260 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="1"><rect x="268" y="316" width="190" height="112" rx="8"/><text class="node-title" x="363" y="343" text-anchor="middle">Global K2 source bank</text><text class="node-sub" x="363" y="369" text-anchor="middle">L29H4 · L35H2</text><text class="node-sub" x="363" y="390" text-anchor="middle">both full-attention layers</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="2" d="M462 372 L506 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="2"><rect x="514" y="316" width="190" height="112" rx="8"/><text class="node-title" x="609" y="343" text-anchor="middle">Donor pre-O z patch</text><text class="node-sub" x="609" y="369" text-anchor="middle">replace z₍₂₉,₄₎ and z₍₃₅,₂₎</text><text class="node-sub" x="609" y="390" text-anchor="middle">model Wᴼ performs the write</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="3" d="M708 372 L752 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="3"><rect x="760" y="316" width="190" height="112" rx="8"/><text class="node-title" x="855" y="343" text-anchor="middle">Distributed residual path</text><text class="node-sub" x="855" y="369" text-anchor="middle">L37 count-aligned mediator</text><text class="node-sub" x="855" y="390" text-anchor="middle">residual carry → L41</text></g>
<path class="paper-edge gemma-edge" data-mechanism-edge="4" d="M954 372 L998 372" marker-end="url(#mechanism-arrow-g)"/>
<g class="paper-node gemma-node" data-mechanism-stage="4"><rect x="1006" y="316" width="190" height="112" rx="8"/><text class="node-title" x="1101" y="346" text-anchor="middle">Executable answer state</text><text class="node-sub" x="1101" y="372" text-anchor="middle">L41 query → LM head</text><text class="node-sub" x="1101" y="393" text-anchor="middle">greedy digit N</text></g>

<text class="window-strip" x="22" y="476">Gemma layer schedule near the causal path:</text>
<g transform="translate(268 452)"><rect class="window-s" x="0" y="0" width="50" height="34"/><text class="window-label" x="25" y="22" text-anchor="middle">S28</text><rect class="window-f" x="54" y="0" width="50" height="34"/><text class="window-label" x="79" y="22" text-anchor="middle">F29</text><rect class="window-s" x="108" y="0" width="50" height="34"/><text class="window-label" x="133" y="22" text-anchor="middle">S30</text><rect class="window-s" x="162" y="0" width="50" height="34"/><text class="window-label" x="187" y="22" text-anchor="middle">S31</text><text class="window-label" x="236" y="22" text-anchor="middle">…</text><rect class="window-f" x="264" y="0" width="50" height="34"/><text class="window-label" x="289" y="22" text-anchor="middle">F35</text><rect class="window-s" x="318" y="0" width="50" height="34"/><text class="window-label" x="343" y="22" text-anchor="middle">S36</text><rect class="window-s" x="372" y="0" width="50" height="34"/><text class="window-label" x="397" y="22" text-anchor="middle">S37</text><text class="window-label" x="446" y="22" text-anchor="middle">…</text><rect class="window-f" x="474" y="0" width="50" height="34"/><text class="window-label" x="499" y="22" text-anchor="middle">F41</text></g>
<text class="lane-sub" x="268" y="516">S: query sees only its previous 512-token causal window · F: query can address the full causal prefix</text>
<text class="lane-sub" x="22" y="562">Boxes name the localization granularity supported by the experiments; arrow length and box size do not encode effect size.</text>
</svg>
<div class="mechanism-live-grid" aria-live="polite"><div><strong id="mechanism-live-q-title"></strong><span id="mechanism-live-q-body"></span></div><div><strong id="mechanism-live-g-title"></strong><span id="mechanism-live-g-body"></span></div></div>
<figcaption id="mechanism-paper-caption"><strong>图 1 · Stepwise non-thinking counting mechanism。</strong>上下两条 lane 使用相同的五步语义，但不是同一组 heads。Qwen 的读取/写入被定位到 L28 H16/H19；Gemma 的 K2 source heads 位于周期性 full-attention layers L29/L35，随后由 L37 residual mediator 将其影响传到 L41。图没有数值坐标轴；从左到右表示计算顺序，高亮仅表示当前讲解步骤。</figcaption>
</figure>

<div class="mechanism-definitions">
<h3>0.1 先定义“提取 state”和“计算 count direction”</h3>
<div class="mechanism-definition-grid">
<div><strong>Prompt state</strong><span>对 seed <em>s</em> 的同一个 N=10 prompt，定位第 <em>n</em> 个 active needle 的最后一个 token <code>t<sup>end</sup><sub>s,n</sub></code>，保存第 ℓ 个 decoder block 后的完整 residual：<code>h<sup>P</sup><sub>s,n,ℓ</sub>=h<sub>ℓ</sub>(t<sup>end</sup><sub>s,n</sub>)</code>。所以 n=1…10 是同一条 prompt 内的读取进度，不是十条不同 prompt。</span></div>
<div><strong>Answer state</strong><span>对 gold count 为 N 的 prompt，在生成第一个答案 token 之前，保存 prompt-final <code>Total:</code> query 的 post-block residual：<code>h<sup>A</sup><sub>s,N,ℓ</sub></code>。这一位置之后直接连接 LM head，因此 answer patching 在此处检验“state 是否可执行”。</span></div>
<div><strong>Count step</strong><span>在独立 discovery seeds 上对完整 residual 做逐维 OLS：<code>b<sub>ℓ</sub>=Σ<sub>i</sub>(c<sub>i</sub>−c̄)(h<sub>i,ℓ</sub>−h̄<sub>ℓ</sub>)/Σ<sub>i</sub>(c<sub>i</sub>−c̄)²</code>。它表示 count 增加 1 时 residual 的平均向量变化；单位轴为 <code>u<sub>ℓ</sub>=b<sub>ℓ</sub>/||b<sub>ℓ</sub>||</code>。PCA 只把 frozen discovery basis 投影成 3D，不参与因果干预。</span></div>
<div><strong>Natural V-path step</strong><span>OV intervention 不把上面的 raw-residual axis 直接塞进 head。先在 value-source layer 对实际 pre-attention input <code>x=RMSNorm(h)</code> 拟合一单位 count slope <code>s<sup>x</sup></code>。在线性 value path 中，query head h 的 pre-O one-count step 是 <code>d<sub>z,h</sub>=W<sub>V</sub><sup>g(h)</sup>s<sup>x</sup></code>；若模型有共享 value-source layer 或 <code>v_norm</code>，则直接对实际 <code>v<sub>g</sub>(c)</code> 做 OLS，取经验 slope <code>d<sub>z,h</sub></code>。因此 injection 使用的是模型自然 V path 中“一次 count 增量”的 head-space 向量，而不是答案轴的任意可达投影。</span></div>
<div><strong>Head read/write</strong><span>先对进入 attention block 的 residual 做该模型的 RMSNorm，记为 <code>x(j)=Norm<sub>ℓ</sub>(h(j))</code>。对 query head h 及其 GQA KV group <code>g(h)</code>：<code>q<sub>h</sub>=W<sub>Q</sub><sup>h</sup>x(q)</code>，<code>k<sub>g</sub>(j)=W<sub>K</sub><sup>g(h)</sup>x(j)</code>，<code>v<sub>g</sub>(j)=W<sub>V</sub><sup>g(h)</sup>x(j)</code>；应用架构中的 Q/K normalization 与 RoPE 后，<code>α<sub>h</sub>(q,j)=softmax<sub>j∈A(q)</sub>(q<sub>h</sub>·k<sub>g</sub>(j)/√d)</code>，其中 <code>A(q)</code> 是 full prefix 或 Gemma 的512-token window。pre-O state 为 <code>z<sub>h</sub>(q)=Σ<sub>j∈A(q)</sub>α<sub>h</sub>(q,j)v<sub>g</sub>(j)</code>，写回为 <code>o<sub>h</sub>(q)=W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q)</code>，block output 再通过 residual addition 接回 <code>h(q)</code>。α 回答“从哪里取”，V 回答“取到什么”，W<sub>O</sub> 回答“以什么 residual direction 写回”。</span></div>
<div><strong>Causal transport</strong><span>head/path intervention 用 gold-count 间距归一化：<code>T<sub>E</sub>=[E(C)<sub>I</sub>−E(C)<sub>R</sub>]/(D−R)</code>。answer-query full-state patch 则以两条自然预测的间距归一化：<code>T<sub>pred</sub>=(y<sub>patch</sub>−y<sub>R</sub>)/(y<sub>D</sub>−y<sub>R</sub>)</code>，只在两端预测均为有效且不同的数字时定义。correct-only 分析要求 donor 与 receiver 的原始输出都正确，再计算 patch 后采用 donor gold count 的比例。</span></div>
</div><p class="mechanism-index-note">除非另行说明，本报告中的 layer/head index 均为 zero-based；所有 axes、head sets 与 mediator layer 都在 confirmation outcome 之前冻结。</p>
</div>

<article class="model-mechanism qwen">
<div class="model-mechanism-header"><h3>Qwen3-8B：从 full-span early retrieval 到局部 OV 写入</h3><p><strong>完整路径：</strong>needle-end running state → L&lt;28 full-span early top-4 → L28 H16/H19 的 α/V mixed read → H16/H19 自身 W<sub>O</sub> 写回 → L29–L35 answer-query count state → <code>Total:N</code>。</p></div>
{table(["执行者/位置", "读取对象", "具体计算", "写入或输出", "支持证据"], q_actor_rows, classes="paper-table compact-result-table mechanism-actor-table")}
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">在 needle endpoint 提取 running state</span><span class="mechanism-step-action">按上面的 <code>h<sup>P</sup><sub>s,n,ℓ</sub></code> 定义，在同一 N=10 prompt 中依次读取十个 endpoint。3D 图把 V4.4 states 投影进冻结的 V4.1 display basis；下列 R² 则来自 V4.4 内按 seed 分组的 held-out full-space ridge，两者不是同一个拟合。它们共同检验“读到第 n 个 occurrence 后 residual 是否有序变化”，不假设神经元里存在字面整数 n。<span class="formula-line">hᴾ(s,1,ℓ) → hᴾ(s,2,ℓ) → … → hᴾ(s,10,ℓ)</span></span><span class="mechanism-step-evidence">L8 ridge R²=0.945 · centroid rank-3=0.988 · needle-token specific Δ|error|=8.930, Holm p=0.046875</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">full-span early bank 改写 slot states</span><span class="mechanism-step-action">在 discovery 中按 full-span literal mass × occurrence coverage 排序，并冻结 nested K=1/2/4/8/16/32。top-4 是 {q_fs_top4_text}。对10个 evaluation seeds、count 1–10、6个双向 donor pairs，只在注册 slot-state positions 把这四个 heads 的 donor pre-O <code>z</code> 写入 receiver，再测 donor-vs-receiver log-odds gain。随后在 L28 H16/H19 的真实 pre-O slices 精确删除 early patch 诱发的 natural-OV component，并与同输出 span、等范数正交 block 比较。</span><span class="mechanism-step-evidence">top-4 slot-state source={q_fs_top4['early_donor_log_odds_gain_mean']:.3f} [{q_fs_top4['early_donor_log_odds_gain_ci_low']:.3f}, {q_fs_top4['early_donor_log_odds_gain_ci_high']:.3f}] · mediation={q_fs_top4['donor_log_odds_mediation_specificity_mean']:.3f} [{q_fs_top4['donor_log_odds_mediation_specificity_ci_low']:.3f}, {q_fs_top4['donor_log_odds_mediation_specificity_ci_high']:.3f}] · both Holm p={fmt_p(q_fs_top4_p)}</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">L28 H16/H19 同时用 routing 与 value 读取</span><span class="mechanism-step-action">在 answer query q 保存 receiver/donor 的全部 α 与 V，并构造四个 pre-O endpoint：RR、RD、DR、DD；第一个字母表示 α 来源，第二个表示 V 来源。Shapley 分解把同一个 donor movement 精确分账：<span class="formula-line">Δz<sub>value</sub>=½[(z<sub>RD</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>DR</sub>)]</span><span class="formula-line">Δz<sub>route</sub>=½[(z<sub>DR</sub>−z<sub>RR</sub>)+(z<sub>DD</sub>−z<sub>RD</sub>)]</span>两个分量都必须既推动 donor count，又通过冻结 natural-OV axis 才算自然读取。</span><span class="mechanism-step-evidence">routing family p={fmt_p(read_write['primary_decision']['read_mode']['routing_family_p'])} · value family p={fmt_p(read_write['primary_decision']['read_mode']['value_family_p'])}</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">真实 pre-O OV 写入并改变坐标</span><span class="mechanism-step-action">先在 value-source layer 的实际 RMSNorm 后输入上拟合单位 count slope <code>s<sup>x</sup></code>，并按 GQA 映射到每个 query head：标准线性 value path 用 <code>d<sub>z,h</sub>=W<sub>V</sub><sup>g(h)</sup>s<sup>x</sup></code>；共享/非线性 value path 则直接拟合实际 value states 的经验 slope。再定义 set 的自然写入方向 <code>m<sub>S</sub>=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>d<sub>z,h</sub></code>。真实 injection 在 W<sub>O</sub> 之前做 <code>z<sub>h</sub>←z<sub>h</sub>+βd<sub>z,h</sub></code>；centered removal 从 <code>z−z₀</code> 中删除沿 <code>m<sub>S</sub></code> 的自然分量，并与同一 W<sub>O</sub> span、等 post-O 范数的正交控制比较。所有 residual 变化都必须经过 heads 自己的 W<sub>O</sub>。</span><span class="mechanism-step-evidence">natural OV global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">沿 frozen answer axes 传播并驱动数字</span><span class="mechanism-step-action">对每个 downstream layer 用 discovery answer-query states 拟合自然 count step <code>b<sup>A</sup><sub>ℓ</sub></code>，再计算注入差分在该轴上的系数：<span class="formula-line">a<sub>ℓ</sub>=&lt;[h<sub>ℓ</sub>(+β)−h<sub>ℓ</sub>(−β)]/(2β), b<sup>A</sup><sub>ℓ</sub>&gt;/||b<sup>A</sup><sub>ℓ</sub>||²</span>最后，用单层 full-residual donor patch 替换 receiver 的 <code>Total:</code> query state，并从原 receiver context 完整 greedy 生成；这检验的是可执行信息，不是 PCA 相似度。</span><span class="mechanism-step-evidence">all-sample transport={q_all['mean_effect']:.3f} · correct-only adoption={100*q_correct['pooled_average_patching_acc']:.1f}% · fresh p={fmt_p(q_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<article class="model-mechanism gemma">
<div class="model-mechanism-header"><h3>Gemma4-E4B：周期性全局读取与窗口内 residual 传递</h3><p><strong>完整路径：</strong>needle-end running state → full-attention L29H4/L35H2 K2 bank → donor pre-O z 经各自 W<sub>O</sub> 写回 → L37 distributed residual mediator → L41 terminal state → <code>Total:N</code>。</p></div>
{table(["执行者/位置", "读取对象", "具体计算", "写入或输出", "支持证据"], g_actor_rows, classes="paper-table compact-result-table mechanism-actor-table")}
<div class="window-explainer"><h4>512-token window 实际改变了什么？</h4><div class="window-explainer-grid"><div><strong>Sliding layer S</strong><p>query q 只能访问 <code>j∈[max(0,q−511),q]</code>。在约 10k-token prompt 的末端，S layer 无法直接重新读取大部分远端 needles。</p></div><div><strong>Full layer F</strong><p>每六层出现一次 full attention；zero-based 为 L5、11、17、23、29、35、41。确认的 L29H4/L35H2 正好都在 F layers，因此能够在 answer query 直接汇集全 prompt。</p></div><div><strong>机制后果</strong><p>L35 先把全局信息写进 answer-query residual；之后 L36–L40 的 S layers 即使看不到远端 needles，residual skip 仍携带这个 state，并可结合近端 query context 局部变换；L41 再进行一次全局层更新。</p></div></div><p class="mechanism-index-note">这不是说 window 内的单个 token 必须保存完整整数；它说明已确认的因果路径具有“F layer 全局刷新 → S layer 在同一 query residual 上保持/变换 → terminal readout”的架构节奏。Gemma4 配置中的 <a href="https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json">layer_types 与 sliding_window</a>固定为本实验所用 revision。</p></div>
<ol class="mechanism-step-list">
<li><span class="mechanism-step-number">01</span><span class="mechanism-step-title">用同一定义提取 prompt running state</span><span class="mechanism-step-action">仍然保存每个 active needle 最后 token 的 post-block residual <code>h<sup>P</sup><sub>s,n,ℓ</sub></code>，并在独立 discovery seeds 上拟合完整空间 count step。因为 S layer 的感受野有限，某层 endpoint state 可能是局部累计、前面 full layer 的全局刷新以及 residual/MLP 变换的合成；ordered geometry 本身不把三者强行拆开。</span><span class="mechanism-step-evidence">L9 ridge R²=0.719 · centroid rank-3=0.979 · needle-token specific Δ|error|=8.780, Holm p=0.046875</span></li>
<li><span class="mechanism-step-number">02</span><span class="mechanism-step-title">冻结 full-attention K2 source bank</span><span class="mechanism-step-action">从 full-span broad-retrieval 排序冻结 L29H4 与 L35H2，并准备三个 layer-matched K2 controls。对 donor/receiver count pair，只在 receiver 的 answer-query pre-O slice 替换这两个 heads 的 <code>z<sub>h</sub></code>；其余 heads、tokens 与 receiver context 不变。因为 replacement 位于 W<sub>O</sub> 输入端，任何 downstream effect 都由 Gemma 自己的 output projections 写入。</span><span class="mechanism-step-evidence">fresh full-span ablation：K1/K2 all-sample Holm-significant；K2 residual-path IUT significant</span></li>
<li><span class="mechanism-step-number">03</span><span class="mechanism-step-title">从候选 layers 中冻结 L37 residual mediator</span><span class="mechanism-step-action">在 discovery seeds 上先对自然 answer-query residual 拟合 <code>h<sub>ℓ</sub>(c)=a<sub>ℓ</sub>+c·b<sub>ℓ</sub></code>；再对 L36–L40 计算 K2 donor patch 引起的 <code>Δh<sub>ℓ</sub></code> 沿单位 count step 的平均投影。按 <code>mean&lt;Δh<sub>ℓ</sub>,u<sub>ℓ</sub>&gt;</code> 最大且 layer index 打破并列的预注册规则选择 L37，之后不再用 confirmation outcome 重选。</span><span class="mechanism-step-evidence">discovery seeds 1456–1465 · fit counts 1/3/5/7/9 · selected L37</span></li>
<li><span class="mechanism-step-number">04</span><span class="mechanism-step-title">用 exact block 与 count-axis block 验证传播</span><span class="mechanism-step-action">在 confirmation trial 中先测 source patch 在 L37 诱发的精确变化 <code>δ=h<sub>37</sub><sup>patch</sup>−h<sub>37</sub><sup>receiver</sup></code>。随后分别加入 <code>−δ</code>（exact block）或 <code>−proj<sub>b37</sub>(δ)</code>（count-axis block）；对照向量与被删分量等范数且正交。若 block 比对照更强地消除 donor log-odds gain，并同时降低 L41 沿 frozen count step 的 adoption，才认为 L37 中介 source-bank effect。</span><span class="mechanism-step-evidence">source + exact/count-axis mediation · global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></li>
<li><span class="mechanism-step-number">05</span><span class="mechanism-step-title">L41 terminal state 与完整答案输出</span><span class="mechanism-step-action">L41 的 state change 用 <code>&lt;Δh<sub>41</sub>,b<sub>41</sub>&gt;/||b<sub>41</sub>||²/(D−R)</code> 量化 donor count adoption；独立 answer-query full-state patch 再检验整个聚合 state 是否足以改变 greedy 数字。前者追踪 K2→L37→L41 的特定路径，后者验证最终 state 的可执行性。</span><span class="mechanism-step-evidence">terminal adoption p={fmt_p(gemma_residual['primary_decision']['families']['terminal_count_adoption']['intersection_union_p'])} · all-sample transport={g_all['mean_effect']:.3f} · correct-only adoption={100*g_correct['pooled_average_patching_acc']:.1f}% · fresh p={fmt_p(g_patch['source_donor_log_odds_gain_p'])}</span></li>
</ol>
</article>

<div class="mechanism-principle"><strong>为什么 prompt counter 与 answer counter 可以方向不同？</strong><div class="equation">h<sup>P</sup> → s<sup>x</sup>=OLS[Norm(h<sup>P</sup>)∼c] → d<sub>z,h</sub>=VPath<sub>g(h)</sub>(s<sup>x</sup>) → m<sub>S</sub>=ΣW<sub>O</sub>d<sub>z</sub> → u<sub>A</sub>∝J<sub>write→answer</sub>m<sub>S</sub></div><p>模型需要保持的是 count ordering、可解码性和 donor-directed causal transport，而不是让两个 token role 在欧氏空间共用一条直线。RMSNorm/value path、W<sub>O</sub> 与后续 attention/MLP Jacobian 都会旋转、缩放或压缩表示；所以判断“同一个 counter”应看 frozen-axis transport 与干预特异性，而不是要求两张 PCA 图视觉平行。</p></div>
<div class="conclusion"><strong>机制总览结论</strong>两模型都实现“prompt 累计 state → attention-head read → pre-O head state → W<sub>O</sub>/residual write → executable answer state”。这里的“prompt 累计 state”指 needle spans 中可读、可被后续 heads 汇集的分布式信息，不预先等同于 PCA 图中的前三个轴，也不声称最后一个 endpoint 是唯一 source。Qwen 的已确认写入粒度是 L28 H16/H19 set；Gemma 的已确认粒度是 L29H4/L35H2 source bank 及其 L37 residual mediator。表中未被单头干预区分的步骤只作 set-level 陈述，不据 attention 排名臆造逐头专职。</div>
</section>
"""


def build_scope_clear(
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    return f"""
<section id="scope">
<h2>1 · 结论先行</h2>
<div class="scope-lines">
<div class="scope-line"><strong>Prompt representation</strong><span>Qwen 与 Gemma 都形成随 occurrence index 有序变化的 running-counter geometry。</span><span class="status">REPRESENTATION</span></div>
<div class="scope-line"><strong>Prompt-subspace boundary</strong><span>同时删除所有 active endpoints 的冻结 rank-3 PCA / centroid component，相对等范数正交删除没有造成显著的行为或下游几何特异损伤。</span><span class="status">32/32 Holm null</span></div>
<div class="scope-line"><strong>Broad retrieval</strong><span>冻结 top-k bank 的 ablation 相对 layer-matched random heads 造成更大的计数行为变化。</span><span class="status">FUNCTIONAL</span></div>
<div class="scope-line"><strong>Qwen causal path</strong><span>full-span early top-4 → L28 H16/H19 mixed α/V read → natural OV write → L35 answer state。</span><span class="status">serial Holm p={fmt_p(q_fs_top4_p)} · OV IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}</span></div>
<div class="scope-line"><strong>Gemma causal path</strong><span>L29H4/L35H2 bank → L37 count-aligned residual → L41 answer state。</span><span class="status">IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}</span></div>
<div class="scope-line"><strong>Answer readout</strong><span>all-sample 与 correct-only answer-query patching 都显示 donor state 能推动 receiver 采用 donor count。</span><span class="status">CAUSAL PATCH</span></div>
</div>
<div class="conclusion"><strong>当前机制</strong>两模型都执行“prompt 中的分布式累计信息 → distributed retrieval → coordinate-changing write → executable answer state”。Qwen 的写入已定位到具体 OV set；Gemma 的写入定位在 L37 residual-level。第一步的 rank-3 geometry 是稳定、可解码的描述坐标，但本轮 set-wide removal 不支持把这三个轴本身写成唯一必要 carrier。</div>
</section>
"""


def build_causal_experiment_ledger(
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    q_answer_conditions = int(
        causal_v2["selection"]["Qwen3-8B"]["answer_patching"][
            "selected_conditions"
        ]
    )
    g_answer_conditions = int(
        causal_v2["selection"]["Gemma4-E4B"]["answer_patching"][
            "selected_conditions"
        ]
    )
    q_steer_conditions = int(
        causal_v2["selection"]["Qwen3-8B"]["steering"]["selected_conditions"]
    )
    g_steer_conditions = int(
        causal_v2["selection"]["Gemma4-E4B"]["steering"]["selected_conditions"]
    )
    directed_pairs_v2 = (
        "R←D: 0←1/1←0, 4←5/5←4, 9←10/10←9; "
        "0←3/3←0, 3←6/6←3, 7←10/10←7; "
        "0←5/5←0, 2←7/7←2, 5←10/10←5"
    )
    bidirectional_pairs = "R←D: 1←6, 6←1, 3←8, 8←3, 5←10, 10←5"
    return table(
        [
            "实验",
            "数据与冻结规则",
            "source → receiver",
            "实际改动位置与内容",
            "matched control",
            "effect 与显著性",
        ],
        [
            [
                "Full-span Top-K ablation",
                "Qwen/Gemma；fresh seeds 1316–1335；count 1–5；每模型100条。full-span score 与 nested K=1/2/4/8/16/32 在结果前冻结。",
                "无 donor；每条样本只改自身 final <code>Total:</code> query。",
                "仅在 final answer query 把 ranked heads 的 pre-<em>O</em> output slices 置零；prompt tokens 和其他 query 不改。",
                "3个 layer-matched random banks；每层删除的 head 数与 ranked bank 完全一致。",
                "all-sample: ranked−random absolute generated-count shift；correct-only: ranked−random correct→wrong rate。20个 seed 等权，10,000次 bootstrap；每 endpoint 12-way Holm。",
            ],
            [
                "All-sample answer-query patch",
                f"counts 0–10；screen seeds 1254–1258，held-out confirmation 1259–1263；screen 后冻结 Qwen {q_answer_conditions} / Gemma {g_answer_conditions} 个 layer×protocol×k 条件。",
                directed_pairs_v2,
                "把 donor 在 final <code>Total:</code> query 的完整 post-block residual 向量复制给 receiver；<code>single_layer</code> 只替换一层，<code>cumulative_from_layer</code> 从该层一直 clamp 到 final layer。",
                "同位置 self-patch；另加同 count、不同 seed 的 answer state。",
                "<code>T=(yP−y0)/(D−R)</code>，再减 matched controls。每个保留条件用5个 held-out seeds；seed bootstrap、exact sign flip，条件内 Holm。",
            ],
            [
                "Correct-only answer-query patch",
                "复用上一行已经冻结的 exact conditions；只保留同一 seed 内 donor 与 receiver clean generation 都正确的 pairs。每个 model×k×direction 至少5个 eligible seed clusters；k=1/3/5。",
                directed_pairs_v2,
                "仍是完整 answer-query residual donor→receiver copy；不重新选 layer、protocol 或 k。Qwen 缺口补入 seeds 1274/1275/1276/1278；Gemma 补入 1275/1277/1281/1295。",
                "正确性是入组条件；invalid patched generation 记 donor-adoption failure。",
                "<code>A=1[patched count=D]</code>；按 eligible pair pooled，seed-cluster bootstrap CI。Qwen/Gemma adoption 为96.6%/96.0%。",
            ],
            [
                "Geometric residual steering（宏观）",
                f"answer-query centroids 用 seeds 1234–1253、counts 0–10 拟合；screen 1254–1258，confirm 1259–1263；singleton screen 保留 Qwen {q_steer_conditions} / Gemma {g_steer_conditions} 个条件，并冻结 multi-layer plans。",
                directed_pairs_v2,
                "不是 donor patch。对 receiver 在 layer L 加 <code>μL,D−μL,R</code>；multi-layer plan 在冻结的多层同时加各层自己的 centroid delta；α=1。",
                "同一 layer、同一 receiver、与 centroid delta 正交且范数相同的确定性方向。",
                "control-adjusted normalized transport。它证明 population count direction 可操纵，不定位某个 OV head set。",
            ],
            [
                "Fresh correct-only answer aggregate patch",
                "20个共同 clean-correct seeds：1700/1701/1702/1704/1706–1713/1715–1722；counts 1/2/3；每 seed 六个有向 pairs。",
                "R←D: 1←2, 2←1, 1←3, 3←1, 2←3, 3←2。",
                "在 answer query 把冻结 source-bank 的完整 pre-<em>O</em> z 从 donor patch 到 receiver。Qwen source=L27H18/L23H28/L23H29/L26H20；Gemma source=L29H4/L35H2/L35H7/L35H1/L35H3/L29H2。",
                "同一 frozen writer space 的 exact block 与等输出范数正交 block；正文这里只使用不依赖 writer 成败的 source-patch endpoint。",
                "donor-vs-receiver sequence log-odds gain；20个 seed exact sign flip。Qwen p=1.43×10<sup>−5</sup>，Gemma p=9.54×10<sup>−7</sup>。",
            ],
            [
                "Qwen full-span early bank → L28",
                "seeds 1284–1293；counts 1–10；L&lt;28 full-span nested K=1/2/4/8/16/32；routes=<code>slot_state</code>/<code>slot_edge_qk</code>/<code>answer_query_full</code>。",
                bidirectional_pairs,
                "主路径把 donor 的 early top-4 pre-<em>O</em> slot-state <code>z</code> patch 给 receiver；随后在 L28 H16/H19 精确删除该 patch 诱发的 natural-axis output component。",
                "在同一 H16/H19 <em>W</em><sub>O</sub> span 删除等 post-<em>O</em> 范数、与 natural axis 正交的 component。",
                f"source donor log-odds gain 与 L28 mediation specificity 必须同时为正；18个K×route组合内 Holm，top-4 两项均 p={fmt_p(q_fs_top4_p)}。",
            ],
            [
                "Qwen route-family fresh replication",
                "fresh seeds 1294–1313；counts 1–10；旧 endpoint-ranked early top-4 与 slot-state route 预先冻结。",
                bidirectional_pairs,
                "同样先做 early donor-z patch，再在 L28 H16–H19 frozen set 做 exact/LOO block。",
                "L28 matched orthogonal block；并做 leave-one-out。",
                f"source+mediation IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}；确认 route family，不冒充新 full-span exact-set replication。",
            ],
            [
                "Qwen α/V read decomposition",
                "direction/center seeds 1264–1273；evaluation 1274–1293；counts 1–10；L28 H16/H19；downstream trace L28–L35。",
                bidirectional_pairs,
                "在同一 receiver query 构造 RR=<code>αRVR</code>、RD=<code>αRVD</code>、DR=<code>αDVR</code>、DD=<code>αDVD</code> 四个 pre-<em>O</em> states；Shapley 分解 full donor movement。",
                "每个 component patch 再与 L28 natural-axis block / same-span orthogonal block配对。",
                f"routing family p={fmt_p(read_write['primary_decision']['read_mode']['routing_family_p'])}；value family p={fmt_p(read_write['primary_decision']['read_mode']['value_family_p'])}。",
            ],
            [
                "Qwen natural pre-O steering",
                "direction seeds 1234–1253（fit counts 1/3/5/7/9；held-out 2/4/6/8/10）；center 1264–1273；confirmation 1274–1293；causal counts 2/5/8。",
                "无 donor。向 receiver 自己的 L28 H16/H19 state 加一个冻结的自然 one-count step。",
                "在实际 pre-attention value-source input <code>x=RMSNorm(h)</code> 上拟合 unit-count slope <code>sˣ</code>，经真实 GQA value path 得 <code>dz,h=WV[g(h)]sˣ</code>；在 <code>Total:</code> query 执行 <code>z←z+βdz</code>，β=−2/−1/0/1/2，变化只经这些 heads 自己的 <em>W</em><sub>O</sub> 写回。",
                "4个 outcome-blind matched K2 head sets；按 GQA relative position、natural-step norm、answer cosine、reachable cosine 与 baseline norm 匹配。",
                f"<code>ΔE[C]</code> 对 β 的 dose slope；candidate effect 与 candidate−control specificity 做 IUT，完整 natural-OV global p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}。",
            ],
            [
                "Qwen centered removal",
                "同上 confirmation seeds 1274–1293；causal counts 2/5/8；count-zero center 只用1264–1273拟合。",
                "无 donor；删除 receiver clean state 当前实际含有的 natural component。",
                "从 <code>zS−z0,S</code> 中减去其经 <em>W</em><sub>O</sub> 后沿 natural output step 的分量；不删除静态 offset。",
                "同一 <em>W</em><sub>O</sub> column span、相同 post-<em>O</em> norm、与 natural output step 正交的 removal。",
                "axis−control absolute-error increase 与 correct-margin decrease 必须同时显著；检验自然必要性。",
            ],
            [
                "Qwen donor-z mediation",
                "confirmation seeds 1274–1293；L28 H16/H19；三组低→高 pairs。",
                "R←D: 1←6, 3←8, 5←10。",
                "先把 donor 的完整 H16/H19 pre-<em>O</em> <code>z</code> 替换到 receiver；再仅删除 donor−receiver patch output 中平行 natural axis 的部分。",
                "在同一 <em>W</em><sub>O</sub> span加入等范数正交 block。",
                "donor transport 必须为正，且 <code>M=Torth−Taxis-block&gt;0</code>；检验 donor effect 是否经过该自然 OV channel。",
            ],
            [
                "Qwen downstream write trace",
                "discovery 1264–1273；evaluation 1274–1293；write counts 2/5/8；trace layers L28–L35；β=1。",
                "无 donor（write trace）；另用上面的双向 pairs做 read decomposition。",
                "比较 +natural step 与 −natural step 对每层 answer-query residual 的中心差分，并投影到该层冻结 answer-count step。",
                "同 span、等 post-<em>O</em> 范数的正交 injection propagation。",
                "natural−orthogonal frozen-axis coefficient；layer-wise Holm，检验写入后是否继续沿 answer-count direction 传播。",
            ],
            [
                "Gemma K2 source write",
                "candidate L29H4/L35H2；discovery 1456–1465；confirmation 1466–1485；counts 1–10，fit odd/held-out even；mediator candidates L36–L40 后冻结 L37。",
                "R←D: 1←6, 3←8, 5←10。",
                "在 answer query 依次把 receiver 的 L29H4 与 L35H2 pre-<em>O</em> <code>z</code> 替换为 donor <code>z</code>；每个 patched head 仍经自身 <em>W</em><sub>O</sub> 写回。",
                "3个冻结 layer-matched K2 banks：L29H1/L35H3、L29H6/L35H2、L29H7/L35H6。",
                f"normalized donor transport candidate core 与 candidate−control specificity 做 IUT；四阶段 global p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}。",
            ],
            [
                "Gemma L37 block → L41 adoption",
                "与上一行同一 frozen split、pairs 与 controls；L37/L41 count axes 只用 discovery 拟合。",
                "source 是 K2 patch 在 receiver 内诱发的 <code>δ37</code>；target 是 receiver 的 L37 residual / L41 terminal state。",
                "exact block 删除完整 <code>δ37</code>；count-axis block 只删除 <code>proj_b37(δ37)</code>；随后测 L41 state 在冻结 <code>b41</code> 上采用 donor gap 的比例。",
                "对 exact 与 count-axis 删除分别构造 residual-space 等范数正交删除；source bank 同时与3个 matched banks 比。",
                "source transport、exact mediation、count-axis mediation、terminal adoption 四个 family 全部通过 IUT。",
            ],
        ],
        classes="paper-table causal-ledger",
    )


def build_methods_clear(
    causal_v2: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    return f"""
<section id="methods">
<h2>2 · 实验设定与统计口径</h2>
<p>每个 stimulus 是约 10,000-token realistic haystack，包含十个可控 slots；gold count 由 active needles 的数量决定。non-thinking 条件关闭原生 thinking，并在 assistant 侧预填 <code>Total:</code>，随后对第一个数字 token 做 greedy generation。V4.4 随 seed 随机化 needle 的位置、内容与排列，从而避免把固定位置或固定文本误认为 count。Qwen3-8B 与 Gemma4-E4B 分开拟合 axes、冻结 heads 与计算显著性；layer/head 编号均为 zero-based。</p>
<h3>2.1 给 Transformer 读者的对象与操作词典</h3>
<p>本报告分析的是 decoder-only Transformer 的一次普通 forward。模型读完 prompt 后，在最后的 <code>Total:</code> token 位置产生一个 residual vector；LM head 再根据这个 vector 选择第一个答案数字。我们不假定 residual 的某个坐标天然叫作“count”，而是用独立数据找出随 gold count 有序变化的方向，再用干预验证模型是否真正使用它。</p>
{table(
    ["术语", "在计算图中的精确定义", "本报告怎样观测或改动"],
    [
        ["Token position", "序列中的一个位置。prompt endpoint 是一条 needle 最后一个 token；answer query 是生成第一个答案 token 前的 <code>Total:</code> 位置。", "同一层在不同 position 的 residual 不能直接当成同一种 state；所有图和干预都标明 position。"],
        ["Residual state <code>h<sub>ℓ,t</sub></code>", "第 ℓ 个 decoder block 后、位置 t 的 d-dimensional 向量；attention 与 MLP 的输出都加回这条 stream。", "PCA、回归和 state patching 都作用于或读取这个向量。"],
        ["Attention read", "head h 在 query q 计算 <code>z<sub>h</sub>(q)=Σ<sub>j</sub>α<sub>h</sub>(q,j)V<sub>h</sub>h(j)</code>；α 决定读哪里，V 决定把被读 state 映射成什么 head-space content。", "attention map 测 α；α/V decomposition 分别交换 routing 与 value content。"],
        ["OV write", "head-space 输出 z 还不是 residual；它必须经该 head 的输出投影 <code>W<sub>O</sub><sup>h</sup>z<sub>h</sub></code> 才写回 full residual。", "真正 OV intervention 在 pre-O z 上改动，再让模型自己的 W<sub>O</sub> 完成写回。"],
        ["Count axis / subspace", "由 discovery samples 拟合、能预测一单位 count 变化的方向，或由多个方向张成的低维空间；它是数据定义的坐标，不是模型参数中预先命名的 neuron。", "confirmation states 只投影到冻结 axis/basis；不会用同一数据先找方向再报告效果。"],
        ["Donor → receiver patch", "从同 seed、不同 gold count 的 donor forward 取 state，替换 receiver forward 的同一 layer/query/site。", "若 receiver 输出向 donor count 移动，说明被搬运的 state 含有可执行信息。"],
        ["Ablation", "把候选 head outputs 或指定 subspace component 置零/删除。", "必须与 layer-matched heads 或同空间等范数正交方向比较，避免把一般扰动误认为 count-specific necessity。"],
        ["Steering", "沿 discovery-frozen count direction 加入不同正负剂量，而不是复制某个 donor sample。", "符号正确且近似剂量响应支持方向的充分性。"],
        ["Mediation", "先由上游 patch 产生 donor-directed effect，再在下游候选通道阻断该 patch 诱发的 component。", "只有阻断候选方向比等范数正交阻断更能消除 effect，才支持这条具体路径被使用。"],
    ],
    classes="paper-table mechanism-actor-table",
)}
<div class="conclusion"><strong>本小节结论。</strong>Representation 回答“哪里含有 count 信息”；ablation/patching/steering 回答“这些信息能否影响输出”；mediation 回答“上游 effect 是否经过指定下游通道”。三类问题不能仅靠同一张 PCA 或 attention 图回答。</div>
<div class="causal-roadmap">
<div><strong>Representation</strong><span>在冻结 PCA / full-space axis 上检验 prompt running state 与 answer state 是否携带 count。</span></div>
<div><strong>Ablation</strong><span>置零 ranked top-k head outputs，并减去三个 layer-matched random sets 的平均影响。</span></div>
<div><strong>Patching</strong><span>把 donor state 搬到 receiver，观察 receiver 是否向 donor count 移动。</span></div>
<div><strong>OV / mediation</strong><span>在真实 pre-O z-space 做 injection/removal，或精确阻断写入 residual 的自然 count component。</span></div>
</div>
{table(
    ["实验", "独立单位", "样本/seed 设计", "主判定"],
    [
        ["V4.4 representation", "seed", "30 V4.4 seeds；count 1–10", "冻结 basis 投影；full-space statistics"],
        ["Needle token corruption", "seed", "10 fresh seeds（1254–1263）；count 1–10；clean / all-needle replacement / equal-token ordinary replacement", "seed-level specificity；50,000次 bootstrap；exact sign flip；population-wise Holm"],
        ["Set-wide prompt subspace ablation", "seed", "10 fresh seeds（1254–1263）；count 2–10；Qwen L8 / Gemma L9 全部 active endpoints", "rank-3或centroid-curve removal减等范数正交 removal；behavior + frozen answer geometry；population-wise Holm"],
        ["Full-span frozen top-k ablation", "seed", "20 fresh seeds；count 1–5；100 examples/model；K=1/2/4/8/16/32", "equal-seed mean；seed bootstrap CI；exact sign flip；每个 endpoint 12-way Holm"],
        ["Answer-query patching", "seed / directed pair", "all-sample held-out confirmation + clean-correct supplement", "control-adjusted donor transport / donor-target adoption"],
        ["Qwen natural OV", "seed", "20 confirmation seeds；matched W<sub>O</sub>-span controls", f"four-family IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}"],
        ["Qwen full-span read→write sweep", "seed", "10 frozen V4.4.4 seeds；count 1–10；K=1/2/4/8/16/32；3 routes", f"top-4 slot-state source与L28 mediation均Holm p={fmt_p(q_fs_top4_p)}（18 comparisons/endpoint）"],
        ["Qwen route-family independent replication", "seed", "20 fresh seeds；旧endpoint-ranked top-4；slot-state route冻结", f"source+mediation IUT p={fmt_p(upstream['primary_decision']['intersection_union_p'])}；仅作为route-family replication"],
        ["Gemma residual path", "seed", "20 confirmation seeds；candidate vs 3 matched banks", f"four-endpoint IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}"],
    ],
)}
<p><strong>统一统计单位：</strong>同一 seed 内的 counts、layers 与 donor pairs 先聚合，再让每个已注册实验中的10、20或30个 seed 等权进入跨-seed推断；具体数量逐行列在上表。主机制实验的图中 point 是 equal-seed macro mean，error bar 是对 seed means 做 bootstrap 的95% CI（既有实验10,000次；新 token/subspace 实验50,000次）；exact sign-flip 也以 seed effect 为输入。correct-only 条件下，不同 seed 的 eligible 样本数不同，因此另报 pooled eligible-example mean 作为敏感性量，但不把它与等权 seed mean 混为同一 estimand。多个必要条件组成一条机制时使用 intersection–union test（IUT）：family 或 global p 取全部必要检验中最大的 p，只有每一项都达到阈值时整条机制才通过。</p>
<p><strong>控制与多重比较：</strong>head-set ablation 的主效应均写成 ranked set 减去三个 layer-matched random sets 的平均效应，以控制“删除任意 K 个同层 heads”造成的一般扰动。对每个 endpoint，2 models × 6 K 形成12个冻结比较，并用 Holm 方法控制 family-wise error rate。OV/path 实验按问题使用两类控制：natural-signal 与 pre-O steering/injection 将候选 set 和四个 outcome-blind matched K=2 sets 比较；centered removal 与 mediation 则使用同一 <em>W</em><sub>O</sub> span、等 post-O 范数且与 natural count direction 正交的向量控制。这样分别排除“候选 heads 只是普通同层 heads”和“任意大向量都能推动输出”两种解释。</p>
<div class="plain-protocol"><h4>显著性数字怎样计算、怎样读</h4><ol><li>先对每个 seed 求配对效应 <code>e<sub>s</sub>=metric(intervention)−metric(control)</code>，并在 seed 内平均所有注册 counts / donor pairs；因此同一模板家族不会因产生更多行而得到更大权重。</li><li>报告效应是 <code>ē=(1/S)Σ<sub>s</sub>e<sub>s</sub></code>。95% bootstrap CI 通过有放回重采样这 <em>S</em> 个 seed means 得到，表示跨新 seed 的不确定性，而非 token-level 方差。</li><li>双侧 exact sign-flip 在“零假设下每个配对效应的正负号可交换”这一条件下枚举全部 <code>2<sup>S</sup></code> 个符号组合；<code>p</code> 是其绝对均值至少和观测值一样大的组合比例。10 seeds 时最小双侧 p 为 <code>2/2¹⁰=0.001953</code>。</li><li>若同一分析 family 同时查看多个模型、K、condition 或 endpoint，Holm 校正把最小原始 p 乘以 family 中尚未排除的检验数，并强制调整后 p 单调不减。正文把 <code>Holm p≤0.05</code> 作为 familywise 显著。</li><li>一个机制若要求多个必要 gate 同时成立，使用 IUT：取这些 gate p-values 的最大值。这样不会因为其中一个很强，就掩盖另一必要步骤没有通过。</li></ol></div>
<h3>2.2 Causal experiment ledger：每个 effect 到底改了什么</h3>
<p>下表使用统一记号 <code>R←D</code>，表示“把 donor count D 的 state 搬到 receiver count R 的 forward 中”。没有 donor 的 steering/removal 会明确写成“无 donor”。同一实验即使共享 seeds，也不会把不同量纲的 effect 相加。</p>
{build_causal_experiment_ledger(causal_v2, ov, read_write, upstream, fullspan_upstream, gemma_residual)}
<div class="conclusion"><strong>本节结论</strong>不同实验量纲不相加；机制由 representation、functional perturbation 与 causal mediation 在同一方向上收敛而建立。</div>
</section>
"""


def build_attention_estimand_note() -> str:
    return r"""
<div class="plain-protocol attention-estimand-note">
<h3>8.1 Full-span literal attention：retrieval bank 的唯一主排序</h3>
<div class="study-preface"><strong>为什么做。</strong><span>running state 存在并不等于 answer query 会读取它；attention atlas 用来定位哪些 heads 在最终 query 从完整 needle 文本取回 evidence，并据此冻结后续 ablation 的 retrieval bank。</span><strong>如何定义与评估。</strong><span>query 固定为 prompt-final <code>Total:</code> token。对每条 active needle 的完整 literal token span 求 attention mass，再乘 occurrence coverage；heads 只用 discovery data 排序，K 与 confirmation outcomes 无关。</span><strong>可视化。</strong><span>下方只保留 full-span atlas：横轴是 layer，纵轴是 head，颜色编码 seed-aggregated broad score。颜色只用于发现候选，功能性必须由第9节的 layer-matched ablation 验证。</span></div>
<p>设第 <em>i</em> 条 active needle 的完整 token span 为 <code>S<sub>i</sub></code>，head <em>h</em> 在最终 query <code>q</code> 的 attention row 为 <code>α<sub>h</sub>(q,j)</code>：</p>
<div class="equation">full-span mass: m<sub>i,h</sub>=Σ<sub>j∈S<sub>i</sub></sub>α<sub>h</sub>(q,j); &nbsp; total mass M<sub>h</sub>=Σ<sub>i</sub>m<sub>i,h</sub>; &nbsp; broad score S<sub>h</sub>=M<sub>h</sub>C<sub>h</sub>.</div>
<p>其中 <code>p<sub>i,h</sub>=m<sub>i,h</sub>/M<sub>h</sub></code>，coverage <code>C<sub>h</sub>=exp(−Σ<sub>i</sub>p<sub>i,h</sub>log p<sub>i,h</sub>)/N</code>。<code>M</code> 衡量分给全部 needles 的总概率质量，<code>C∈[1/N,1]</code> 衡量它是否覆盖多个 occurrences。高分因此要求“读得多”且“读得广”，不会因为只盯某一条 needle 而成为 broad-retrieval head。</p>
<p><strong>长度敏感性。</strong>literal sum 会随 span token 数增加而机械变大；当前 needle 使用同一固定模板，因此它比 span mean 更贴近“分给整条 needle 的总注意力概率”，但正式重排仍应同时报告 token-length-adjusted sensitivity（例如按 span length 分层，或把 mass 对 token length 回归后使用残差）。这样可以确认 top heads 来自检索强度，而不是某些 city/score 文本恰好分词更长。</p>
<div class="conclusion"><strong>本报告的判定规则</strong>只用 full-span literal score 发现 retrieval heads。我们已据此冻结 Qwen/Gemma 的 K=1/2/4/8/16/32 并在20个 fresh seeds 上做 matched ablation；Qwen upstream path 使用同一 score、但先限制到 L&lt;28，以免把 L28 writer 自己算进上游。Gemma的冻结 top-2 为 L29H4/L35H2。</div>
</div>
"""


def build_causal_section_clear(
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    correct_state: dict[str, Any],
) -> str:
    topk_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        for k_text, metrics in sorted(
            seed_confirmation["models"][model].items(), key=lambda item: int(item[0])
        ):
            all_shift = metrics["all_absolute_shift"]
            correct = metrics["clean_correct_to_wrong"]
            heads = [str(value) for value in metrics["heads"]]
            head_text = ", ".join(heads[:4])
            if len(heads) > 4:
                head_text += f" … (+{len(heads) - 4})"
            topk_rows.append(
                [
                    model,
                    f"K={k_text}",
                    head_text,
                    f"{fmt(all_shift['effect'], 4)} [{fmt(all_shift['ci95_low'], 4)}, {fmt(all_shift['ci95_high'], 4)}]",
                    f"{fmt_p(all_shift.get('two_sided_exact_seed_sign_flip_p'))} / {fmt_p(all_shift.get('holm_p_across_twelve_frozen_sets'))}",
                    f"{fmt(correct['effect'], 4)} [{fmt(correct['ci95_low'], 4)}, {fmt(correct['ci95_high'], 4)}]",
                    f"{fmt_p(correct.get('two_sided_exact_seed_sign_flip_p'))} / {fmt_p(correct.get('holm_p_across_twelve_frozen_sets'))}",
                ]
            )

    def significant_ks(model: str, metric: str) -> list[int]:
        return [
            int(k_text)
            for k_text, metrics in sorted(
                seed_confirmation["models"][model].items(),
                key=lambda item: int(item[0]),
            )
            if float(metrics[metric]["holm_p_across_twelve_frozen_sets"]) <= 0.05
        ]

    q_all_sig = significant_ks("Qwen3-8B", "all_absolute_shift")
    q_clean_sig = significant_ks("Qwen3-8B", "clean_correct_to_wrong")
    g_all_sig = significant_ks("Gemma4-E4B", "all_absolute_shift")
    g_clean_sig = significant_ks("Gemma4-E4B", "clean_correct_to_wrong")

    def format_k_set(values: list[int]) -> str:
        return ", ".join(str(value) for value in values) if values else "none"

    patch_rows: list[list[str]] = []
    for model in ("Qwen3-8B", "Gemma4-E4B"):
        all_patch = causal_v2["primary_confirmation_family_summary"][
            f"{model}::answer_patching"
        ]
        correct_patch = causal_v2["correct_interventions"]["patch_pooled"][
            f"{model}::answer_patching"
        ]
        fresh = _route_row(correct_state, model, "answer_query_aggregate")
        patch_rows.append(
            [
                model,
                fmt(all_patch["mean_effect"], 4),
                f"{int(all_patch['ci95_excludes_zero'])}/{int(all_patch['conditions'])}",
                f"{100 * correct_patch['pooled_average_patching_acc']:.1f}%",
                f"{fmt(fresh['source_donor_log_odds_gain_mean'], 4, signed=True)} "
                f"[{fmt(fresh['source_donor_log_odds_gain_ci95_low'], 4)}, {fmt(fresh['source_donor_log_odds_gain_ci95_high'], 4)}]",
                fmt_p(fresh["source_donor_log_odds_gain_p"]),
            ]
        )

    return f"""
<section id="causal">
<h2>9 · 因果实验：先定位功能 bank，再验证可执行 answer state</h2>
<h3>9.1 这部分只回答两个问题</h3>
<div class="causal-roadmap">
<div><strong>① 必要性</strong><span>在注册层删除 discovery-frozen rank-3 count component，并减去固定正交、近等范数 control 的损伤；分别估计 needle-end 与 answer-query 两个位置。</span></div>
<div><strong>② 方向充分性</strong><span>沿跨层 transport basis 注入 1×/2× count step，并与等范数正交注入比较，检验 donor-directed movement 与剂量响应。</span></div>
<div><strong>③ Retrieval-bank 功能</strong><span>删除 full-span score 冻结的 nested top-K heads，并减去三个 layer-matched random banks 的平均影响。</span></div>
<div><strong>④ 可执行 state 与路径</strong><span>把 donor answer state patch 到 receiver，再用 pre-O OV 与 residual mediation 检验 donor effect 是否经指定 write/relay 到达输出。</span></div>
</div>

<h3>9.2 Effect 的逐样本定义与跨 seed 聚合</h3>
<div class="test-card"><h4>Ablation：ranked bank 必须超过同层随机删除</h4><dl>
<dt>记号</dt><dd>对样本 <em>i</em>，<code>y<sub>0i</sub></code> 是 clean greedy count，<code>y<sub>Ki</sub></code> 是删除 ranked top-K 后的 count，<code>y<sub>Kir</sub><sup>rand</sup></code> 是第 <em>r</em> 个 layer-matched random set 的输出；本实验每个样本有 <code>R=3</code> 个随机 set。</dd>
<dt>All-sample absolute-shift effect</dt><dd><span class="formula-line">d<sub>i</sub><sup>abs</sup>=|y<sub>Ki</sub>−y<sub>0i</sub>|−R<sup>−1</sup>Σ<sub>r</sub>|y<sub>Kir</sub><sup>rand</sup>−y<sub>0i</sub>|</span>正值表示 ranked bank 的删除比删除同层随机 heads 更能改变模型实际生成的 count；它只度量变化大小，不把 over-count 与 under-count 抵消。</dd>
<dt>Correct-only failure effect</dt><dd>先限制到 clean 输出正确且格式有效的样本 <code>y<sub>0i</sub>=g<sub>i</sub></code>，再计算：<span class="formula-line">d<sub>i</sub><sup>fail</sup>=1[y<sub>Ki</sub>≠g<sub>i</sub>]−R<sup>−1</sup>Σ<sub>r</sub>1[y<sub>Kir</sub><sup>rand</sup>≠g<sub>i</sub>]</span>正值是 ranked bank 额外造成的 correct→wrong 概率。</dd>
<dt>Companion ΔMAE</dt><dd><span class="formula-line">d<sub>i</sub><sup>MAE</sup>=(|y<sub>Ki</sub>−g<sub>i</sub>|−|y<sub>0i</sub>−g<sub>i</sub>|)−R<sup>−1</sup>Σ<sub>r</sub>(|y<sub>Kir</sub><sup>rand</sup>−g<sub>i</sub>|−|y<sub>0i</sub>−g<sub>i</sub>|)</span>正值表示 ranked ablation 相对随机 ablation 额外增加绝对计数误差。</dd>
</dl></div>
<div class="test-card"><h4>Patching：把 receiver state 朝 donor count 搬运多少</h4><dl>
<dt>All-sample normalized transport</dt><dd>receiver/donor 的真实 counts 分别为 <code>R</code> 与 <code>D</code>，clean receiver 与 patch 后输出为 <code>y<sub>0</sub></code>、<code>y<sub>P</sub></code>：<span class="formula-line">T=(y<sub>P</sub>−y<sub>0</sub>)/(D−R)</span><code>T=1</code> 表示生成变化恰好覆盖完整 donor–receiver count gap，<code>T=0</code> 表示未沿 donor 方向移动，负值表示反向移动；无效生成在 strict estimand 中记为 0。主 effect 再减去同一样本 self-patch / same-count controls 的平均 transport：<span class="formula-line">T<sub>adj</sub>=T<sub>donor</sub>−mean(T<sub>control</sub>)</span></dd>
<dt>Correct-only donor adoption</dt><dd>只保留 donor 与 receiver 的 clean 输出都等于各自 gold count 的 pair；<span class="formula-line">A=1[y<sub>P</sub>=D]</span>表中百分比是所有 eligible pairs 的 pooled mean <code>mean(A)</code>，即 patch 后精确生成 donor gold count 的比例。</dd>
<dt>Fresh low-count source gain</dt><dd>对候选数字序列计算 donor-vs-receiver log-odds，令 <code>ℓ<sub>D</sub>−ℓ<sub>R</sub></code> 为 donor count 相对 receiver count 的优势：<span class="formula-line">G=([ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>patch</sub>−[ℓ<sub>D</sub>−ℓ<sub>R</sub>]<sub>clean</sub>)</span>正值表示 patch 提高 donor count 的相对概率；这是 fresh correct-only path 实验的连续 endpoint。</dd>
</dl></div>
<p>所有逐样本值先在 seed 内平均，再把 seed 当作独立 cluster 求总体均值；95% CI 用 10,000 次 seed-cluster bootstrap。注册的 p 值来自 seed-level exact sign-flip；每个 endpoint 的12个冻结 model×K 比较使用 Holm 校正。图中的点不是把 token、head 或 donor pair 当独立样本得到的。</p>

<h3>9.3 Full-span Top-K ablation：检索 bank 是否具有因果功能？</h3>
<div class="study-preface"><strong>为什么做。</strong><span>attention score 只能说明某些 heads 把概率质量分配给 needle spans；它不能证明这些 heads 对生成 count 有用。我们因此删除按 full-span literal score 冻结的 nested head sets，并要求其影响超过删除同层随机 heads。</span><strong>如何定义与评估。</strong><span>排序分数对每个 head 先求全部 active needle spans 的 attention mass，再乘 entropy-based occurrence coverage；K 在任何 outcome 之前冻结为 1/2/4/8/16/32。实验使用 fresh seeds 1316–1335、count 1–5，共100条/model。只在 final <code>Total:</code> query 把 selected heads 的 pre-<em>O</em> output slices 置零；其他 prompt/query positions 不变。每个 ranked K 配3个层分配完全相同的 layer-matched random banks。主 estimands 是上节定义的 <code>d<sup>abs</sup></code> 与 <code>d<sup>fail</sup></code>；exact sign-flip 以20个 seed effects 为输入，并在每个 endpoint 的12个 model×K tests 内做 Holm 校正。</span></div>
<figure>{ablation_topk_svg_fullspan(seed_confirmation)}<figcaption><strong>图 8E · Full-span-ranked nested-K ablation。</strong>横轴是冻结的 K=1/2/4/8/16/32，等距展示对应 log<sub>2</sub> 剂量。左图纵轴为 all-sample ranked-minus-random absolute generated-count shift；右图为 baseline-correct 样本的 correct-to-wrong probability excess。圆形为 Qwen，方形为 Gemma；实心表示该 model×K 在本 endpoint 的12比较 Holm p≤.05，空心表示未通过。竖线是对等权 seed means 做10,000次 bootstrap 的95% CI。</figcaption></figure>
{table(["model", "K", "full-span-ranked heads（前4名）", "all-sample effect [95% CI]", "all exact/Holm p", "correct-only effect [95% CI]", "correct exact/Holm p"], topk_rows, classes="paper-table compact-result-table")}
<p><strong>结果。</strong>Qwen 的 all-sample effect 在 K={format_k_set(q_all_sig)} 通过12比较 Holm，correct-only effect 在 K={format_k_set(q_clean_sig)} 通过；Gemma 的 all-sample effect 在 K={format_k_set(g_all_sig)} 通过，correct-only effect 在 K={format_k_set(g_clean_sig)} 通过。全-layer behavior ranking 的 Qwen top-4 为 L27H18、L28H19、L23H29、L23H13；Gemma top-2 为 L29H4、L35H2。Qwen 的 upstream path 另按同一 full-span score 在 L&lt;28 候选域排序，其 early top-4 为 L27H18、L23H29、L23H13、L23H28。这个区分保证 mediator L28H19 不会同时被计作上游 source。</p>
<p>剂量曲线明显非单调：Qwen K=8 小于 K=4，而 K=16/32 急剧增大；Gemma K=4 小于 K=1/2，而 K=8 达到峰值。这说明 K 不是“同一种独立 head 功能”的线性剂量。较大 set 同时删除协同、冗余和可能互相补偿的 heads；因此本实验定位的是功能 bank 的范围，而不是从曲线反推出唯一最小 circuit。</p>
<div class="conclusion"><strong>本小节结论</strong>full-span-ranked heads 对两模型的 counting 都有可重复、matched-control-specific 的因果贡献；核心机制 heads 位于该排序前部。结果支持分布式 retrieval bank，但不支持 effect 随 K 单调增加或存在唯一最佳 K。</div>

<h3>9.4 Answer-query patching：all samples 与 correct-only</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Ablation 只表明 retrieval bank 对行为重要，尚不能证明最终 answer-query residual 已经携带可执行的 count state。若把 donor 的 answer state 搬给 receiver 能系统推动 donor count，才说明该 representation 位于输出因果链上。</span><strong>如何定义与评估。</strong><span>all-sample 使用 counts 0–10、18个有向 anchor pairs；seeds 1254–1258 只做 screen，1259–1263 才做 held-out confirmation。每个条件把 donor 在 final <code>Total:</code> query 的完整 residual patch 给 receiver：single-layer 只替换 layer L，cumulative protocol 从 L 到 final layer 持续 clamp；控制为 self-patch 与 same-count different-seed state。correct-only 完全复用冻结条件，只筛选 donor/receiver clean 都正确的 pairs，测 donor gold adoption。</span><strong>可视化。</strong><span>左 panel 的纵轴是 control-adjusted normalized transport，右 panel 是 donor adoption probability；两者是不同 estimands，不能直接比较柱高。表中同时给出 held-out CI coverage、fresh continuous effect 的95% CI 与 p 值。</span></div>
<div class="ov-data-strip"><strong>具体配对。</strong><code>k=1</code>: 0↔1、4↔5、9↔10；<code>k=3</code>: 0↔3、3↔6、7↔10；<code>k=5</code>: 0↔5、2↔7、5↔10（每个↔都执行两个方向）。all-sample screen 后冻结 Qwen 149、Gemma 177 个 answer layer×protocol×k 条件。correct-only 要求每个 model×k×direction 至少5个 eligible seed clusters；Qwen 追加 1274/1275/1276/1278，Gemma 追加 1275/1277/1281/1295，追加规则只看 clean correctness，不看 patch outcome。</div>
<figure>{answer_patch_comparison_svg(causal_v2)}<figcaption><strong>图 8F · Answer-query state transport。</strong>横轴为模型（Qwen、Gemma）。左图纵轴是 all-sample held-out confirmation 中 answer-query patch 相对 control 的平均 normalized donor transport；右图纵轴是 donor 与 receiver 都 clean-correct 时，patch 后 receiver 生成 donor gold count 的比例。两个 panel 的统计量不同，因此只在各自 panel 内比较模型。</figcaption></figure>
{table(["model", "all-sample mean transport", "conditions with CI>0", "correct-only donor adoption", "fresh low-count source gain [95% CI]", "fresh p"], patch_rows, classes="paper-table compact-result-table")}
<p>all-sample estimand 中，Qwen 的平均 control-adjusted donor transport 为 0.7580，149/149 个冻结条件的 held-out CI 均大于 0；Gemma 为 0.7010，177/177 个条件均大于 0。clean-correct donor adoption 分别为 96.6% 与 96.0%。在另一组 20 fresh seeds、count 1–3 且 donor/receiver 都正确的实验中，Qwen 与 Gemma 的 answer aggregate source gain 也分别显著（p=1.43×10<sup>−5</sup> 与 9.54×10<sup>−7</sup>）。</p>
<div class="conclusion"><strong>本小节结论</strong>answer-query hidden state 不是只与 count 相关；它包含足以改变后续数字输出的可执行 count information，而且这一结果同时出现在全样本与 correct-only 分析中。</div>
</section>
"""


def _ov_component(
    ov: dict[str, Any], family: str, endpoint: str
) -> dict[str, Any]:
    hits = [
        item
        for item in ov["primary_decision"]["families"][family]["components"]
        if item["endpoint"] == endpoint
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Missing OV result {family}/{endpoint}")
    return hits[0]


def build_positive_mechanism_section(
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_rows = []
    for label, family, endpoint, meaning in (
        ("Natural carrier", "natural_signal", "natural_carrier_count_slope", "clean forward 中 H16/H19 output 随 count 有序变化"),
        ("True pre-O injection", "pre_o_injection", "injection_dose_slope", "在真实 z slice 加 natural step，输出按剂量向更高 count 移动"),
        ("Centered removal: error", "centered_removal", "removal_error_axis_minus_control", "删除自然 component 比等范数正交删除造成更大误差"),
        ("Centered removal: margin", "centered_removal", "removal_margin_axis_minus_control", "删除自然 component 特异降低正确 count margin"),
        ("Donor transport", "path_mediation", "donor_patch_transport", "donor z patch 把 receiver 推向 donor count"),
        ("Path mediation", "path_mediation", "mediation_control_minus_axis_block", "自然轴 block 比正交 block 消除更多 donor effect"),
    ):
        row = _ov_component(ov, family, endpoint)
        q_rows.append(
            [label, meaning, ci_text(row), fmt_p(row["p"])]
        )

    g_rows: list[list[str]] = []
    labels = {
        "source_donor_transport": ("Source transport", "K2 source-bank patch 把 answer computation 推向 donor count"),
        "exact_residual_mediation": ("Exact L37 mediation", "精确删除 patch-induced L37 residual change，特异削弱 transport"),
        "count_axis_mediation": ("Count-axis mediation", "只删除 L37 residual 中的 count-aligned component，同样削弱 transport"),
        "terminal_count_adoption": ("L41 adoption", "L37 写入提高 terminal layer 对 donor count 的采用"),
    }
    for family, document in gemma_residual["primary_decision"]["families"].items():
        core = next(
            item for item in document["components"] if item["role"] == "candidate_core"
        )
        label, meaning = labels[family]
        g_rows.append(
            [
                label,
                meaning,
                f"{fmt(core['mean'], 4)} [{fmt(core['ci95_low'], 4)}, {fmt(core['ci95_high'], 4)}]",
                fmt_p(core["p"]),
            ]
        )

    rw = read_write["primary_decision"]
    up = upstream["primary_decision"]
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    q_fs_supported_rows: list[list[str]] = []
    q_fs_route_labels = {
        "slot_state": "slot-state pre-O z",
        "slot_edge_qk": "slot-edge α/V",
        "answer_query_full": "answer-query full",
    }
    for row in sorted(
        (
            item
            for item in fullspan_upstream["summary"]
            if bool(item["serial_path_supported"])
        ),
        key=lambda item: (
            int(str(item["early_set"]).replace("top", "")),
            str(item["route"]),
        ),
    ):
        q_fs_supported_rows.append(
            [
                str(row["early_set"]),
                q_fs_route_labels.get(str(row["route"]), str(row["route"])),
                f"{row['early_donor_log_odds_gain_mean']:.4f} [{row['early_donor_log_odds_gain_ci_low']:.4f}, {row['early_donor_log_odds_gain_ci_high']:.4f}]",
                fmt_p(row["early_donor_log_odds_gain_holm_p"]),
                f"{row['donor_log_odds_mediation_specificity_mean']:.4f} [{row['donor_log_odds_mediation_specificity_ci_low']:.4f}, {row['donor_log_odds_mediation_specificity_ci_high']:.4f}]",
                fmt_p(row["donor_log_odds_mediation_specificity_holm_p"]),
            ]
        )
    q_rows.insert(
        0,
        [
            "Full-span early→L28 serial path",
            "L<28 full-span top-4 slot-state patch 先推动 donor count；阻断 L28 H16/H19 natural component 再特异消除该 effect",
            (
                f"source {q_fs_top4['early_donor_log_odds_gain_mean']:.4f} "
                f"[{q_fs_top4['early_donor_log_odds_gain_ci_low']:.4f}, {q_fs_top4['early_donor_log_odds_gain_ci_high']:.4f}]; "
                f"mediation {q_fs_top4['donor_log_odds_mediation_specificity_mean']:.4f} "
                f"[{q_fs_top4['donor_log_odds_mediation_specificity_ci_low']:.4f}, {q_fs_top4['donor_log_odds_mediation_specificity_ci_high']:.4f}]"
            ),
            f"both Holm {fmt_p(q_fs_top4_p)}",
        ],
    )
    q_families = ov["primary_decision"]["families"]
    q_natural = _ov_component(ov, "natural_signal", "natural_carrier_count_slope")
    q_injection = _ov_component(ov, "pre_o_injection", "injection_dose_slope")
    q_error = _ov_component(ov, "centered_removal", "removal_error_axis_minus_control")
    q_margin = _ov_component(ov, "centered_removal", "removal_margin_axis_minus_control")
    q_donor = _ov_component(ov, "path_mediation", "donor_patch_transport")
    q_mediation = _ov_component(ov, "path_mediation", "mediation_control_minus_axis_block")
    q_gate = evidence_gate_svg(
        [
            {"title": "Natural carrier", "main": f"count slope {ci_text(q_natural)}", "sub": "clean forward; candidate also exceeds matched sets", "p": f"family IUT p={fmt_p(q_families['natural_signal']['intersection_union_p'])}", "passed": q_families["natural_signal"]["passes_alpha"]},
            {"title": "True pre-O sufficiency", "main": f"dose slope {ci_text(q_injection)}", "sub": "z injection; output written only through H16/H19 W_O", "p": f"family IUT p={fmt_p(q_families['pre_o_injection']['intersection_union_p'])}", "passed": q_families["pre_o_injection"]["passes_alpha"]},
            {"title": "Centered necessity", "main": f"error specificity {ci_text(q_error)}", "sub": f"margin specificity {ci_text(q_margin)}", "p": f"family IUT p={fmt_p(q_families['centered_removal']['intersection_union_p'])}", "passed": q_families["centered_removal"]["passes_alpha"]},
            {"title": "Path mediation", "main": f"donor transport {ci_text(q_donor)}", "sub": f"axis-block specificity {ci_text(q_mediation)}", "p": f"family IUT p={fmt_p(q_families['path_mediation']['intersection_union_p'])}", "passed": q_families["path_mediation"]["passes_alpha"]},
        ],
        id_prefix="qwen-natural-ov-gates",
        title="Qwen L28 H16/H19 natural-OV evidence gates",
        description="Natural carrier, true pre-output sufficiency, centered necessity, and donor-path mediation must all pass. The global intersection-union p value is the largest family p value.",
    )
    g_gate_entries = []
    for family in (
        "source_donor_transport",
        "exact_residual_mediation",
        "count_axis_mediation",
        "terminal_count_adoption",
    ):
        document = gemma_residual["primary_decision"]["families"][family]
        core = next(
            item for item in document["components"] if item["role"] == "candidate_core"
        )
        label, meaning = labels[family]
        g_gate_entries.append(
            {
                "title": label,
                "main": f"effect {core['mean']:.4f} [{core['ci95_low']:.4f}, {core['ci95_high']:.4f}]",
                "sub": meaning,
                "p": f"family IUT p={fmt_p(document['intersection_union_p'])}",
                "passed": document["passes_alpha_and_ci"],
            }
        )
    g_gate = evidence_gate_svg(
        g_gate_entries,
        id_prefix="gemma-residual-gates",
        title="Gemma K2 source-to-residual causal gates",
        description="Source transport, exact L37 residual mediation, count-axis mediation, and L41 terminal adoption must all pass against three matched source banks.",
    )
    q_write_rows = sorted(
        [
            row
            for row in read_write["summary"]
            if row.get("metric") == "write_residual_specificity"
            and row.get("stratum") == "all"
        ],
        key=lambda row: int(row["layer"]),
    )
    q_write_svg = write_trace_svg(
        q_write_rows,
        id_prefix="qwen-write-propagation",
        title="Qwen L28 natural OV write propagation",
        description="Layer is on the horizontal axis. Natural-minus-orthogonal count-axis specificity is on the vertical axis. Points are equal-seed means and bars are 95 percent confidence intervals.",
    )
    return f"""
<section id="natural-ov">
<h2>10 · 已确认的写入与传播机制</h2>

<h3>10.1 OV 到底是什么：head 先读成 z，再由 W<sub>O</sub> 写进 residual</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Attention mass 只说明 head 看向哪里，不能说明读到的内容如何改变答案。OV 分析把一个 head 拆成“读出 head-space 内容”与“把内容写回 residual”两步，检验 count 是否真的经过这条计算边界。</span><strong>具体例子。</strong><span>receiver 当前倾向输出3。我们只在某个 head 的 pre-O <code>z</code> 中加入一个 discovery-frozen 的+1 count step；若这一步经过该 head 自己的 <code>W<sub>O</sub></code> 后使答案期望值向4移动，而等范数正交 step 不会，就说明这条 OV path 能写 count。</span><strong>读图顺序。</strong><span>先看本节四格流程理解变量，再看 Qwen/Gemma 各自的“source→write→mediator→answer”。折叠表保留完整公式和统计，正文只追踪每个干预到底改了哪里。</span></div>
<p><strong>先区分四个对象。</strong><code>α</code> 是“地址权重”（从哪些 token 读、各读多少）；<code>v(j)</code> 是每个 source token 可提供的内容；<code>z</code> 是一个 head 在把这些内容加权汇总后、尚未写回模型主干的向量；<code>W<sub>O</sub></code> 把 <code>z</code> 旋转/缩放成 residual stream 中的写入 <code>w</code>。OV 不是另一类 head，而是 <code>value aggregation → output projection</code> 这条真实计算路径。</p>
<p>对 answer query <code>q</code>，计算为：</p>
<div class="equation">z<sub>h</sub>(q)=Σ<sub>j</sub>α<sub>h</sub>(q,j)v<sub>h</sub>(j); &nbsp;&nbsp; w<sub>h</sub>(q)=W<sub>O</sub><sup>h</sup>z<sub>h</sub>(q); &nbsp;&nbsp; h′(q)=h(q)+Σ<sub>h</sub>w<sub>h</sub>(q).</div>
<div class="ov-short-flow" role="img" aria-label="OV computation from source count state to answer output">
<div class="ov-box"><strong>1 · 提取 count step</strong><span>从独立 discovery data 拟合一单位 count 变化；Qwen 再让它经过真实 value path，得到每个 head 的 <code>d<sub>z,h</sub></code>。</span></div><div class="ov-arrow" aria-hidden="true">→</div>
<div class="ov-box"><strong>2 · 在 pre-O z 改动</strong><span>在指定 query/head slice 做 add、replace 或 remove；此处仍是 head-space，不直接碰 residual answer axis。</span></div><div class="ov-arrow" aria-hidden="true">→</div>
<div class="ov-box"><strong>3 · W<sub>O</sub> 写回</strong><span>模型自己的 <code>W<sub>O</sub></code> 把改动映射成 full residual direction；这一步可以旋转和缩放 count direction。</span></div><div class="ov-arrow" aria-hidden="true">→</div>
<div class="ov-box"><strong>4 · 下游形成答案</strong><span>后续 attention、MLP 与 residual skip 继续传播该变化，最后改变候选数字的概率与生成 count。</span></div>
</div>
<p><strong>为什么 prompt 与 answer counter 可以不平行。</strong>prompt state 中的“一单位 count”先被 <code>V</code> 投影到 head space，再被 <code>W<sub>O</sub></code> 写到 residual，后续 attention/MLP 还可继续旋转。因此“语义相同”不要求两个原始向量的欧氏夹角接近0。行为端先由数字候选概率算 <code>E[C]=Σ<sub>c</sub>c·p(c)/Σ<sub>c</sub>p(c)</code>；donor/receiver 实验再用 <code>T=(E[C]<sub>I</sub>−E[C]<sub>R</sub>)/(D−R)</code>。例如 R=3、D=8，干预使期望值从3到5.5，则 <code>T=0.5</code>：走完了 donor gap 的一半。</p>
{table(
    ["操作", "向量从哪里来", "加到/替换哪里", "它单独验证什么"],
    [
        ["Steering / injection", "独立 discovery data 拟合的 one-count step <code>d<sub>z</sub></code>；不是 donor sample", "receiver 自己的指定 pre-O <code>z</code> 加 <code>βd<sub>z</sub></code>", "带符号充分性：这个 channel 能否按剂量推动 count"],
        ["Donor patch", "同 seed、不同 count 的 donor 完整 <code>z</code> 或 residual state", "用 donor state 替换 receiver 的同一 query/layer/site", "样本特异充分性：完整 donor state 能否被 receiver 使用"],
        ["Centered removal", "receiver 自己当前 <code>z−z<sub>0</sub></code> 中沿 natural axis 的分量", "只从 receiver pre-O <code>z</code> 删除可由该 head set 实现的分量", "自然必要性：模型原本是否依赖这条 channel"],
        ["Mediation", "先由 donor patch 产生 effect，再计算其中沿 natural axis 的 patch-induced component", "donor patch 保持不变，只在 mediator 处删除 natural component；与正交删除比较", "路径使用：donor effect 是否正经由该 write/relay 传递"],
    ],
    classes="paper-table ov-operation-table",
)}
<div class="conclusion"><strong>最短判读</strong>natural carrier 问“clean forward 里有没有这份 count”；steering 问“加进去能不能推动答案”；removal 问“自然运行是否需要”；mediation 问“上游 donor effect 是否经过这里”。本报告的 OV 干预都发生在 pre-O <code>z</code>，随后必须通过候选 heads 自己的 <code>W<sub>O</sub></code>；没有直接把 answer axis 加到 residual。</div>

<article class="positive-mechanism-model qwen">
<h3>10.2 Qwen：L28 H16/H19 的 localized natural-OV write</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Top-K ablation 只说明某个 bank 对 counting 有功能贡献；我们还要证明自然运行的 count component 确实进入 L28 H16/H19 的 <code>z</code>，经这两个 heads 自己的 <code>W<sub>O</sub></code> 写回，并继续影响答案。</span><strong>如何评估。</strong><span>候选 H16/H19、方向、center、matched controls 和20个 confirmation seeds 都在因果 outcome 前冻结。四个必要 families 是 natural carrier、pre-O steering、centered removal 与 donor-path mediation；global IUT p 取四个 family p 中最大者。</span></div>
<div class="ov-data-strip"><strong>冻结数据。</strong>one-count direction：seeds 1234–1253，odd counts 1/3/5/7/9 拟合、even counts 2/4/6/8/10 验证；count-zero center：1264–1273；因果确认：1274–1293。steering/removal 用 counts 2/5/8、β=−2/−1/0/1/2；donor mediation 用 <code>R←D: 1←6, 3←8, 5←10</code>。query 始终是第一个答案 token 前的 <code>Total:</code> position。</div>
<div class="test-card"><h4>Qwen 四步验证</h4><dl>
<dt>① 提取 natural one-count step</dt><dd>先在实际 value-source input <code>x=RMSNorm(h)</code> 上拟合“一单位 count”的 slope <code>s<sup>x</sup></code>。Qwen 使用 GQA，因此 query head <code>h</code> 对应 KV group <code>g(h)</code>；让 slope 经过模型真实 value projection：<span class="formula-line">d<sub>z,h</sub>=W<sub>V</sub><sup>g(h)</sup>s<sup>x</sup></span>堆叠 H16/H19 后得到 <code>d<sub>S</sub></code>。再让它经过各 head 自己的 output projection：<span class="formula-line">m<sub>S</sub>=Σ<sub>h∈S</sub>W<sub>O</sub><sup>h</sup>d<sub>z,h</sub></span><code>m<sub>S</sub></code> 才是写进 residual 的一单位 natural output step。</dd>
<dt>② Clean forward 是否自然携带它</dt><dd>用独立 center seeds 对每个 head 拟合 count=0 截距 <code>z<sub>0,h</sub></code>。将 clean <code>z<sub>S</sub>−z<sub>0,S</sub></code> 经 <code>W<sub>O</sub></code> 后投影到 <code>m<sub>S</sub></code>；该系数随 gold count 的 seed-level OLS slope 为 natural-carrier effect。它不做干预，只确认自然信号确实存在于该 write channel。</dd>
<dt>③ Steering 与 removal</dt><dd><strong>Steering：</strong>在 receiver 自己的 H16/H19 pre-O slices 加 <code>βd<sub>S</sub></code>，β 从−2到+2；不使用 donor，也不直接加 answer axis。主效应是 <code>ΔE[C]</code> 对 β 的 slope。<strong>Removal：</strong>从 receiver 当前 centered <code>z</code> 中删除沿 <code>m<sub>S</sub></code> 的可实现分量；与同 <code>W<sub>O</sub></code> span、等 post-O norm 的正交删除比较。前者检验充分性，后者检验自然必要性。</dd>
<dt>④ Donor mediation 与下游 trace</dt><dd>把 donor 的完整 H16/H19 <code>z</code> patch 给 receiver 后，计算 patch output 沿 <code>m<sub>S</sub></code> 的分量，并只删除该分量；中介效应 <code>M=T<sub>orth</sub>−T<sub>axis-block</sub></code>。随后将 ±steering 在 L28–L35 造成的 residual 中心差分投影到每层冻结 answer-count step；natural−orthogonal specificity 检验写入是否继续传播。L35 specificity={rw['write_propagation']['final_residual_specificity_mean']:.4f}，Holm p={fmt_p(rw['write_propagation']['final_residual_specificity_holm_p'])}。</dd>
</dl></div>
<p><strong>H16/H19 到底是靠“地址”还是“内容”读取。</strong>对同一 R←D pair，我们重放四种 head 计算：RR=receiver 的地址权重+receiver 的 value 内容；RD=receiver 地址+donor 内容；DR=donor 地址+receiver 内容；DD=donor 地址+donor 内容。这样只换 <code>α</code> 就是在问“读哪里”是否贡献 donor shift，只换 <code>V</code> 就是在问“同一位置读到的内容”是否贡献。Shapley average 对两种更换顺序取平均，避免把交互项全归给先换的变量。routing family p={fmt_p(rw['read_mode']['routing_family_p'])}，value family p={fmt_p(rw['read_mode']['value_family_p'])}；因此 H16/H19 的读取既依赖路由，也依赖内容。</p>
<p><strong>上游接入。</strong>使用同一 full-span score 在 L&lt;28 候选域冻结 nested K 后，top-4=L27H18/L23H29/L23H13/L23H28 的 slot-state patch 产生 donor log-odds gain={q_fs_top4['early_donor_log_odds_gain_mean']:.4f} [{q_fs_top4['early_donor_log_odds_gain_ci_low']:.4f}, {q_fs_top4['early_donor_log_odds_gain_ci_high']:.4f}]；在 L28 H16/H19 阻断 induced natural component 的 mediation specificity={q_fs_top4['donor_log_odds_mediation_specificity_mean']:.4f} [{q_fs_top4['donor_log_odds_mediation_specificity_ci_low']:.4f}, {q_fs_top4['donor_log_odds_mediation_specificity_ci_high']:.4f}]，两项在6个K×3 routes内 Holm p={fmt_p(q_fs_top4_p)}。这直接把 full-span early bank 接到 H16/H19 writer。另一个旧 endpoint-ranked top-4（与新 set 共享3/4成员）在20个独立 fresh seeds 上也得到 source+mediation IUT p={fmt_p(up['intersection_union_p'])}；它只作为 route-family 的独立 replication，不冒充新 top-4 的 exact-set replication。</p>
{table(["full-span early set", "source intervention", "source gain [95% CI]", "source Holm p", "L28 mediation [95% CI]", "mediation Holm p"], q_fs_supported_rows, classes="paper-table compact-result-table")}
<p>上表只列同时通过 source 与 L28 mediation 的组合。K=4/8/16/32 的 slot-state route 通过，K=16 的 slot-edge α/V route 也通过；K=1/2 没有通过串联 conjunction。因为所有 K 与 routes 在 outcome 前冻结且 p 已在18个组合内 Holm 校正，我们采用较小且已通过的 top-4 slot-state route 作为主图，而不以最大 effect 的 K 事后选路。</p>
<figure>{q_gate}<figcaption><strong>图 9A · Qwen natural-OV conjunction。</strong>该流程图没有数值坐标轴；四个框依次对应 natural carrier、pre-O sufficiency、centered necessity 和 donor-path mediation。框内给出 family 核心 effect、95% CI 与 IUT p；global IUT p 是四个 family p 的最大值，因此只有四门全部显著才支持 natural OV channel。</figcaption></figure>
<figure>{q_write_svg}<figcaption><strong>图 9B · Qwen write propagation。</strong>横轴为 decoder layer L28–L35；纵轴为 natural pre-O intervention 相对同 W<sub>O</sub> span、等 post-O norm 正交 control 多产生的 frozen answer-count-axis coefficient。点为20个 seed effects 的均值，竖线为95% CI；0 表示 natural 与 control 在该层的 count-aligned propagation 没有差异。</figcaption></figure>
<p class="result-sentence"><strong>结果。</strong>H16/H19 的自然载荷、pre-O 充分性、centered 必要性和 donor-path mediation 四个 families 全部通过，global IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}。routing 与 value 两个读取分量也分别显著；写入后的 specificity 一直保留到 L35。</p>
{table(["证据", "具体验证什么", "effect [95% CI]", "p"], q_rows, classes="paper-table compact-result-table")}
<div class="conclusion"><strong>Qwen 机制结论</strong>prompt running state 经 full-span-ranked early top-4 到达 L28；H16/H19 用 α 与 V 共同读取，再由自身 W<sub>O</sub> 写入新的 answer-residual direction，并传播到最终输出。新 exact-set 串联证据为复用冻结 seeds 的多重比较校正探索性确认；独立 fresh-seed 结果确认的是同一路由家族。</div>
</article>

<article class="positive-mechanism-model gemma">
<h3>10.3 Gemma：K2 bank 写入 L37 distributed residual</h3>
<div class="study-preface"><strong>为什么做。</strong><span>Gemma 的 sliding layers 只能直接查看512-token窗口，所以后层不必重新注意全部远端 needles。更合适的命题是：周期性 full-attention heads 先把全局信息写入 answer-query residual；中间局部 blocks 再传播这个已经写入的 state。</span><strong>如何评估。</strong><span>full-span ranking 冻结 K2=L29H4/L35H2；discovery 只选 mediator 与 count axes，confirmation 不再换 heads 或 layer。source transport、exact L37 mediation、count-axis mediation、L41 adoption 四项都必须通过 candidate 与 matched-control 两道门。</span></div>
<div class="ov-data-strip gemma"><strong>冻结数据。</strong>discovery seeds 1456–1465，confirmation 1466–1485；counts 1–10，odd counts 1/3/5/7/9 拟合 axes、even counts 2/4/6/8/10 held out；pairs 为 <code>R←D: 1←6, 3←8, 5←10</code>。mediator 只在 L36–L40 中用 discovery 选择，最终冻结 L37；terminal layer 固定 L41。三个 K2 controls 为 (L29H1,L35H3)、(L29H6,L35H2)、(L29H7,L35H6)。</div>
<p><strong>这条路径只问三件事：</strong>（1）L29H4/L35H2 的 head outputs 能否把 donor count 写进 receiver；（2）这次写入在 L37 留下的 residual change 是否是必经中介；（3）该 change 是否继续到 L41 的 terminal count state。下列四个统计 gate 是这三个问题的可检验拆分，其中第2项把 L37 mediator 又分成“完整向量”和“其中的 count-aligned 投影”。</p>
<div class="test-card"><h4>Gemma 四步验证</h4><dl>
<dt>① K2 pre-O source patch</dt><dd>在 receiver 的 <code>Total:</code> query，先于 output projection 把 <code>z<sub>29,4</sub></code> 和 <code>z<sub>35,2</sub></code> 替换成同 seed donor 的值；其他 heads 与 receiver residual 不改。被替换的两个 <code>z</code> 分别通过各自的 <code>W<sub>O</sub></code> 写入。因此它不是直接把 donor answer direction 加到 residual，而是真实的 head-output-path patch。source effect 是 <code>T<sub>source</sub></code>。</dd>
<dt>② 在 L37 定义实际写入结果</dt><dd>同一 K2 patch 到达 L37 时产生 <span class="formula-line">δ<sub>37</sub>=h<sub>37</sub><sup>K2 patch</sup>−h<sub>37</sub><sup>clean R</sup></span><code>δ<sub>37</sub></code> 包含 L29/L35 两次 <code>W<sub>O</sub></code> 写回及其间 blocks 的合成结果；它是在 residual space 中观察到的实际 mediator。</dd>
<dt>③ 阻断 L37 mediator</dt><dd><strong>Exact block：</strong>在保留 K2 patch 的 forward 中删除完整 <code>δ<sub>37</sub></code>。<strong>Count-axis block：</strong>只删除 <code>δ<sub>37</sub></code> 在冻结 L37 count direction <code>b<sub>37</sub></code> 上的投影。两者都和 residual-space 等范数正交删除比较：<span class="formula-line">M=T<sub>orth</sub>−T<sub>block</sub></span>正值说明 K2 donor effect 确实经过 L37，并且其中有可识别的 count-aligned component。</dd>
<dt>④ 检查是否到达 terminal answer state</dt><dd>计算 K2 patch 在 L41 造成的 <code>Δh<sub>41</sub></code>，投影到 discovery 冻结的 terminal count step：<span class="formula-line">A<sub>41</sub>=&lt;Δh<sub>41</sub>,b<sub>41</sub>&gt;/||b<sub>41</sub>||²/(D−R)</span>它表示 terminal state 沿 donor–receiver count gap 前进的比例。</dd>
</dl></div>
<p><strong>Window 的具体作用。</strong>L29H4 与 L35H2 位于 full-attention layers，能在 answer query 汇集完整 causal prefix；L30–L34、L36–L40 的 sliding layers不能直接重新看见大部分远端 needles，但 residual connection 会把此前已写入的 state 带入下一层，局部 attention/MLP 再变换它。因而 Gemma 路径不是“每层重新检索”，而是“full-attention 写入 → windowed residual 传播 → L41 terminal readout”。</p>
<p>表中的四个 candidate-core effects 均显著；完整判定还要求每个 endpoint 同时优于三个冻结 layer-matched K2 controls。family p 取候选效应与 candidate-minus-control specificity 中较弱者，global IUT p 再取四个 family p 的最大值；因此 global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])} 表示四步 conjunction 全部通过，而不是从多个 endpoint 中挑最小 p。</p>
<figure>{g_gate}<figcaption><strong>图 9C · Gemma source→residual→terminal conjunction。</strong>该流程图没有数值坐标轴；四个框按计算顺序展示 K2 source transport、删除完整 L37 induced residual、仅删除其 count-axis component，以及 L41 donor-count adoption。每框给出 candidate-core effect、95% CI 和包含 matched-specificity 的 family IUT p；global p 为四框中最大的 family p。</figcaption></figure>
<p class="result-sentence"><strong>结果。</strong>L29H4/L35H2 source patch 产生显著 donor transport；删除完整 δ<sub>37</sub> 或只删除其 count-aligned component 都会特异削弱 transport；L41 state 显著采用 donor count。四个 families 全部通过，global IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}。</p>
{table(["证据", "具体验证什么", "effect [95% CI]", "p"], g_rows, classes="paper-table compact-result-table")}
<div class="conclusion"><strong>Gemma 机制结论</strong>L29H4/L35H2 在 full-attention answer queries 聚合全 prompt，并经各自 W<sub>O</sub> 写回；写入形成 L37 的分布式、部分 count-aligned residual mediator。后续 sliding layers 不必重新访问远端 needles，而是让该 residual state 继续传播到 L41 answer computation。</div>
</article>
</section>
"""


def build_synthesis_clear(
    ov: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
) -> str:
    q_fs_top4 = _fullspan_upstream_row(
        fullspan_upstream, early_set="top4", route="slot_state"
    )
    q_fs_top4_p = max(
        float(q_fs_top4["early_donor_log_odds_gain_holm_p"]),
        float(q_fs_top4["donor_log_odds_mediation_specificity_holm_p"]),
    )
    return f"""
<section id="synthesis">
<h2>11 · 最终机制对照</h2>
{table(
    ["阶段", "Qwen3-8B", "Gemma4-E4B"],
    [
        ["Prompt state", "needle-end running counter", "needle-end running counter"],
        ["Retrieval", "full-span top-4 L27H18/L23H29/L23H13/L23H28", "full-span top-2 L29H4/L35H2"],
        ["Read / write", "L28 H16/H19 mixed α/V read + localized W<sub>O</sub> write", "K2 source-bank output writes L37 distributed residual"],
        ["Propagation", "L29–L35 answer-count axes", "L37 → L41 residual path"],
        ["Answer", "Total query residual drives N", "Total query residual drives N"],
        ["Causal conjunction", f"early→L28 Holm p={fmt_p(q_fs_top4_p)}；natural OV IUT p={fmt_p(ov['primary_decision']['global_intersection_union_p'])}", f"residual path IUT p={fmt_p(gemma_residual['primary_decision']['global_intersection_union_p'])}"],
    ],
    classes="paper-table compact-result-table",
)}
<p class="paper-wording"><strong>论文式表述。</strong>Both non-thinking models formed an ordered prompt-side running state, aggregated this state through distributed attention-head banks, and transformed it into an executable answer-query count representation. In Qwen3-8B, a localized L28 H16/H19 OV channel performed a mixed routing/value read and wrote the count signal into a downstream answer-residual direction. In Gemma4-E4B, a frozen L29H4/L35H2 bank causally wrote a count-aligned distributed residual state at L37 that propagated to the terminal answer computation.</p>
<div class="conclusion"><strong>最终结论</strong>两模型的共同算法是“累计状态 → 分布式读取 → 坐标变换/写入 → answer readout”；差异主要在 state 被因果定位的空间粒度，而不是是否存在 prompt running counter。</div>
</section>
"""


def build_limits_clear(
    causal_v2: dict[str, Any],
    seed_confirmation: dict[str, Any],
    ov: dict[str, Any],
    read_write: dict[str, Any],
    upstream: dict[str, Any],
    fullspan_upstream: dict[str, Any],
    gemma_residual: dict[str, Any],
    correct_state: dict[str, Any],
    token_audit: dict[str, Any],
    subspace_audit: dict[str, Any],
) -> str:
    rows = [
        ["Question specification", "reports/non-thinking extension.md", "registered question and implementation map; paper claims summarized in §10"],
        ["Deterministic report builder", "scripts/build_realistic_niah_v4_4_integrated_report.py", "self-contained HTML; no runtime network dependency"],
        ["Extension code map", "scripts/v4_4_extension/README_v4_4_extension.md", "GPU runners, CPU analyzers, launch commands, split definitions"],
        ["Macro representation / patching", "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json", causal_v2["schema_version"]],
        ["Full-span frozen top-k extrapolation", "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/seed_extrapolation_summary_v2.json", f"audit {seed_confirmation['audit']['passed']}/{seed_confirmation['audit']['checks']} PASS"],
        ["Full-span top-k primary statistics", "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/full_span_topk_primary_statistics.csv", "equal-seed estimand + pooled sensitivity"],
        ["Qwen natural OV", "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json", ov["schema_version"]],
        ["Qwen α/V read-write", "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json", read_write["schema_version"]],
        ["Qwen full-span early→L28 K-sweep", "reports/v4_non-thinking_causal/v4_4_4/qwen/full_span_upstream/realistic_niah_v4_4_4_upstream_path_analysis.json", f"audit {str(fullspan_upstream['audit']['all_checks_pass']).upper()} · 1080 effects"],
        ["Qwen fresh serial path", "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json", upstream["schema_version"]],
        ["Gemma K2 residual path", "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json", gemma_residual["schema_version"]],
        ["Fresh correct-only answer routes", "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json", correct_state["schema_version"]],
        ["Extension rank / regression / clustering", "reports/v4_non-thinking_causal/v4_4_extension/geometry/", "discovery-frozen geometry audit PASS"],
        ["All-token endpoint-gate controls", "reports/v4_non-thinking_causal/v4_4_extension/all_token/", "all-token audit PASS"],
        ["Fixed answer classifiers", "reports/v4_non-thinking_causal/v4_4_extension/classification/", "six fixed algorithms; grouped seed CV"],
        ["Qwen/Gemma earlier-span heads + endpoint attention mask", "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/", "Qwen/Gemma 10-head confirmation complete；Gemma 500/500 rows，audit PASS"],
        ["Gemma selected-row attention runner", "scripts/v4_4_extension/run_gemma_earlier_span_selected_rows.py", "same estimand as Qwen；reconstruct frozen single-query rows inside SDPA"],
        ["Needle token corruption", "reports/v4_non-thinking_causal/v4_4_extension/token_corruption/", html.escape(str(token_audit["status"]))],
        ["Set-wide prompt subspace ablation", "reports/v4_non-thinking_causal/v4_4_extension/prompt_subspace_ablation/", html.escape(str(subspace_audit["status"]))],
        ["Adjacent-layer transport-aligned confirmation", "work/v445_transport_aligned_confirmation/analysis/", "frozen rank-3 basis; exact seed tests"],
        ["Layerwise needle-end/answer-query rank-3 removal, transport, and map stability", "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/", "positions reported as separate estimands; discovery-frozen; confirmation seeds 1254–1263; all audits PASS"],
    ]
    return f"""
<section id="limits">
<h2>13 · 复现信息</h2>
<p>报告中的显著性均来自保存的 seed-level 聚合与审计文件；HTML 不复制 raw hidden states、full value tensors 或 raw attention rows。原始捕获继续保留在 FileStream。</p>
{details_table("Source ledger", ["component", "relative path", "schema/audit"], rows, opened=True)}
<div class="conclusion"><strong>复现结论</strong>所有进入正文的因果结果都对应 audit PASS 的机器可读 analysis；图形只负责展示 effect 与结构，不替代统计文件。</div>
</section>
"""


def validate_inputs(repo_root: Path) -> dict[str, Path]:
    paths = {
        "base": repo_root
        / "reports/v4_non-thinking_causal/v4_4_3/realistic_niah_v4_4_mechanism_report.html",
        "causal_v2": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/report_summary.json",
        "seed_confirmation": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/seed_extrapolation_summary.json",
        "full_span_topk": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/full_span_topk/seed_extrapolation_summary_v2.json",
        "exact_reanalysis": repo_root
        / "reports/v4_non-thinking_causal/v4_4_causal_v2/exact_sign_flip_reanalysis.json",
        "cue": repo_root
        / "reports/v4_non-thinking_causal/v4_4_2/realistic_niah_v4_4_2_mode_geometry_attention_report.html",
        "ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_analysis.json",
        "read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/read_write/realistic_niah_v4_4_4_read_write_analysis.json",
        "relay": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/realistic_niah_v4_4_4_relay_analysis.json",
        "upstream": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/upstream_confirmation/realistic_niah_v4_4_4_upstream_confirmation_analysis.json",
        "fullspan_upstream": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/qwen/full_span_upstream/realistic_niah_v4_4_4_upstream_path_analysis.json",
        "gemma_l37_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/natural_ov/realistic_niah_v4_4_4_analysis.json",
        "gemma_l29_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l29h4/realistic_niah_v4_4_4_analysis.json",
        "gemma_l35_ov": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l35h2/realistic_niah_v4_4_4_analysis.json",
        "gemma_l29_read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l29h4/realistic_niah_v4_4_4_read_write_analysis.json",
        "gemma_l35_read_write": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/search/l35h2/realistic_niah_v4_4_4_read_write_analysis.json",
        "gemma_cross_layer": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/cross_layer/realistic_niah_v4_4_4_cross_layer_analysis.json",
        "gemma_residual_k2": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k2/realistic_niah_v4_4_4_residual_analysis.json",
        "gemma_residual_k6": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/gemma/residual/k6/realistic_niah_v4_4_4_residual_analysis.json",
        "correct_state": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/correct_state_route_analysis.json",
        "correct_state_geometry": repo_root
        / "reports/v4_non-thinking_causal/v4_4_4/correct_state_routes/geometry_summary.csv",
        "transport_conditions": repo_root
        / "work/v445_transport_aligned_confirmation/analysis/condition_summary.csv",
        "transport_contrasts": repo_root
        / "work/v445_transport_aligned_confirmation/analysis/seed_contrasts.csv",
        "extension_rank": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/geometry/rank_and_compression_by_layer.csv",
        "extension_regression": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/geometry/count_regression_summary.csv",
        "extension_clustering": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/geometry/clustering_summary.csv",
        "extension_noise": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/geometry/prompt_noise_two_way_decomposition.csv",
        "extension_all_token_metrics": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/all_token/all_token_category_metrics.csv",
        "extension_formula": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/all_token/gated_curve_formula_tests.csv",
        "extension_projection": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/all_token/all_token_frozen_pca_projections.csv.gz",
        "extension_classifier_prompt_qwen": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_prompt_qwen/answer_classifier_metrics.csv",
        "extension_classifier_prompt_gemma": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_prompt_gemma/answer_classifier_metrics.csv",
        "extension_classifier_answer_qwen": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_all_qwen/answer_classifier_metrics.csv",
        "extension_classifier_answer_gemma": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_all_gemma/answer_classifier_metrics.csv",
        "extension_classifier_correct_qwen": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_correct_qwen/answer_classifier_metrics.csv",
        "extension_classifier_correct_gemma": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/classification/classification_correct_gemma/answer_classifier_metrics.csv",
        "extension_attention_stats": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/attention_mask_statistics.csv",
        "extension_earlier_heads": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/earlier_span_head_confirmation.csv",
        "extension_gemma_earlier_heads": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/gemma_earlier_span_head_confirmation.csv",
        "extension_gemma_earlier_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/gemma_earlier_span_audit.json",
        "extension_attention_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/endpoint_attention_mask/endpoint_attention_mask_analysis_audit.json",
        "extension_token_stats": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/token_corruption/token_corruption_statistics.csv",
        "extension_token_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/token_corruption/token_corruption_analysis_audit.json",
        "extension_subspace_stats": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/prompt_subspace_ablation/subspace_ablation_statistics.csv",
        "extension_subspace_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/prompt_subspace_ablation/analysis_audit.json",
        "layerwise_prompt_statistics": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/prompt_removal/layerwise_prompt_removal_statistics.csv",
        "layerwise_prompt_trends": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/prompt_removal/layerwise_prompt_removal_depth_trends.csv",
        "layerwise_prompt_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/prompt_removal/analysis_audit.json",
        "layerwise_answer_statistics": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/answer_query_removal/layerwise_answer_query_removal_statistics.csv",
        "layerwise_answer_damage_statistics": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/answer_query_removal/layerwise_answer_query_removal_damage_statistics.csv",
        "layerwise_answer_trends": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/answer_query_removal/layerwise_answer_query_removal_depth_trends.csv",
        "layerwise_answer_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/answer_query_removal/analysis_audit.json",
        "layerwise_transport_statistics": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/transport/layerwise_transport_statistics.csv",
        "layerwise_transport_trends": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/transport/layerwise_transport_depth_trends.csv",
        "layerwise_transport_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/transport/analysis_audit.json",
        "layerwise_map_summary": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/layer_maps/layerwise_linear_map_summary.csv",
        "layerwise_map_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/layer_maps/analysis_audit.json",
        "layerwise_registered_stability": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/map_causal_link/registered_boundary_stability.csv",
        "layerwise_map_causal_tests": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/map_causal_link/stable_minus_unstable_tests.csv",
        "layerwise_map_causal_correlations": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/map_causal_link/boundary_spearman_descriptive.csv",
        "layerwise_map_causal_audit": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/layerwise_subspace/map_causal_link/analysis_audit.json",
        "first_locator_qwen_scores": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator/Qwen3-8B.all_head_first_span_scores.csv",
        "first_locator_gemma_scores": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator/Gemma4-E4B.all_head_first_span_scores.csv",
        "first_locator_layer_matched": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator/first_span_ablation_statistics.csv",
        "first_locator_m10_matched": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator/first_span_M10_matched_statistics.csv",
        "first_locator_matching": repo_root
        / "reports/v4_non-thinking_causal/v4_4_extension/first_span_locator/first_span_M10_matching_summary.csv",
    }
    required = {
        "base",
        "causal_v2",
        "seed_confirmation",
        "full_span_topk",
        "exact_reanalysis",
        "cue",
        "ov",
        "read_write",
        "relay",
        "upstream",
        "fullspan_upstream",
        "gemma_l37_ov",
        "correct_state",
        "correct_state_geometry",
        "transport_conditions",
        "transport_contrasts",
        "extension_rank",
        "extension_regression",
        "extension_clustering",
        "extension_noise",
        "extension_all_token_metrics",
        "extension_formula",
        "extension_projection",
        "extension_classifier_prompt_qwen",
        "extension_classifier_prompt_gemma",
        "extension_classifier_answer_qwen",
        "extension_classifier_answer_gemma",
        "extension_classifier_correct_qwen",
        "extension_classifier_correct_gemma",
        "extension_attention_stats",
        "extension_earlier_heads",
        "extension_gemma_earlier_heads",
        "extension_gemma_earlier_audit",
        "extension_attention_audit",
        "extension_token_stats",
        "extension_token_audit",
        "extension_subspace_stats",
        "extension_subspace_audit",
        "layerwise_prompt_statistics",
        "layerwise_prompt_trends",
        "layerwise_prompt_audit",
        "layerwise_answer_statistics",
        "layerwise_answer_damage_statistics",
        "layerwise_answer_trends",
        "layerwise_answer_audit",
        "layerwise_transport_statistics",
        "layerwise_transport_trends",
        "layerwise_transport_audit",
        "layerwise_map_summary",
        "layerwise_map_audit",
        "layerwise_registered_stability",
        "layerwise_map_causal_tests",
        "layerwise_map_causal_correlations",
        "layerwise_map_causal_audit",
        "first_locator_qwen_scores",
        "first_locator_gemma_scores",
        "first_locator_layer_matched",
        "first_locator_m10_matched",
        "first_locator_matching",
    }
    missing = [
        str(paths[name]) for name in sorted(required) if not paths[name].is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing integrated-report inputs: {missing}")
    return paths


def replace_section(document: str, section_id: str, replacement: str) -> str:
    pattern = re.compile(rf'<section id="{re.escape(section_id)}">.*?</section>', re.S)
    updated, count = pattern.subn(replacement.strip(), document, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one section #{section_id}; replaced {count}")
    return updated


def build_report(repo_root: Path, output: Path) -> None:
    paths = validate_inputs(repo_root)
    base = paths["base"].read_text(encoding="utf-8")
    answer_data = extract_embedded_json(base, "ANSWER_DATA")
    causal_v2 = read_json(paths["causal_v2"])
    seed_confirmation = read_json(paths["seed_confirmation"])
    exact_reanalysis = read_json(paths["exact_reanalysis"])
    cue_doc = paths["cue"].read_text(encoding="utf-8")
    ov = read_json(paths["ov"])
    read_write = read_json(paths["read_write"])
    relay = read_json(paths["relay"])
    upstream = read_json(paths["upstream"])
    gemma_l37_ov = read_json(paths["gemma_l37_ov"])
    gemma_singles = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("l29h4", "gemma_l29_ov"),
            ("l35h2", "gemma_l35_ov"),
        )
        if paths[path_key].is_file()
    }
    gemma_read_writes = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("l29h4", "gemma_l29_read_write"),
            ("l35h2", "gemma_l35_read_write"),
        )
        if paths[path_key].is_file()
    }
    gemma_cross_layer = (
        read_json(paths["gemma_cross_layer"])
        if paths["gemma_cross_layer"].is_file()
        else None
    )
    gemma_residuals = {
        name: read_json(paths[path_key])
        for name, path_key in (
            ("k2", "gemma_residual_k2"),
            ("k6", "gemma_residual_k6"),
        )
        if paths[path_key].is_file()
    }
    correct_state = read_json(paths["correct_state"])
    transport_conditions = read_csv_rows(paths["transport_conditions"])
    transport_contrasts = read_csv_rows(paths["transport_contrasts"])
    extension_rank = read_csv_rows(paths["extension_rank"])
    extension_regression = read_csv_rows(paths["extension_regression"])
    extension_clustering = read_csv_rows(paths["extension_clustering"])
    extension_noise = read_csv_rows(paths["extension_noise"])
    extension_all_token_metrics = read_csv_rows(paths["extension_all_token_metrics"])
    extension_formula = read_csv_rows(paths["extension_formula"])
    extension_projection = read_csv_rows_gzip(paths["extension_projection"])
    extension_classifier: list[dict[str, str]] = []
    for key in (
        "extension_classifier_prompt_qwen",
        "extension_classifier_prompt_gemma",
        "extension_classifier_answer_qwen",
        "extension_classifier_answer_gemma",
    ):
        extension_classifier.extend(read_csv_rows(paths[key]))
    extension_classifier_correct: list[dict[str, str]] = []
    for key in (
        "extension_classifier_correct_qwen",
        "extension_classifier_correct_gemma",
    ):
        extension_classifier_correct.extend(read_csv_rows(paths[key]))
    extension_attention_stats = read_csv_rows(paths["extension_attention_stats"])
    extension_earlier_heads = read_csv_rows(paths["extension_earlier_heads"])
    extension_earlier_heads.extend(
        read_csv_rows(paths["extension_gemma_earlier_heads"])
    )
    extension_gemma_earlier_audit = read_json(
        paths["extension_gemma_earlier_audit"]
    )
    extension_attention_audit = read_json(paths["extension_attention_audit"])
    if extension_gemma_earlier_audit.get("status") != "PASS":
        raise RuntimeError("Gemma earlier-span confirmation audit did not pass")
    if int(extension_gemma_earlier_audit.get("observed_raw_rows", -1)) != 500:
        raise RuntimeError("Gemma earlier-span confirmation is incomplete")
    extension_token_stats = read_csv_rows(paths["extension_token_stats"])
    extension_token_audit = read_json(paths["extension_token_audit"])
    extension_subspace_stats = read_csv_rows(paths["extension_subspace_stats"])
    extension_subspace_audit = read_json(paths["extension_subspace_audit"])
    cue_doc = paths["cue"].read_text(encoding="utf-8")
    correct_state_geometry = read_csv_rows(paths["correct_state_geometry"])
    gemma_story = resolve_gemma_story(
        l37=gemma_l37_ov,
        singles=gemma_singles,
        read_writes=gemma_read_writes,
        cross_layer=gemma_cross_layer,
        residuals=gemma_residuals,
    )

    if int(exact_reanalysis["method"]["assignments_enumerated"]) != 2**20:
        raise RuntimeError(
            "Correct-only exact reanalysis did not enumerate all 2^20 assignments"
        )
    exact_rows = {
        (str(row["model"]), str(row["top_k"])): row
        for row in exact_reanalysis["results"]
    }
    for model, model_rows in seed_confirmation["models"].items():
        for k_text, metrics in model_rows.items():
            audit_row = exact_rows[(str(model), str(k_text))]
            # The original summary used deterministic Monte Carlo for n=20 but
            # retained an ``exact`` field name.  The separately audited full
            # 2^20 enumeration is authoritative and is injected into the
            # in-memory report payload without mutating the archived source.
            metrics["clean_correct_to_wrong"]["two_sided_exact_seed_sign_flip_p"] = (
                float(audit_row["clean_correct_failure"]["exact_p"])
            )
            metrics["clean_correct_to_wrong"]["holm_p_across_four_frozen_sets"] = float(
                audit_row["clean_correct_failure"]["holm_p"]
            )
            metrics["absolute_error"]["exact_p"] = float(
                audit_row["absolute_error"]["exact_p"]
            )
            metrics["absolute_error"]["holm_p_across_four_frozen_sets"] = float(
                audit_row["absolute_error"]["holm_p"]
            )
    confirmation_order = [
        ("Qwen3-8B", "2"),
        ("Qwen3-8B", "4"),
        ("Gemma4-E4B", "1"),
        ("Gemma4-E4B", "2"),
    ]
    absolute_error_holm = holm_adjusted_pvalues(
        [
            float(
                seed_confirmation["models"][model][k_text]["absolute_error"]["exact_p"]
            )
            for model, k_text in confirmation_order
        ]
    )
    for (model, k_text), adjusted_p in zip(
        confirmation_order, absolute_error_holm, strict=True
    ):
        if not math.isclose(
            adjusted_p,
            float(exact_rows[(model, k_text)]["absolute_error"]["holm_p"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"Absolute-error Holm-p mismatch for {model} K={k_text}")

    if not ov["audit"]["all_checks_pass"]:
        raise RuntimeError("Natural-OV audit did not pass")
    if not read_write["audit"]["all_checks_pass"]:
        raise RuntimeError("Read/write audit did not pass")
    if not relay["audit"]["all_checks_pass"]:
        raise RuntimeError("Relay audit did not pass")
    if not upstream["audit"]["all_checks_pass"]:
        raise RuntimeError("Upstream-confirmation audit did not pass")
    gemma_audits: list[tuple[str, dict[str, Any]]] = [
        ("L37 natural-OV", gemma_l37_ov),
        *((f"{name} natural-OV", doc) for name, doc in gemma_singles.items()),
        *((f"{name} read/write", doc) for name, doc in gemma_read_writes.items()),
    ]
    if gemma_cross_layer is not None:
        gemma_audits.append(("cross-layer", gemma_cross_layer))
    gemma_audits.extend(
        (f"{name} residual", document) for name, document in gemma_residuals.items()
    )
    for label, document in gemma_audits:
        if not document.get("audit", {}).get("all_checks_pass", False):
            raise RuntimeError(f"Gemma {label} audit did not pass")
    if not correct_state.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only state-route audit did not pass")

    base = re.sub(
        r"<title>.*?</title>",
        "<title>Realistic NIAH V4.4 · non-thinking integrated mechanism</title>",
        base,
        count=1,
    )
    base = ensure_viewport_meta(base)
    if "</style>" not in base:
        raise RuntimeError("Base report has no style terminator")
    base = base.replace("</style>", EXTRA_CSS + "\n</style>", 1)
    nav = """<nav><a href="#mechanism-overview">Main figure</a><a href="#scope">结论</a><a href="#methods">设定/定义</a><a href="#prompt">Prompt geometry</a><a href="#cue-robustness">Cue robustness</a><a href="#answer">Answer geometry</a><a href="#attention">Attention</a><a href="#causal">Causal design</a><a href="#natural-ov">Natural OV</a><a href="#read-write">Read/write</a><a href="#upstream">Serial path</a><a href="#synthesis">Synthesis</a><a href="#limits">边界</a></nav>"""
    base, nav_count = re.subn(r"<nav>.*?</nav>", nav, base, count=1, flags=re.S)
    if nav_count != 1:
        raise RuntimeError("Could not replace report navigation")
    base = base.replace(
        "</nav>",
        '<a href="#representation-extension">Representation tests</a></nav>',
        1,
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""<header>
<div class="eyebrow">Realistic NIAH · V4.4 · non-thinking · integrated evidence</div>
<h1>从 running-index representation 到自然 read–write causal circuit</h1>
<p class="lead">一份统一的 representation → retrieval → state transport → causal write/relay 报告。Qwen3-8B 与 Gemma4-E4B 使用相同 estimands、各自冻结的候选与独立 seed 外推；Gemma 的定位粒度由顺序证据阶梯中实际通过的最强 conjunction 决定，而不是强迫复制 Qwen 的 layer/head identity。</p>
<p class="meta">generated {generated} · source campaigns V4.4 / V4.4.2 / causal-v2 / V4.4.4 · self-contained HTML</p>
</header>"""
    base, header_count = re.subn(
        r"<header>.*?</header>", header, base, count=1, flags=re.S
    )
    if header_count != 1:
        raise RuntimeError("Could not replace report header")

    base = replace_section(
        base,
        "scope",
        build_scope(
            causal_v2,
            ov,
            read_write,
            relay,
            upstream,
            gemma_l37_ov,
            gemma_story,
        ),
    )
    overview = build_mechanism_overview(ov, read_write, upstream, gemma_story)
    base = base.replace(
        '<section id="scope">', overview + '\n\n<section id="scope">', 1
    )
    methods = build_methods(
        ov,
        upstream,
        gemma_l37_ov,
        gemma_singles,
        gemma_read_writes,
        gemma_cross_layer,
        gemma_residuals,
    )
    base = base.replace(
        '<section id="prompt">', methods + '\n\n<section id="prompt">', 1
    )
    base = base.replace(
        "<h2>1 · Prompt-reading counter representation</h2>",
        "<h2>3 · Prompt-reading counter representation</h2>",
        1,
    )
    running_block = build_running_index_block()
    prompt_marker = (
        '<div class="figure-block"><h3>1.1 Interactive V4.4 prompt counter</h3>'
    )
    if prompt_marker not in base:
        raise RuntimeError("Could not locate prompt 3D figure block")
    base = base.replace(
        prompt_marker,
        running_block
        + '\n\n<div class="figure-block"><h3>3.2 Seed-level prompt counter · 完整交互</h3>',
        1,
    )
    cue_section = build_cue_section(cue_doc)
    base = base.replace(
        '<section id="answer">', cue_section + '\n\n<section id="answer">', 1
    )
    base = base.replace(
        "<h2>2 · Answer-query counter representation</h2>",
        "<h2>5 · Answer-query counter representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>2.1 Interactive V4.4 answer-query counter</h3>",
        "<h3>5.2 Interactive V4.4 answer-query counter</h3>",
        1,
    )
    base = base.replace(
        "<h3>2.2 Prompt 与 answer counter 的共同坐标</h3>",
        "<h3>5.2 Prompt 与 answer counter 的共同坐标</h3>",
        1,
    )
    base = base.replace("<h3>5.2 Prompt ", "<h3>5.3 Prompt ", 1)
    answer_marker = (
        '<div class="figure-block"><h3>5.2 Interactive V4.4 answer-query counter</h3>'
    )
    if answer_marker not in base:
        raise RuntimeError("Could not locate answer-query 3D figure block")
    base = base.replace(
        answer_marker,
        build_answer_fit_sensitivity(answer_data) + "\n\n" + answer_marker,
        1,
    )
    base = base.replace(
        "<h2>3 · V4.4 attention-head representation</h2>",
        "<h2>6 · V4.4 attention-head representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>3.1 All-head V4.4 atlas</h3>", "<h3>6.1 All-head V4.4 atlas</h3>", 1
    )
    causal_header = (
        '<section id="causal">\n<h2>7 · 因果证据：设计、聚合与宏观定位</h2>\n'
        + build_causal_design(
            ov,
            upstream,
            seed_confirmation,
            gemma_l37_ov,
            gemma_singles,
            gemma_read_writes,
            gemma_cross_layer,
            gemma_residuals,
        )
        + build_causal_v2_intro(causal_v2, seed_confirmation)
    )
    base, causal_count = re.subn(
        r'<section id="causal">\s*<h2>.*?</h2>',
        causal_header,
        base,
        count=1,
        flags=re.S,
    )
    if causal_count != 1:
        raise RuntimeError("Could not update causal section")
    # Renumber only the legacy causal subsections.  The cue-robustness section
    # that is inserted above also has 4.1/4.2 headings; a document-wide replace
    # would rename those first and then let the ablation regex backtrack across
    # the answer-query figure, deleting its canvas.
    causal_section_pattern = re.compile(
        r'(<section id="causal">)(.*?)(</section>)', re.S
    )

    def renumber_causal_subsections(match: re.Match[str]) -> str:
        body = match.group(2)
        for old, new in (
            ("4.1", "7.4"),
            ("4.2", "7.5"),
            ("4.3", "7.6"),
            ("4.4", "7.7"),
            ("4.5", "7.8"),
        ):
            body = body.replace(f"<h3>{old} ", f"<h3>{new} ", 1)
        return match.group(1) + body + match.group(3)

    base, causal_renumber_count = causal_section_pattern.subn(
        renumber_causal_subsections, base, count=1
    )
    if causal_renumber_count != 1:
        raise RuntimeError("Could not isolate causal section for subsection renumbering")
    ablation_pattern = re.compile(
        r'(<h3>7\.4 [^<]*</h3>)\s*<p class="figure-intro">.*?</p>\s*<figure>.*?</figure>',
        re.S,
    )
    ablation_replacement = r"""\1
<div class="callout warning"><strong>旧高-count screen 的定位。</strong>下表保留原 V4.4 count 7–10、K=4/8 screen 作为历史敏感性分析；它没有 correct-only eligibility，也没有本轮冻结 K 的独立 seed 外推，因此不再承担主 head-bank necessity 结论。主图与主统计见上方 7.3。</div>"""
    base, ablation_count = ablation_pattern.subn(ablation_replacement, base, count=1)
    if ablation_count != 1:
        raise RuntimeError("Could not replace top-k ablation figure")

    natural_appendices = [
        build_gemma_evidence_ladder(
            l37=gemma_l37_ov,
            singles=gemma_singles,
            cross_layer=gemma_cross_layer,
            residuals=gemma_residuals,
            story=gemma_story,
        ),
        build_gemma_natural_ov_appendix(
            gemma_l37_ov,
            heading="8.5",
            context_label="最初冻结且完整保留的负结果",
        ),
    ]
    for index, (name, document) in enumerate(gemma_singles.items(), start=6):
        natural_appendices.append(
            build_gemma_natural_ov_appendix(
                document,
                heading=f"8.{index}",
                context_label=f"independent-ablation candidate {name}",
            )
        )
    if gemma_cross_layer is not None:
        natural_appendices.append(
            build_gemma_natural_ov_appendix(
                gemma_cross_layer,
                heading=f"8.{6 + len(gemma_singles)}",
                context_label="条件式跨层 K2 fallback",
            )
        )
    natural_section = append_to_section(
        build_natural_ov_section(ov), "\n".join(natural_appendices)
    )

    read_write_appendices: list[str] = []
    for index, (name, document) in enumerate(gemma_read_writes.items(), start=3):
        parent = gemma_singles.get(name)
        if parent is None:
            raise RuntimeError(
                f"Gemma read/write has no parent natural-OV result: {name}"
            )
        read_write_appendices.append(
            build_gemma_read_write_appendix(
                document,
                parent,
                heading=f"9.{index}",
                natural_heading=(f"8.{6 + list(gemma_singles).index(name)}"),
            )
        )
    read_write_section = build_read_write_section(read_write)
    if read_write_appendices:
        read_write_section = append_to_section(
            read_write_section, "\n".join(read_write_appendices)
        )

    upstream_appendices: list[str] = []
    if gemma_cross_layer is not None:
        upstream_appendices.append(build_gemma_cross_layer_appendix(gemma_cross_layer))
    upstream_appendices.extend(
        build_gemma_residual_appendix(document) for document in gemma_residuals.values()
    )
    upstream_section = build_upstream_section(relay, upstream)
    if upstream_appendices:
        upstream_section = append_to_section(
            upstream_section, "\n".join(upstream_appendices)
        )
    synthesis_section = append_to_section(
        build_synthesis_section(),
        "\n".join(
            [
                build_gemma_synthesis_ladder(
                    l37=gemma_l37_ov,
                    singles=gemma_singles,
                    read_writes=gemma_read_writes,
                    cross_layer=gemma_cross_layer,
                    residuals=gemma_residuals,
                    story=gemma_story,
                ),
                build_correct_state_boundary(correct_state, correct_state_geometry),
            ]
        ),
    )
    additions = "\n\n".join(
        [natural_section, read_write_section, upstream_section, synthesis_section]
    )
    base = base.replace(
        '<section id="limits">', additions + '\n\n<section id="limits">', 1
    )
    base = replace_section(
        base,
        "limits",
        build_limits_dynamic(
            causal_v2=causal_v2,
            seed_confirmation=seed_confirmation,
            ov=ov,
            read_write=read_write,
            relay=relay,
            upstream=upstream,
            gemma_l37=gemma_l37_ov,
            gemma_singles=gemma_singles,
            gemma_read_writes=gemma_read_writes,
            gemma_cross_layer=gemma_cross_layer,
            gemma_residuals=gemma_residuals,
            gemma_story=gemma_story,
            correct_state=correct_state,
        ),
    )

    if "function makeProjector" not in base:
        raise RuntimeError("Could not locate embedded visualization script")
    base = base.replace(
        "function makeProjector", EXTRA_JS + "\nfunction makeProjector", 1
    )
    old_boot = "makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    new_boot = "makeMechanismWalkthrough();makeRunningIndex();makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    if old_boot not in base:
        raise RuntimeError("Could not locate visualization bootstrap")
    base = base.replace(old_boot, new_boot, 1)

    # Normalize the legacy report's model accents to the user-specified Aurora
    # palette.  The final CSS already controls surface tints and typography;
    # these replacements also cover colors embedded directly in older SVGs.
    aurora_legacy_map = {
        "#27685F": "#6750E8",  # legacy Qwen green -> Polar Violet
        "#A66A45": "#00D4B4",  # legacy Gemma brown -> Aurora Teal
        "#6A958A": "#6750E8",
        "#B78767": "#00D4B4",
        "#588BD2": "#00C2FF",
    }
    for old_color, new_color in aurora_legacy_map.items():
        base = re.sub(re.escape(old_color), new_color, base, flags=re.I)

    base = insert_concrete_examples(base)
    base = make_all_tables_collapsible(base)
    base = merge_transport_into_section_5_4(base)

    required_sections = [
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "cue-robustness",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "read-write",
        "upstream",
        "synthesis",
        "limits",
    ]
    for section_id in required_sections:
        if base.count(f'id="{section_id}"') != 1:
            raise RuntimeError(f"Section id count is not one: {section_id}")
    for canvas_id in (
        "running-index-canvas",
        "prompt-canvas",
        "answer-canvas",
        "joint-canvas",
    ):
        if base.count(f'id="{canvas_id}"') != 1:
            raise RuntimeError(f"Interactive canvas id count is not one: {canvas_id}")
    for heading in (
        "<h3>4.1 Hidden-state geometry</h3>",
        "<h3>4.2 Attention map：同一 broad-retrieval score 的左右对照</h3>",
        "<h3>7.4 Head ablation · mixed ranked bank 是否比 layer-matched random 更重要？</h3>",
        "<h3>7.5 Needle-end patching · 单个 toggled endpoint state 是否足以运输 count increment？</h3>",
    ):
        if base.count(heading) != 1:
            raise RuntimeError(f"Expected one renumbered report heading: {heading}")
    if len(re.findall(r"<figcaption\b", base)) != base.count("</figcaption>"):
        raise RuntimeError("Unbalanced figure captions")
    if base.count("<section") != base.count("</section>"):
        raise RuntimeError("Unbalanced sections")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(base, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sections": required_sections,
                "figures": len(re.findall(r"<figure\b", base)),
                "figcaptions": len(re.findall(r"<figcaption\b", base)),
                "conclusion_boxes": base.count('<div class="conclusion">'),
                "natural_ov_global_iut_p": ov["primary_decision"][
                    "global_intersection_union_p"
                ],
                "upstream_global_iut_p": upstream["primary_decision"][
                    "intersection_union_p"
                ],
                "gemma_strongest_kind": gemma_story["kind"],
                "gemma_strongest_global_iut_p": gemma_story.get("global_p"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


MECHANISM_DETAILED_JS = r"""
function makeMechanismWalkthrough(){
 const root=document.getElementById('mechanism-paper-figure');if(!root)return;
 const nodes=[...root.querySelectorAll('[data-mechanism-stage]')];
 const edges=[...root.querySelectorAll('[data-mechanism-edge]')];
 const dots=[...root.querySelectorAll('[data-mechanism-step]')];
 const prev=document.getElementById('mechanism-prev');
 const next=document.getElementById('mechanism-next');
 const play=document.getElementById('mechanism-play');
 const qTitle=document.getElementById('mechanism-live-q-title');
 const qBody=document.getElementById('mechanism-live-q-body');
 const gTitle=document.getElementById('mechanism-live-g-title');
 const gBody=document.getElementById('mechanism-live-g-body');
 if(!prev||!next||!play||!qTitle||!qBody||!gTitle||!gBody)return;
 const stages=[
  {qTitle:'1/5 · Qwen 提取 prompt running state',qBody:'在同一 N=10 prompt 内，于第 n 个 active needle 的最后 token 保存 post-block residual；n=1…10 是读取进度。',gTitle:'1/5 · Gemma 提取 prompt running state',gBody:'使用相同 endpoint 定义；sliding layers 的 state 可以同时包含局部更新、前序 full-layer 刷新与 residual/MLP 变换。'},
  {qTitle:'2/5 · Qwen full-span early top-4',qBody:'按完整 needle spans 冻结 L27H18/L23H29/L23H13/L23H28，并只在注册 slot-state positions 替换 donor pre-O z；随后在 L28 H16/H19 阻断 induced natural-OV component。',gTitle:'2/5 · Gemma full-attention K2 bank',gBody:'L29H4 与 L35H2 位于 full-attention layers，可在 answer query 汇集整个 causal prefix；三个 layer-matched K2 sets 作为对照。'},
  {qTitle:'3/5 · Qwen α/V mixed read',qBody:'在 L28 H16/H19 构造 RR、RD、DR、DD 四个 pre-O endpoints，用 Shapley identity 把 donor-z movement 分成 routing 与 value 两部分。',gTitle:'3/5 · Gemma true pre-O source patch',gBody:'只替换 answer-query 的 z(29,4) 与 z(35,2)；写入必须经过这两个 heads 自己的 W_O，receiver 其他状态保持不变。'},
  {qTitle:'4/5 · Qwen natural OV write',qBody:'在 W_O 前注入或删除 natural V-path count step，并与同 W_O span、等 post-O 范数的正交方向比较；影响沿 L29–L35 frozen count axes 追踪。',gTitle:'4/5 · Gemma L37 residual mediator',gBody:'测量 K2 patch 在 L37 诱发的 δ；exact block 删除整个 δ，count-axis block 只删除其 count-aligned component，再追踪至 L41。'},
  {qTitle:'5/5 · Qwen answer state 执行输出',qBody:'把 donor 的 Total: query full residual 单层替换给 receiver，再从 receiver context 完整 greedy 生成；输出显著向 donor count 移动。',gTitle:'5/5 · Gemma terminal state 执行输出',gBody:'L41 frozen count-axis adoption 与独立 full-state answer patch 共同显示：窗口化传播后的 terminal query state 可以改变最终数字。'}
 ];
 let step=0,timer=null;
 function stop(){if(timer){clearInterval(timer);timer=null}play.textContent='▶ 播放一次';play.setAttribute('aria-pressed','false')}
 function render(){
  nodes.forEach(node=>{const i=Number(node.dataset.mechanismStage);node.classList.toggle('is-active',i===step);node.classList.toggle('is-complete',i<step)});
  edges.forEach(edge=>{const i=Number(edge.dataset.mechanismEdge);edge.classList.toggle('is-active',i===step);edge.classList.toggle('is-complete',i<step)});
  dots.forEach(dot=>dot.setAttribute('aria-current',Number(dot.dataset.mechanismStep)===step?'step':'false'));
  prev.disabled=step===0;next.disabled=step===stages.length-1;
  qTitle.textContent=stages[step].qTitle;qBody.textContent=stages[step].qBody;
  gTitle.textContent=stages[step].gTitle;gBody.textContent=stages[step].gBody;
 }
 prev.addEventListener('click',()=>{stop();step=Math.max(0,step-1);render()});
 next.addEventListener('click',()=>{stop();step=Math.min(stages.length-1,step+1);render()});
 dots.forEach(dot=>dot.addEventListener('click',()=>{stop();step=Number(dot.dataset.mechanismStep);render()}));
 play.addEventListener('click',()=>{if(timer){stop();return}if(step===stages.length-1)step=0;render();play.textContent='Ⅱ 暂停';play.setAttribute('aria-pressed','true');timer=setInterval(()=>{if(step>=stages.length-1){stop();return}step+=1;render()},1600)});
 render();
}
"""


def build_report_clear(repo_root: Path, output: Path) -> None:
    paths = validate_inputs(repo_root)
    base = paths["base"].read_text(encoding="utf-8")
    base = ensure_first_locator_appendix(base, repo_root)
    base = remove_legacy_endpoint_phenotype_table(base)
    base = remove_first_locator_body_claims(base)
    answer_data = extract_embedded_json(base, "ANSWER_DATA")
    causal_v2 = read_json(paths["causal_v2"])
    seed_confirmation = read_json(paths["full_span_topk"])
    ov = read_json(paths["ov"])
    read_write = read_json(paths["read_write"])
    upstream = read_json(paths["upstream"])
    fullspan_upstream = read_json(paths["fullspan_upstream"])
    gemma_residual = read_json(paths["gemma_residual_k2"])
    correct_state = read_json(paths["correct_state"])
    transport_conditions = read_csv_rows(paths["transport_conditions"])
    transport_contrasts = read_csv_rows(paths["transport_contrasts"])
    extension_rank = read_csv_rows(paths["extension_rank"])
    extension_regression = read_csv_rows(paths["extension_regression"])
    extension_clustering = read_csv_rows(paths["extension_clustering"])
    extension_noise = read_csv_rows(paths["extension_noise"])
    extension_all_token_metrics = read_csv_rows(paths["extension_all_token_metrics"])
    extension_formula = read_csv_rows(paths["extension_formula"])
    extension_projection = read_csv_rows_gzip(paths["extension_projection"])
    extension_classifier: list[dict[str, str]] = []
    for key in (
        "extension_classifier_prompt_qwen",
        "extension_classifier_prompt_gemma",
        "extension_classifier_answer_qwen",
        "extension_classifier_answer_gemma",
    ):
        extension_classifier.extend(read_csv_rows(paths[key]))
    extension_classifier_correct: list[dict[str, str]] = []
    for key in (
        "extension_classifier_correct_qwen",
        "extension_classifier_correct_gemma",
    ):
        extension_classifier_correct.extend(read_csv_rows(paths[key]))
    extension_attention_stats = read_csv_rows(paths["extension_attention_stats"])
    extension_earlier_heads = read_csv_rows(paths["extension_earlier_heads"])
    extension_earlier_heads.extend(
        read_csv_rows(paths["extension_gemma_earlier_heads"])
    )
    extension_gemma_earlier_audit = read_json(
        paths["extension_gemma_earlier_audit"]
    )
    extension_attention_audit = read_json(paths["extension_attention_audit"])
    if extension_gemma_earlier_audit.get("status") != "PASS":
        raise RuntimeError("Gemma earlier-span confirmation audit did not pass")
    if int(extension_gemma_earlier_audit.get("observed_raw_rows", -1)) != 500:
        raise RuntimeError("Gemma earlier-span confirmation is incomplete")
    extension_token_stats = read_csv_rows(paths["extension_token_stats"])
    extension_token_audit = read_json(paths["extension_token_audit"])
    extension_subspace_stats = read_csv_rows(paths["extension_subspace_stats"])
    extension_subspace_audit = read_json(paths["extension_subspace_audit"])
    layerwise_prompt_statistics = read_csv_rows(paths["layerwise_prompt_statistics"])
    layerwise_prompt_trends = read_csv_rows(paths["layerwise_prompt_trends"])
    layerwise_prompt_audit = read_json(paths["layerwise_prompt_audit"])
    layerwise_answer_statistics = read_csv_rows(paths["layerwise_answer_statistics"])
    layerwise_answer_damage_statistics = read_csv_rows(
        paths["layerwise_answer_damage_statistics"]
    )
    layerwise_answer_trends = read_csv_rows(paths["layerwise_answer_trends"])
    layerwise_answer_audit = read_json(paths["layerwise_answer_audit"])
    layerwise_transport_statistics = read_csv_rows(
        paths["layerwise_transport_statistics"]
    )
    layerwise_transport_trends = read_csv_rows(paths["layerwise_transport_trends"])
    layerwise_transport_audit = read_json(paths["layerwise_transport_audit"])
    layerwise_map_summary = read_csv_rows(paths["layerwise_map_summary"])
    layerwise_map_audit = read_json(paths["layerwise_map_audit"])
    layerwise_registered_stability = read_csv_rows(
        paths["layerwise_registered_stability"]
    )
    layerwise_map_causal_tests = read_csv_rows(paths["layerwise_map_causal_tests"])
    layerwise_map_causal_correlations = read_csv_rows(
        paths["layerwise_map_causal_correlations"]
    )
    layerwise_map_causal_audit = read_json(paths["layerwise_map_causal_audit"])
    cue_doc = paths["cue"].read_text(encoding="utf-8")

    if any(
        causal_v2["audits"][model]["status"] != "PASS"
        for model in ("Qwen3-8B", "Gemma4-E4B")
    ):
        raise RuntimeError("causal-v2 audit did not pass")
    if str(seed_confirmation["audit"]["status"]).lower() != "passed":
        raise RuntimeError("Frozen top-k audit did not pass")
    for label, document in (
        ("Qwen natural OV", ov),
        ("Qwen read/write", read_write),
        ("Qwen upstream", upstream),
        ("Qwen full-span upstream", fullspan_upstream),
        ("Gemma residual", gemma_residual),
    ):
        if not document.get("audit", {}).get("all_checks_pass", False):
            raise RuntimeError(f"{label} audit did not pass")
    if not correct_state.get("audits", {}).get("all_checks_pass", False):
        raise RuntimeError("Correct-only route audit did not pass")
    for label, audit in (
        ("layerwise needle-end prompt removal", layerwise_prompt_audit),
        ("layerwise answer-query removal", layerwise_answer_audit),
        ("layerwise transport", layerwise_transport_audit),
        ("layerwise linear map", layerwise_map_audit),
        ("map-causal link", layerwise_map_causal_audit),
    ):
        if audit.get("status") != "PASS":
            raise RuntimeError(f"{label} audit did not pass")
    required_map_columns = {
        "full_operator_cosine_to_next",
        "full_operator_relative_drift_to_next",
    }
    if not layerwise_map_summary or not required_map_columns.issubset(
        layerwise_map_summary[0]
    ):
        raise RuntimeError("Layerwise map summary lacks gauge-invariant operator metrics")

    base = re.sub(
        r"<title>.*?</title>",
        "<title>Realistic NIAH V4.4 · non-thinking mechanism</title>",
        base,
        count=1,
    )
    base = ensure_viewport_meta(base)
    if "</style>" not in base:
        raise RuntimeError("Base report has no style terminator")
    base = base.replace(
        "</style>",
        EXTRA_CSS
        + CLEAR_CSS
        + REPORT_REFINEMENT_CSS
        + AURORA_CSS
        + PAPER_TYPOGRAPHY_CSS
        + "\n</style>",
        1,
    )
    base, count_palette_replacements = re.subn(
        r"const COUNT_COLORS=\[[^;]+\];",
        "const COUNT_COLORS=['#23165C','#6750E8','#00C2FF','#00D4B4','#39E58C','#C04DFF','#FF5FA2','#F6E36A','#765347','#8190A5'];",
        base,
        count=1,
    )
    if count_palette_replacements != 1:
        raise RuntimeError("Could not apply Aurora count palette")
    nav = """<nav><a href="#mechanism-overview">Mechanism claim</a><a href="#scope">Conclusions</a><a href="#methods">Design &amp; estimands</a><a href="#prompt">Prompt representation</a><a href="#representation-extension">Representation audit</a><a href="#formation-tests">State formation</a><a href="#answer">Answer state</a><a href="#transport-subspace">Transport</a><a href="#attention">Retrieval definition</a><a href="#causal">Causal identification</a><a href="#natural-ov">Write &amp; mediation</a><a href="#synthesis">Synthesis</a><a href="#limits">Limits &amp; reproducibility</a><a href="#appendix-first-locator">Appendix A</a></nav>"""
    base, nav_count = re.subn(r"<nav>.*?</nav>", nav, base, count=1, flags=re.S)
    if nav_count != 1:
        raise RuntimeError("Could not replace report navigation")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = f"""<header>
<div class="eyebrow">Realistic NIAH · V4.4 · non-thinking</div>
<h1>从 prompt running state 到可执行 answer count：non-thinking counting 的表征与因果机制</h1>
<p class="lead">本文按照“机制命题—表征证据—因果识别”的顺序，研究 Qwen3-8B 与 Gemma4-E4B 如何在约10k-token prompt 中形成累计状态、在最终 query 汇集该状态、经 OV/residual 路径改写为 answer representation，并生成数字。每项主张都明确区分 discovery 与 confirmation、相关与干预、必要性与充分性，以及模型共同机制与架构特异路径。</p>
<p class="meta">generated {generated} · self-contained HTML · raw tensors remain in FileStream · Aurora palette · editorial reference: <a href="https://transformer-circuits.pub/2025/linebreaks/index.html">Transformer Circuits · Line Breaks</a></p>
</header>"""
    base, header_count = re.subn(
        r"<header>.*?</header>", header, base, count=1, flags=re.S
    )
    if header_count != 1:
        raise RuntimeError("Could not replace report header")

    scope = build_scope_clear(causal_v2, ov, fullspan_upstream, gemma_residual)
    base = replace_section(base, "scope", scope)
    overview = build_mechanism_overview_detailed(
        ov,
        read_write,
        upstream,
        fullspan_upstream,
        gemma_residual,
        causal_v2,
        correct_state,
    )
    overview, overview_transport_count = re.subn(
        r"</section>\s*$",
        """<div class="conclusion"><strong>新增的跨层 transport 证据。</strong>除图中的 read/write 路径外，我们在 discovery count centroids 上冻结 source→target rank-3 transport basis，并只在 confirmation seeds 注入 aligned 1×/2× dose。Qwen L28→L29 与 Gemma L36→L37 都表现出方向特异的正向 transport，且2× dose显著超过1×；aligned−orthogonal 与 dose2−dose1 均为 exact p=0.001953。该“剂量响应”描述干预effect随dose增加，不表示PCA centroid trajectory是一条直线。结论只覆盖测试的相邻层和同一 answer-query position，不把最后一个 prompt endpoint 写成直接 source；完整定义见第 5 节。</div>\n</section>""",
        overview,
        count=1,
    )
    if overview_transport_count != 1:
        raise RuntimeError("Could not append transport summary to mechanism overview")
    base = base.replace('<section id="scope">', overview + '\n\n<section id="scope">', 1)
    methods = build_methods_clear(
        causal_v2,
        ov,
        read_write,
        upstream,
        fullspan_upstream,
        gemma_residual,
    )
    base = base.replace('<section id="prompt">', methods + '\n\n<section id="prompt">', 1)

    base = base.replace(
        "<h2>1 · Prompt-reading counter representation</h2>",
        "<h2>3 · Prompt running-counter representation</h2>",
        1,
    )
    prompt_marker = '<div class="figure-block"><h3>1.1 Interactive V4.4 prompt counter</h3>'
    if prompt_marker not in base:
        raise RuntimeError("Could not locate prompt counter figure")
    base = base.replace(
        prompt_marker,
        build_running_index_block()
        + '\n\n<div class="figure-block"><h3>3.2 Seed-level prompt counter · 完整交互</h3>',
        1,
    )

    base = base.replace(
        "<h2>2 · Answer-query counter representation</h2>",
        "<h2>6 · Answer-query counter representation</h2>",
        1,
    )
    base = base.replace(
        "<h3>2.1 Interactive V4.4 answer-query counter</h3>",
        "<h3>6.3 Interactive V4.4 answer-query counter</h3>",
        1,
    )
    base = base.replace(
        "<h3>2.2 Prompt 与 answer counter 的共同坐标</h3>",
        "<h3>6.4 Prompt 与 answer counter 的共同坐标</h3>",
        1,
    )
    extension_section = build_extension_representation_section(
        extension_rank,
        extension_regression,
        extension_clustering,
        extension_noise,
        extension_all_token_metrics,
        extension_formula,
        extension_projection,
        extension_classifier,
        extension_classifier_correct,
        cue_doc,
    )
    formation_section = build_endpoint_formation_section(
        extension_attention_stats,
        extension_earlier_heads,
        extension_attention_audit,
    )
    formation_section, direct_causal_count = re.subn(
        r"</section>\s*$",
        build_prompt_direct_causal_subsections(
            extension_token_stats,
            extension_token_audit,
            extension_subspace_stats,
            extension_subspace_audit,
        )
        + "\n</section>",
        formation_section,
        count=1,
    )
    if direct_causal_count != 1:
        raise RuntimeError("Could not append direct prompt causal experiments")
    base = base.replace(
        '<section id="answer">',
        extension_section
        + '\n\n'
        + formation_section
        + '\n\n<section id="answer">',
        1,
    )

    answer_marker = '<div class="figure-block"><h3>6.3 Interactive V4.4 answer-query counter</h3>'
    if answer_marker not in base:
        raise RuntimeError("Could not locate answer counter figure")
    fit_block = (
        build_absolute_deviation_section(answer_data)
        + build_answer_geometry_preface()
        + build_answer_fit_sensitivity(answer_data).replace(
            "<h3>5.1 ", "<h3>6.2 ", 1
        )
    )
    fit_block = fit_block.replace("<h3>4.1 ", "<h3>6.1 ", 1)
    base = base.replace(answer_marker, fit_block + "\n\n" + answer_marker, 1)

    base = base.replace(
        "<h2>3 · V4.4 attention-head representation</h2>",
        "<h2>8 · Attention-head retrieval representation</h2>",
        1,
    )
    attention_heading = "<h2>8 · Attention-head retrieval representation</h2>"
    if attention_heading not in base:
        raise RuntimeError("Could not locate attention section heading")
    transport_placeholder = "<!--TRANSPORT_ALIGNED_FULL-->"
    if base.count(transport_placeholder) != 1:
        raise RuntimeError("Could not locate the 5.4B transport placeholder")
    base = base.replace(
        transport_placeholder,
        build_transport_aligned_section(transport_conditions, transport_contrasts),
        1,
    )
    layerwise_placeholder = "<!--LAYERWISE_SUBSPACE_SWEEP-->"
    if base.count(layerwise_placeholder) != 1:
        raise RuntimeError("Could not locate the Section 5.4 layerwise placeholder")
    base = base.replace(
        layerwise_placeholder,
        build_layerwise_subspace_section(
            layerwise_prompt_statistics,
            layerwise_prompt_trends,
            layerwise_prompt_audit,
            layerwise_answer_statistics,
            layerwise_answer_damage_statistics,
            layerwise_answer_trends,
            layerwise_answer_audit,
            layerwise_transport_statistics,
            layerwise_transport_trends,
            layerwise_transport_audit,
            layerwise_map_summary,
            layerwise_map_audit,
            layerwise_registered_stability,
            layerwise_map_causal_tests,
            layerwise_map_causal_correlations,
            layerwise_map_causal_audit,
        ),
        1,
    )
    base = base.replace(
        attention_heading,
        attention_heading + "\n" + build_attention_estimand_note(),
        1,
    )
    base = base.replace(
        "<h3>3.1 All-head V4.4 atlas</h3>",
        "<h3>8.2 Full-span all-head atlas</h3>",
        1,
    )
    base = base.replace(
        "用按钮切换 endpoint-key 与 full-span-key pooling。横轴是 post-block decoder layer，纵轴是 head index；只有 full-attention layers 才存在完整行。",
        "这里只显示 full-span literal score。横轴是 post-block decoder layer，纵轴是 head index；只有 full-attention layers 才存在完整行。",
        1,
    )
    base, switcher_count = re.subn(
        r'<div class="switcher"><button type="button" data-atlas="span_end".*?</div>',
        "",
        base,
        count=1,
        flags=re.S,
    )
    base, endpoint_panel_count = re.subn(
        r'<div class="atlas-panel" data-atlas-panel="span_end"(?: hidden)?>.*?</svg></div>',
        "",
        base,
        count=1,
        flags=re.S,
    )
    base = base.replace(
        '<div class="atlas-panel" data-atlas-panel="span_sum" hidden>',
        '<div class="atlas-panel" data-atlas-panel="span_sum">',
        1,
    )
    base = base.replace(
        "Atlas 的每一个格子是一个实际保存 full-attention row 的 layer/head。Query 固定为最终 <code>Total:</code> token；key pooling 可切换为 needle endpoint 或完整 needle span 的 literal sum。颜色是 discovery primary score 的对数尺度，只用于同一视图内排序。它不是单个 needle 的概率，也不能跨两种 pooling 直接比较颜色深浅。",
        "Atlas 的每一个格子是一个实际保存 full-attention row 的 layer/head。Query 固定为最终 <code>Total:</code> token；key pooling 固定为完整 needle span 的 literal sum。颜色是 discovery primary score 的对数尺度，只用于同一模型内排序；它不是单个 needle 的概率，也不能跨模型直接比较颜色深浅。",
        1,
    )
    base = base.replace(
        "Endpoint phenotype symbols are shown only in the span-end view.",
        "Only the full-span literal pooling view is included in this report.",
        1,
    )
    if switcher_count != 1 or endpoint_panel_count != 1:
        raise RuntimeError(
            "Could not reduce the attention atlas to the full-span panel"
        )

    base = replace_section(
        base,
        "causal",
        build_causal_section_clear(causal_v2, seed_confirmation, correct_state),
    )
    positive_section = build_positive_mechanism_section(
        ov, read_write, upstream, fullspan_upstream, gemma_residual
    )
    synthesis_section = build_synthesis_clear(ov, fullspan_upstream, gemma_residual)
    base = base.replace(
        '<section id="limits">',
        positive_section
        + "\n\n"
        + synthesis_section
        + '\n\n<section id="limits">',
        1,
    )
    base = replace_section(
        base,
        "limits",
        build_limits_clear(
            causal_v2,
            seed_confirmation,
            ov,
            read_write,
            upstream,
            fullspan_upstream,
            gemma_residual,
            correct_state,
            extension_token_audit,
            extension_subspace_audit,
        ),
    )

    if "function makeProjector" not in base:
        raise RuntimeError("Could not locate embedded visualization script")
    running_only_js = (
        MECHANISM_DETAILED_JS
        + EXTRA_JS[EXTRA_JS.index("function makeRunningIndex") :]
    )
    base = base.replace(
        "function makeProjector", running_only_js + "\nfunction makeProjector", 1
    )
    old_boot = "makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    new_boot = "makeMechanismWalkthrough();makeRunningIndex();makeProjector('prompt',PROMPT_DATA,'prompt');makeProjector('answer',ANSWER_DATA,'answer');makeJoint();"
    if old_boot not in base:
        raise RuntimeError("Could not locate visualization bootstrap")
    base = base.replace(old_boot, new_boot, 1)

    # Normalize colors embedded in legacy SVG/CSS fragments as well as the new
    # Aurora theme.  This keeps Qwen/Gemma visual identities consistent across
    # every figure, including diagrams inherited from the earlier report.
    aurora_legacy_map = {
        "#27685F": "#6750E8",
        "#A66A45": "#00D4B4",
        "#6A958A": "#6750E8",
        "#B78767": "#00D4B4",
        "#588BD2": "#00C2FF",
    }
    for old_color, new_color in aurora_legacy_map.items():
        base = re.sub(re.escape(old_color), new_color, base, flags=re.I)

    base = insert_concrete_examples(base)
    base = make_all_tables_collapsible(base)
    base = merge_transport_into_section_5_4(base)
    base = make_secondary_content_collapsible(base)
    base = apply_publication_editorial_pass(base)
    base = apply_paper_math_typesetting(base)

    required_sections = [
        "mechanism-overview",
        "scope",
        "methods",
        "prompt",
        "representation-extension",
        "formation-tests",
        "answer",
        "attention",
        "causal",
        "natural-ov",
        "synthesis",
        "limits",
        "appendix-first-locator",
    ]
    for section_id in required_sections:
        if base.count(f'id="{section_id}"') != 1:
            raise RuntimeError(f"Section id count is not one: {section_id}")
    if base.count('id="transport-subspace"') != 1 or '<section id="transport-subspace">' in base:
        raise RuntimeError("Transport results were not merged into section 5.4B")
    if base.count('id="layerwise-subspace-sweep"') != 1:
        raise RuntimeError("Layerwise subspace results were not merged into section 5.4")
    for removed_section in ("cue-robustness", "read-write", "upstream"):
        if f'id="{removed_section}"' in base:
            raise RuntimeError(f"Removed section unexpectedly present: {removed_section}")
    for canvas_id in (
        "running-index-canvas",
        "joint-canvas",
    ):
        if base.count(f'id="{canvas_id}"') != 1:
            raise RuntimeError(f"Interactive canvas id count is not one: {canvas_id}")
    if len(re.findall(r"<figcaption\b", base)) != base.count("</figcaption>"):
        raise RuntimeError("Unbalanced figure captions")
    if base.count("<section") != base.count("</section>"):
        raise RuntimeError("Unbalanced sections")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(base, encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "sections": required_sections,
                "figures": len(re.findall(r"<figure\b", base)),
                "figcaptions": len(re.findall(r"<figcaption\b", base)),
                "qwen_ov_global_iut_p": ov["primary_decision"]["global_intersection_union_p"],
                "gemma_residual_global_iut_p": gemma_residual["primary_decision"]["global_intersection_union_p"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the paper-grade integrated V4.4 non-thinking mechanism report"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report_clear(args.repo_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

"""3-stage document pipeline: evidence -> contribution -> pfb. Each stage saves one StageRun."""
import time

from pfb_app.domain.models import (ContributionOutput, EvidenceItem, EvidenceOutput, Judgment,
                                   Kind, PfbOutput, Stage, StageRun)
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.structured import call_structured
from pfb_app.pipeline import checks
from pfb_app.pipeline.common import (StageContext, assignment_of as _assignment,
                                     fail_run as _fail, save_run as _save, usable as _usable)
from pfb_app.pipeline.prompts import build_user_text, load_prompt
from pfb_app.preprocess.documents import Page


def run_evidence(ctx: StageContext, submission_id: int) -> StageRun:
    started = time.monotonic()
    prepare = ctx.repo.latest_run(submission_id, Stage.prepare, None)
    if not _usable(prepare):
        return _fail(ctx, submission_id, Stage.evidence,
                     "ページの準備結果がありません。先に書類の準備を実行してください。",
                     started=started)
    pages = [Page.from_dict(d) for d in prepare.output["pages"]]
    assignment = _assignment(ctx, submission_id)
    prompt = load_prompt(ctx.settings, Kind.document, Stage.evidence)
    sections = [("活動の内容", assignment["task_text"])]
    sections += [(f"p{p.number}", p.text) for p in pages]
    req = ChatRequest(
        system=prompt.text, user_text=build_user_text(sections),
        image_paths=[p.image_path for p in pages if p.image_path is not None],
        json_schema=EvidenceOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.evidence.value])
    result = call_structured(ctx.backend, ctx.repo, req, EvidenceOutput,
                             max_retries=ctx.settings.max_retries, yield_check=ctx.yield_check)
    common = dict(upstream=[prepare.id], prompt_hash=prompt.hash,
                  model_id=ctx.backend.model_id, started=started)
    if result.parsed is None:
        return _fail(ctx, submission_id, Stage.evidence, "\n".join(result.errors), **common)

    page_text = {p.number: p.text for p in pages}
    per_page: dict[int, int] = {}
    items, warnings = [], []
    for draft in result.parsed.items:
        n = per_page[draft.page] = per_page.get(draft.page, 0) + 1
        item_id = f"p{draft.page}-{n:03d}"
        if draft.evidence_source == "figure":
            verified = True  # no quote to verify
        else:
            verified = checks.verify_quote(draft.quote, page_text.get(draft.page, ""))
            if not verified:
                warnings.append(f"引用がページの本文と一致しません: {item_id}")
        items.append(EvidenceItem(**draft.model_dump(), item_id=item_id,
                                  quote_verified=verified).model_dump())
    return _save(ctx, submission_id, Stage.evidence, output={"items": items},
                 warnings=warnings, **common)


def _evidence_items(ctx: StageContext, run: StageRun) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(d) for d in run.output["items"]]


def run_contribution(ctx: StageContext, submission_id: int) -> StageRun:
    started = time.monotonic()
    evidence = ctx.repo.latest_run(submission_id, Stage.evidence, None)
    if not _usable(evidence):
        return _fail(ctx, submission_id, Stage.contribution,
                     "根拠項目の抽出結果がありません。先に根拠の抽出を実行してください。",
                     started=started)
    items = _evidence_items(ctx, evidence)
    assignment = _assignment(ctx, submission_id)
    prompt = load_prompt(ctx.settings, Kind.document, Stage.contribution)
    common = dict(upstream=[evidence.id], prompt_hash=prompt.hash, started=started)
    if not items:
        return _save(ctx, submission_id, Stage.contribution, output={"judgments": []}, **common)

    ref_ids = [i.item_id for i in items]
    body = "\n\n".join(
        f"ref_id: {i.item_id}\nページ: {i.page}\n引用: {i.quote}\n内容: {i.observed_content}"
        for i in items)
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([("活動の内容", assignment["task_text"]),
                                   ("ルーブリック", assignment["rubric_text"]),
                                   ("根拠項目", body)]),
        image_paths=[], json_schema=ContributionOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.contribution.value])
    result = call_structured(
        ctx.backend, ctx.repo, req, ContributionOutput,
        max_retries=ctx.settings.max_retries, yield_check=ctx.yield_check,
        validate=lambda out: checks.judgment_alignment_errors(ref_ids, out.judgments))
    common["model_id"] = ctx.backend.model_id
    if result.parsed is None:
        return _fail(ctx, submission_id, Stage.contribution, "\n".join(result.errors), **common)
    return _save(ctx, submission_id, Stage.contribution,
                 output={"judgments": [j.model_dump() for j in result.parsed.judgments]},
                 **common)


def run_pfb(ctx: StageContext, submission_id: int, subject_id: int) -> StageRun:
    started = time.monotonic()
    contribution = ctx.repo.latest_run(submission_id, Stage.contribution, None)
    if not _usable(contribution):
        return _fail(ctx, submission_id, Stage.pfb, "貢献判定の結果がありません。"
                     "先に貢献判定を実行してください。", subject_id=subject_id, started=started)
    evidence = (ctx.repo.get_run(contribution.upstream_run_ids[0])
                if contribution.upstream_run_ids else None)
    if not _usable(evidence):
        return _fail(ctx, submission_id, Stage.pfb, "根拠項目の抽出結果がありません。"
                     "先に根拠の抽出を実行してください。", subject_id=subject_id, started=started)
    items = {i.item_id: i for i in _evidence_items(ctx, evidence)}
    judgments = [Judgment.model_validate(d) for d in contribution.output["judgments"]]
    missing = sorted({j.ref_id for j in judgments} - set(items))
    if missing:
        return _fail(ctx, submission_id, Stage.pfb,
                     "貢献判定が存在しない根拠項目を参照しています: " + ", ".join(missing),
                     subject_id=subject_id, upstream=[contribution.id], started=started)
    true_judgments = [j for j in judgments if j.is_positive_contribution]
    prompt = load_prompt(ctx.settings, Kind.document, Stage.pfb)
    common = dict(subject_id=subject_id, upstream=[contribution.id],
                  prompt_hash=prompt.hash, started=started)
    if not true_judgments:
        out = PfbOutput(pfb_status="insufficient_positive_evidence", pfb=None, evidence_refs=[])
        return _save(ctx, submission_id, Stage.pfb, output=out.model_dump(), **common)

    true_ids = {j.ref_id for j in true_judgments}
    body = "\n\n".join(
        f"ref_id: {j.ref_id}\n引用: {items[j.ref_id].quote}\n"
        f"内容: {items[j.ref_id].observed_content}\n貢献の解釈: {j.contribution_interpretation}"
        for j in true_judgments)
    assignment = _assignment(ctx, submission_id)
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([("活動の内容", assignment["task_text"]),
                                   ("貢献ありと判定された根拠項目", body)]),
        image_paths=[], json_schema=PfbOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.pfb.value])
    result = call_structured(
        ctx.backend, ctx.repo, req, PfbOutput,
        max_retries=ctx.settings.max_retries, yield_check=ctx.yield_check,
        validate=lambda out: checks.evidence_ref_errors(out, true_ids))
    common["model_id"] = ctx.backend.model_id
    if result.parsed is None:
        return _fail(ctx, submission_id, Stage.pfb, "\n".join(result.errors), **common)
    pfb = result.parsed
    warnings = []
    if pfb.pfb:
        internal_ids = set(items)
        subject = next((s for s in ctx.repo.list_subjects(submission_id)
                        if s["id"] == subject_id), None)
        if subject and subject["participant_id"]:
            internal_ids.add(subject["participant_id"])
        warnings = checks.pfb_text_warnings(pfb.pfb, internal_ids, set())
    return _save(ctx, submission_id, Stage.pfb, output=pfb.model_dump(), warnings=warnings,
                 **common)

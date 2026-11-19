"""確認・承認：段階ごとの表示と編集、再生成、人物の確認、承認、手書きPFB。"""
import json

import streamlit as st
from pydantic import ValidationError

from pfb_app.domain.models import Kind, Stage
from pfb_app.domain.vocab import localize, mark_name
from pfb_app.pipeline.runner import submission_engine
from pfb_app.pipeline.tracked_identity import RETRACKED, ROSTER_RESET, roster_was_reset
from pfb_app.services import review, short_text
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.ui import access
from pfb_app.ui.common import (SUBJECT_STATUS_LABELS, flash, get_repo, mapping_form,
                               mapping_from_choices, select_assignment, show_flash,
                               validation_text)
from pfb_app.ui.wizard.step_people import SUGGESTED
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

repo = get_repo()
st.title("確認・承認")
show_flash("school")

# stage -> (key of the row list in the run output, row id key)
ROWS = {Stage.identity: ("participants", "participant_id"), Stage.behavior: ("events", "event_id"),
        Stage.evidence: ("items", "item_id"), Stage.contribution: ("judgments", "ref_id")}
LIST_COLUMNS = {"verification_points", "rubric_criteria"}  # edited as one item per line
TAB_LABELS = {Stage.identity: "人物の特定", Stage.behavior: "行動の観察", Stage.evidence: "根拠の抽出",
              Stage.contribution: "貢献の判定", Stage.pfb: "PFB"}
MAPPING_TAB = "人物と参加者の対応づけ"   # the identity tab of the tracking engine
PER_CAMERA = {Stage.identity, Stage.behavior, Stage.contribution}


def guarded(action, success: str | None = None):
    """Run a service call; show its errors in Japanese. True on success."""
    try:
        result = action()
    except (BusyError, ApprovalError) as e:
        st.error(str(e))
        return False
    except ValidationError as e:
        st.error(validation_text(e))
        return False
    if success:
        flash(success)
        st.rerun()
    return True if result is None else result


def run_header(run) -> None:
    kind = (short_text.SHORT_SOURCE if short_text.is_short(run)
            else "あなたが編集" if run.source == "teacher" else "モデルの出力")
    st.caption(f"版 {run.version}（{kind}）")
    if review.is_stale_chain(repo, run):
        st.badge("古い（上流が更新されました）", color="orange")
    english = review.english_left(run) if run.stage == Stage.behavior else 0
    blocking = english and run.source != "teacher" and (
        (run.output or {}).get("provenance") or {}).get("translated") is False
    if blocking:
        st.badge(review.UNTRANSLATED, color="red")
    elif english:   # e.g. saved by the teacher, whose review is the check: a reminder only
        st.badge(f"英語の記録が残っています（{english}件）", color="orange")
    if run.status == "failed":
        st.error(f"失敗しました: {run.error}")
    for w in run.warnings:
        if w == short_text.MARK:    # a marker, not a problem: the text is the short one (L6)
            st.info(f"{short_text.MARK}: 映っている場面が少ない人に用意した文です。")
        else:
            st.warning(w)


def to_cells(rows: list[dict]) -> list[dict]:
    return [{k: ("\n".join(v) if k in LIST_COLUMNS else v) for k, v in r.items()} for r in rows]


def from_cells(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        r = json.loads(json.dumps(r, default=str))
        for k in LIST_COLUMNS & set(r):
            r[k] = [line.strip() for line in (r[k] or "").splitlines() if line.strip()]
        out.append(r)
    return out


def show_frames(sid: int, run, key: str) -> None:
    """Pick a row of a behavior/contribution table to see its frames."""
    behavior = run if run.stage == Stage.behavior else (
        repo.get_run(run.upstream_run_ids[0]) if run.upstream_run_ids else None)
    events = {e["event_id"]: e for e in
              ((behavior.output or {}).get("events", []) if behavior else [])}
    refs = [r[ROWS[run.stage][1]] for r in run.output[ROWS[run.stage][0]]]
    refs = [r for r in refs if r in events]
    if not refs:
        return
    chosen = st.selectbox("行を選ぶとフレーム画像を表示します", refs, index=None, key=key)
    if chosen:
        event = events[chosen]
        shown = [f for f in review.frames_for(repo, behavior, event) if f.path.is_file()]
        if shown:
            cols = st.columns(len(shown))
            for col, f in zip(cols, shown):
                col.image(str(f.path), caption=f.caption)
        else:
            st.info("該当するフレーム画像がありません。")


def table_stage(sid: int, stage: Stage, camera: str | None, kind: Kind) -> None:
    run = repo.latest_run(sid, stage, camera)
    if run is None:
        st.info("まだ結果がありません。")
    else:
        run_header(run)
        if run.status == "ok" and run.output:
            list_key, _ = ROWS[stage]
            rows = to_cells(run.output[list_key])
            if rows:
                edited = st.data_editor(rows, hide_index=True, key=f"ed-{run.id}")
                if st.button("編集を保存", key=f"save-{run.id}"):
                    guarded(lambda: review.edit_stage_output(
                        repo, run.id, {list_key: from_cells(edited)}), "編集を保存しました。")
            else:
                st.info("結果の行がありません。")
            if kind == Kind.video and stage in (Stage.behavior, Stage.contribution) and rows:
                show_frames(sid, run, f"frames-{run.id}")
    if st.button("この段階から再生成", key=f"regen-{stage.value}-{camera}"):
        guarded(lambda: review.regenerate_from(repo, sid, stage, camera=camera),
                "再生成をキューに入れました。")


def mapping_stage(sid: int, camera: str) -> None:
    """Tracking engine: the people sheet and one select box per person, 「n番の人」 (student or
    対象外), with the reason of each pre-filled guess."""
    track = repo.latest_run(sid, Stage.track, camera)
    form = mapping_form(repo, sid, camera)
    if track is None:
        st.info("人物の追跡結果がまだありません。追跡が終わると人物一覧が表示されます。")
    elif form is None:
        st.error(f"人物の追跡に失敗しました: {track.error}")
    else:
        for w in form.track.warnings:
            st.warning(w)
        if form.sheet is not None:
            st.image(str(form.sheet), caption=f"人物一覧（{camera}）")
        else:
            st.info("人物一覧の画像が見つかりません。")
        if form.identity is not None:
            run_header(form.identity)
            if not form.current:   # retracked, or saved before the roster was replaced
                meta = repo.get_submission(sid)["meta"]
                st.warning(localize(ROSTER_RESET, form.domain)
                           if roster_was_reset(meta, form.identity) else RETRACKED)
        if form.suggested:
            st.caption(SUGGESTED)
        st.caption("人物一覧の番号ごとに、どの参加者かを選んでください。"
                   "評価しない人（指導する人や見学者など）は「対象外」にします。")
        chosen = {}
        for label in form.labels:
            chosen[label] = st.selectbox(
                mark_name(label), form.options, index=form.index(label), format_func=form.name,
                placeholder="選んでください", key=f"map-{form.track.id}-{label}")
            if label in form.reasons:          # 「背番号10から推定」: why it is pre-filled
                st.caption(f"推定: {form.reasons[label]}")
        if st.button("対応づけを保存", key=f"map-save-{form.track.id}", type="primary"):
            mapping, unset = mapping_from_choices(chosen)
            if unset:
                st.error("まだ選んでいない人物があります: " + "、".join(map(mark_name, unset)))
            else:
                guarded(lambda: review.save_tracked_identity(repo, sid, camera, mapping),
                        "対応づけを保存しました。「参加者特定を確認」で確定してください。")
    if st.button("人物の追跡からやり直す", key=f"regen-track-{camera}"):
        guarded(lambda: review.regenerate_from(repo, sid, Stage.track, camera=camera),
                "人物の追跡をキューに入れました。")


def pfb_stage(sid: int, subject: dict, kind: Kind) -> None:
    run = repo.latest_run(sid, Stage.pfb, subject_id=subject["id"])
    if run is None:
        st.info("まだ結果がありません。")
    else:
        run_header(run)
        out = run.output or {}
        if out.get("pfb_status") == "generated":
            st.text_area("PFB", out["pfb"], disabled=True, key=f"pfb-{run.id}")
            st.caption("根拠: " + "、".join(out["evidence_refs"]) if out["evidence_refs"] else "根拠: なし")
        elif run.status == "ok":
            st.info("PFBを書くための根拠が足りません。")
    if st.button("この段階から再生成", key=f"regen-pfb-{subject['id']}"):
        guarded(lambda: review.regenerate_from(repo, sid, Stage.pfb, subject_id=subject["id"]),
                "再生成をキューに入れました。")
    st.divider()
    manual = st.text_area("手書きのPFB", key=f"manual-{subject['id']}")
    if st.button("手書きのPFBを保存", key=f"manual-save-{subject['id']}"):
        guarded(lambda: review.write_manual_pfb(repo, subject["id"], manual), "手書きのPFBを保存しました。")
    if review.approved_automatically(repo, subject):
        st.caption("この人は「確認せずに自動で承認する」で、自動で承認されました。")
    c1, c2 = st.columns(2)
    if c1.button("承認", key=f"approve-{subject['id']}", type="primary"):
        guarded(lambda: review.approve(repo, subject["id"]), "承認しました。")
    if c2.button("承認を取り消す", key=f"unapprove-{subject['id']}"):
        guarded(lambda: review.unapprove(repo, subject["id"]), "承認を取り消しました。")


SUBJECT_QUERY = "subject"     # ?subject=<id>: the wizard's 「根拠を管理用の画面で直す」
SUBJECT_KEY = "review_subject"


def wanted_subject() -> int | None:
    """The subject named in the URL (read once, then dropped so the picker stays free); its
    work becomes the chosen assignment."""
    raw = st.query_params.get(SUBJECT_QUERY)
    if raw is None:
        return None
    st.query_params.pop(SUBJECT_QUERY, None)
    try:
        subject = repo.get_subject(int(raw))
        submission = repo.get_submission(subject["submission_id"])
    except (KeyError, ValueError):
        return None
    st.session_state["current_assignment"] = submission["assignment_id"]
    return subject["id"]


wanted = wanted_subject()
assignment = select_assignment(repo)
submissions = repo.list_submissions(assignment["id"]) if assignment else []
if assignment and not submissions:
    st.info("ファイルがありません。「ファイルの登録」ページで登録してください。")


def set_auto_approve(assignment_id: int, key: str) -> None:
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    fresh = get_repo()   # a callback runs before the next script run, in its own thread
    try:
        review.set_auto_approve(fresh, assignment_id, bool(st.session_state.get(key)))
    finally:
        fresh.conn.close()


if assignment:
    auto_key = f"auto-approve-{assignment['id']}"
    st.session_state[auto_key] = bool(assignment.get("auto_approve"))
    st.toggle("確認せずに自動で承認する", key=auto_key, on_change=set_auto_approve,
              args=(assignment["id"], auto_key),
              help="オンにすると、これから作られる文章のうち、気になる点がなく根拠が足りているものは、"
                   "確認せずに承認されます。気になる点がある文章は、人が確かめて承認します。")
if submissions:
    kind = Kind(assignment["kind"])
    pairs = [(sub, s) for sub in submissions for s in repo.list_subjects(sub["id"])]
    if not pairs:
        st.info("対象者がいません。")
        st.stop()
    labels = {s["id"]: f"{s['display_name']}（{sub['label']}）／{SUBJECT_STATUS_LABELS[review.subject_status(repo, s['id'])]}"
              for sub, s in pairs}
    if wanted in labels:
        st.session_state[SUBJECT_KEY] = wanted
    elif st.session_state.get(SUBJECT_KEY) not in labels:
        st.session_state.pop(SUBJECT_KEY, None)
    chosen = st.selectbox("対象者", list(labels), format_func=labels.get, key=SUBJECT_KEY)
    submission, subject = next((sub, s) for sub, s in pairs if s["id"] == chosen)
    sid = submission["id"]
    st.write(f"状態: **{SUBJECT_STATUS_LABELS[review.subject_status(repo, subject['id'])]}**")

    tracked = kind == Kind.video and submission_engine(submission) == "pfb3x3"
    tab_labels = {**TAB_LABELS, **({Stage.identity: MAPPING_TAB} if tracked else {})}
    if kind == Kind.video:
        if submission["identity_confirmed"]:
            st.success("参加者の特定は確認済みです。")
        else:
            st.warning("参加者の特定がまだ確認されていません。承認する前に、"
                       f"「{tab_labels[Stage.identity]}」を確かめてください。")
        if st.button("参加者特定を確認"):
            guarded(lambda: review.confirm_identity(repo, sid), "参加者の特定を確認しました。")
        untranslated = review.untranslated_behavior(repo, sid) if tracked else {}
        if untranslated:
            st.warning(review.untranslated_message(untranslated))

    stages = ([Stage.identity, Stage.behavior, Stage.contribution, Stage.pfb] if kind == Kind.video
              else [Stage.evidence, Stage.contribution, Stage.pfb])
    for tab, stage in zip(st.tabs([tab_labels[s] for s in stages]), stages):
        with tab:
            if stage == Stage.pfb:
                pfb_stage(sid, subject, kind)
            elif kind == Kind.video and stage in PER_CAMERA:
                cameras = list(submission["meta"].get("cameras") or {})
                camera = st.selectbox("カメラ", cameras, key=f"cam-{stage.value}") if cameras else None
                if camera and stage == Stage.identity and tracked:
                    mapping_stage(sid, camera)
                elif camera:
                    table_stage(sid, stage, camera, kind)
                else:
                    st.info("カメラが登録されていません。")
            else:
                table_stage(sid, stage, None, kind)

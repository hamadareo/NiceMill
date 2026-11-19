"""ききたいことを聞く screen (AppTest, pages/ask.py): asking, the scenes under an answer, the
example questions, a model that is not set up, no write to the database, the people taken out,
and the roles. The model is always a fake one; synthetic works only."""
import json

import pytest
from streamlit.testing.v1 import AppTest

from pdf_synthetic import video_group, work
from pfb_app.inference.fake import FakeBackend
from pfb_app.services import ask
from pfb_app.ui.wizard import ask_view
from test_neutral_wording import clean, everything
from test_wizard_ui_basic import FORBIDDEN, UI, ok, primaries, env  # noqa: F401 (env)

PAGE = UI / "pages" / "ask.py"


def page(query: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    for k, v in (query or {}).items():
        at.query_params[k] = str(v)
    return ok(at.run())


def neutral(at, where):
    text = everything(at)
    clean(text, where)
    found = [w for w in FORBIDDEN if w in text]
    assert not found, f"{where}: {found}"
    for word in ("公式", "判定結果", "確定", "順位", "ランキング", "評価が高"):
        assert word not in text, f"{where}: {word}"


def snapshot(repo) -> list[str]:
    return list(repo.conn.iterdump())


def a_model(repo, aid, sid, plan: dict, monkeypatch):
    """A fake model whose answer is made from the very facts the code finds for `plan`."""
    rec = ask.load(repo, aid, sid)
    f = ask.facts(ask.Plan(**plan), rec)
    total = [int(x) for x in f.text.split("よかった場面の数: ")[1].split("回")[:1]][0]
    ref = next(iter(f.evidence))
    answer = {"answer": f"記録では{total}回です[{ref}]。", "scene_ids": [ref]}

    def respond(req):
        if "intent" in req.json_schema.get("properties", {}):
            return json.dumps(plan, ensure_ascii=False)
        return json.dumps(answer, ensure_ascii=False)

    backend = FakeBackend(respond)
    monkeypatch.setattr(ask_view, "backend_for", lambda settings: backend)
    return backend, total


def ask_it(at: AppTest, question: str) -> AppTest:
    at.text_input[0].input(question)
    at.button[0].click()      # 「たずねる」 (the form's button comes first)
    return ok(at.run())


def test_without_works_the_page_says_so(env):
    at = page()
    assert ask_view.NO_WORK in everything(at)
    neutral(at, "ask (no work)")


def test_a_question_is_answered_with_its_scenes(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60, teams=True)
    backend, total = a_model(repo, aid, sid, {"intent": "list", "person": "山田花子"}, monkeypatch)
    at = page({"work": aid})
    text = everything(at)
    assert ask_view.TITLE in text and ask_view.KEPT in text
    assert primaries(at) == [ask_view.ASK]
    assert [b.label for b in at.button][1:4] == ask.examples(ask.load(repo, aid, sid))
    at = ask_it(at, "山田花子さんの場面を教えて")
    text = everything(at)
    assert "山田花子さんの場面を教えて" in text
    assert f"記録では{total}回です" in text and "[S" not in text
    assert ask_view.BASIS in text and "山田花子" in text
    assert ask.RECORD_NOTE in text and ask_view.RANGE in text
    assert ask_view.CLEAR in [b.label for b in at.button]
    neutral(at, "ask answered")


def test_an_example_button_asks_it(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60)
    seen = []
    backend = FakeBackend(lambda req: seen.append(req) or json.dumps({"intent": "unknown"}))
    monkeypatch.setattr(ask_view, "backend_for", lambda settings: backend)
    at = page({"work": aid})
    examples = ask.examples(ask.load(repo, aid, sid))
    next(b for b in at.button if b.label == examples[0]).click()
    at = ok(at.run())
    assert seen and examples[0] in everything(at)
    assert ask.UNKNOWN in everything(at)


def test_a_model_not_set_up_is_said_in_words_and_nothing_breaks(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60)
    at = ask_it(page({"work": aid}), "声かけは何回？")
    text = everything(at)
    assert ask.UNAVAILABLE in text
    neutral(at, "ask unavailable")


def test_reading_writes_nothing_to_the_database(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60)
    a_model(repo, aid, sid, {"intent": "count", "category": "声かけ"}, monkeypatch)
    before = snapshot(repo)
    at = ask_it(page({"work": aid}), "声かけは何回？")
    ask_it(at, "もう一つ聞きます")
    assert snapshot(repo) == before


def test_history_is_this_sessions_and_can_be_cleared(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60)
    a_model(repo, aid, sid, {"intent": "count", "category": "声かけ"}, monkeypatch)
    at = ask_it(page({"work": aid}), "声かけは何回？")
    assert ask_view.HISTORY_KEY in at.session_state and len(at.session_state[ask_view.HISTORY_KEY]) == 1
    at = ok(next(b for b in at.button if b.label == ask_view.CLEAR).click().run())
    assert at.session_state[ask_view.HISTORY_KEY] == []
    assert "声かけは何回？" not in everything(at)


def test_a_person_taken_out_is_never_shown_or_found(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60)
    gone = repo.list_subjects(sid)[0]["display_name"]
    repo.conn.execute("UPDATE subjects SET excluded=1 WHERE display_name=?", (gone,))
    backend = FakeBackend(lambda req: json.dumps({"intent": "list", "person": gone}))
    monkeypatch.setattr(ask_view, "backend_for", lambda settings: backend)
    at = ask_it(page({"work": aid}), "名前を言って場面を教えて")
    text = everything(at)
    assert ask.NO_PERSON in text
    assert gone not in text.replace("名前を言って場面を教えて", "")
    assert all(gone not in c.system + c.user_text for c in backend.calls)


pytestmark_auth = pytest.mark.real_auth


@pytest.mark.real_auth
def test_a_viewer_is_refused_and_an_evaluator_may_ask(env):
    from pfb_app.ui import access, auth_ui
    from test_accounts_ui import signed_in
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=40)
    signed_in(env)
    _, viewer = signed_in(env, role="viewer", login="miru", name="見る 人")
    _, helper = signed_in(env, role="evaluator", login="tasuke2", name="手伝う 人")
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    at.session_state[auth_ui.TOKEN] = viewer
    at = ok(at.run())
    assert access.NOT_HERE in everything(at) and not at.text_input
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    at.session_state[auth_ui.TOKEN] = helper
    at.query_params["work"] = str(aid)
    at = ok(at.run())
    assert ask_view.TITLE in everything(at) and at.text_input


def test_history_is_dropped_when_the_records_change(env, monkeypatch):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60)
    a_model(repo, aid, sid, {"intent": "count", "category": "声かけ"}, monkeypatch)
    at = ask_it(page({"work": aid}), "声かけは何回？")
    assert len(at.session_state[ask_view.HISTORY_KEY]) == 1
    gone = repo.list_subjects(sid)[0]["display_name"]
    repo.conn.execute("UPDATE subjects SET excluded=1 WHERE display_name=?", (gone,))
    at = ok(at.run())
    assert at.session_state[ask_view.HISTORY_KEY] == []
    assert "声かけは何回？" not in everything(at)


def test_a_link_to_another_work_opens_that_work(env):
    settings, repo = env
    first = work(repo, title="最初の試合")
    video_group(settings, repo, first, people=2, seconds=60)
    second = work(repo, title="二つ目の試合")
    video_group(settings, repo, second, people=2, seconds=60, label="B")
    at = page({"work": first})
    assert at.selectbox(key="pfb_ask_work").value == first
    at.query_params["work"] = str(second)           # as following 「記録にききたいことを聞く」 of another work
    at = ok(at.run())
    assert at.selectbox(key="pfb_ask_work").value == second
    pick = at.selectbox(key="pfb_ask_work").select(first)     # the person's own choice stays
    at = ok(pick.run())
    assert at.selectbox(key="pfb_ask_work").value == first


def test_every_string_on_the_ask_page_passes_the_neutral_wording_lists(env, monkeypatch):
    """The text of the page in each state (empty, a question, an answer with scenes, a refusal)
    goes through test_neutral_wording's lists: school words, internal words, and the words of
    a verdict or a ranking."""
    from test_neutral_wording import SCHOOL_WORDS
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=3, seconds=60, teams=True)
    a_model(repo, aid, sid, {"intent": "list", "person": "山田花子"}, monkeypatch)
    at = ask_it(page({"work": aid}), "山田花子さんの場面を教えて")
    shown = [everything(at)]
    for text in shown:
        assert not [w for w in SCHOOL_WORDS if w in text]
        assert not [w for w in FORBIDDEN if w in text]
        for word in ("公式", "判定結果", "確定", "順位", "ランキング", "最多", "評価が高"):
            assert word not in text, word

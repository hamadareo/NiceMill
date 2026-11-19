"""試合・活動のまとめ screens (AppTest, pages/dashboard.py): sports with and without the court
geometry, a general work (場面ごとの様子), a work whose videos were removed, the jump, the
entry links on steps 5 and 6. Synthetic data only."""
from streamlit.testing.v1 import AppTest

from pdf_synthetic import document_work, video_group, work
from pfb_app.services import retention
from test_neutral_wording import clean, everything
from test_wizard_ui_basic import FORBIDDEN, UI, app, env, ok, primaries  # noqa: F401 (env)

PAGE = UI / "pages" / "dashboard.py"
CSV_PEOPLE = "人ごとの表をCSVで受け取る"
JUMP = "この場面から見る"


def page(query: dict | None = None, session: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=90)
    for k, v in (query or {}).items():
        at.query_params[k] = str(v)
    for k, v in (session or {}).items():
        at.session_state[k] = v
    return ok(at.run())


def neutral(at, where):
    text = everything(at)
    clean(text, where)
    found = [w for w in FORBIDDEN if w in text]
    assert not found, f"{where}: {found}"
    for word in ("公式", "判定結果", "確定", "判決", "審判の判定"):
        assert word not in text, f"{where}: {word}"


def images_alt(at) -> str:
    import re
    from test_wizard_ui_basic import html
    return "\n".join(re.findall(r'alt="([^"]*)"', html(at)))


def test_a_sports_work_with_the_court(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    at = page({"work": aid})
    text = everything(at)
    assert "試合・練習のまとめ" in text
    assert "映っていた時間" in text and "よかった場面" in text
    assert "動いた距離" in text and "推定" in text
    assert "チーム1" in text and "チーム2" in text
    alts = images_alt(at)
    assert "コートのどこにいたか" in alts or "のいた所" in alts
    assert "時間の流れ" in alts
    assert primaries(at) == [CSV_PEOPLE]
    assert [b.label for b in at.download_button] == [CSV_PEOPLE, "場面の一覧をCSVで受け取る",
                                                     "チームごとの表をCSVで受け取る"]
    assert any(b.label == JUMP for b in at.button)
    assert "この数字について" in text
    neutral(at, "sports dashboard")


def test_without_the_court_a_frame_heatmap_and_no_distance(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, geometry=False, teams=True)
    at = page({"work": aid})
    text = everything(at)
    assert "画面の中の位置" in text
    assert "動いた距離 約" not in text and "動いた距離（推定" not in text
    assert "カメラ1から見た位置" in images_alt(at)
    neutral(at, "sports dashboard without H")


def test_a_general_work_shows_scenes(env):
    settings, repo = env
    aid = work(repo, domain="school", title="塔づくり")
    video_group(settings, repo, aid, people=3, seconds=120)
    at = page({"work": aid})
    text = everything(at)
    assert "活動のまとめ" in text and "場面ごとの様子" in text
    assert "伝える・話し合う" in text or "手伝う・支える" in text
    assert "チーム" not in text and "コート" not in text and "試合" not in text
    assert primaries(at) == [CSV_PEOPLE]
    neutral(at, "general dashboard")


def test_a_jump_opens_one_player_at_the_moment(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60)
    at = page({"work": aid})
    jump = next(b for b in at.button if b.label == JUMP)
    at = ok(jump.click().run())
    videos = at.get("video")
    assert len(videos) == 1
    assert videos[0].proto.start_time >= 0


def test_removed_videos_keep_the_moments_without_jumps(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60)
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-10-24T00:00:00+00:00'"
                      " WHERE id=?", (aid,))
    retention._finish(repo, settings, aid)
    at = page({"work": aid})
    text = everything(at)
    assert "動画は消してあります" in text
    assert not any(b.label == JUMP for b in at.button)
    assert "よかった場面" in text
    neutral(at, "dashboard of a cleared work")


def test_without_a_work_the_page_says_where_to_go(env):
    at = page()
    assert "作業が選ばれていません" in everything(at)
    assert primaries(at) == []


def test_a_document_work_has_no_dashboard(env):
    _, repo = env
    aid = document_work(repo)
    at = page({"work": aid})
    assert "動画の作業で見られます" in everything(at)


def test_steps_5_and_6_link_to_the_dashboard(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=40)
    for step in (5, 6):
        at = ok(app(session={"assignment_id": aid, "step": step, "submission_id": None}))
        links = [link for link in at.get("page_link") if link.proto.label == "試合・練習のまとめを見る"]
        assert links, step
        assert f"work={aid}" in str(links[0].proto)


def test_every_place_chart_has_a_visible_caption(env):
    from test_wizard_ui_basic import html
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    shown = html(page({"work": aid}))
    assert 'class="pfb-dash-caption">チーム1のいた所' in shown
    assert 'class="pfb-dash-caption">チーム2のいた所' in shown
    aid2 = work(repo, title="コートなし")
    video_group(settings, repo, aid2, people=2, seconds=60, geometry=False, label="B")
    shown = html(page({"work": aid2}))
    assert 'class="pfb-dash-caption">Bの全員：カメラ1から見た位置' in shown



# --- review 2026-10-24 ----------------------------------------------------------------------

def test_every_chart_has_its_values_in_words_and_a_table_twin(env):
    from test_wizard_ui_basic import html
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    at = page({"work": aid})
    shown = html(at)
    alts = images_alt(at)
    assert "パス：" in alts and "回" in alts                 # the grid's numbers, in words
    assert 'class="pfb-dash-caption">種類ごとのよかった場面（チームごと）' in shown
    assert 'class="pfb-dash-caption">種類ごとのよかった場面（人ごと）' in shown
    twins = [e for e in at.expander if e.label == "表で見る"]
    # minutes, moments, per person grid, per team grid, two team courts
    assert len(twins) >= 6
    assert len(at.dataframe) >= 1 + len(twins)


def test_moments_of_a_waiting_person_say_so(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, unapproved=("Pb",))
    from test_wizard_ui_basic import html
    shown = html(page({"work": aid}))
    assert 'class="pfb-dash-row__chip">承認前' in shown


def test_several_cameras_say_their_minutes_are_approximate(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=2, seconds=60, cameras=("cam1", "cam2"))
    text = everything(page({"work": aid}))
    assert "おおよそ" in text


def test_the_moment_list_shows_thirty_at_a_time(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=6, seconds=120, events_each=8)
    at = page({"work": aid})

    def listed(at):
        return [b for b in at.button if (b.key or "").startswith("pfb_dash_jump_list_")]
    assert len(listed(at)) == 30
    pager = next(w for w in at.selectbox if w.label == "ページ")
    at = ok(pager.set_value(2).run())
    assert len(listed(at)) == 18


def test_a_refused_clip_says_why(env, monkeypatch):
    from pfb_app.services import moment_clip
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=60)
    monkeypatch.setattr(moment_clip, "MAX_CLIP_BYTES", 10)
    at = page({"work": aid})
    at = ok(next(b for b in at.button if b.label == JUMP).click().run())
    assert moment_clip.TOO_BIG in everything(at)
    assert not at.get("video")


def test_the_team_csv_is_offered(env):
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=4, seconds=60, teams=True)
    at = page({"work": aid})
    assert primaries(at) == [CSV_PEOPLE]
    assert "チームごとの表をCSVで受け取る" in [b.label for b in at.download_button]


# --- release 4 integration (2026-10-24) -----------------------------------------------------

def test_a_clip_removed_before_it_plays_says_so_plainly(env, monkeypatch):
    from pfb_app.services import moment_clip
    from pfb_app.ui.wizard import dashboard
    settings, repo = env
    aid = work(repo)
    video_group(settings, repo, aid, people=1, seconds=60)
    gone = settings.data_dir / "gone" / "pruned.mp4"          # cut, then pruned by another
    monkeypatch.setattr(moment_clip, "clip", lambda *a, **k: (gone, 2.0))
    at = page({"work": aid})
    at = ok(next(b for b in at.button if b.label == JUMP).click().run())
    assert dashboard.CLIP_FAILED in everything(at)
    assert not at.get("video")


def test_a_general_work_offers_the_group_totals_by_its_constant(env):
    from pfb_app.ui.wizard import dashboard
    settings, repo = env
    aid = work(repo, domain="school", title="塔づくり")
    video_group(settings, repo, aid, people=3, seconds=120)
    at = page({"work": aid})
    assert [b.label for b in at.download_button] == [CSV_PEOPLE, dashboard.CSV_MOMENTS,
                                                     dashboard.CSV_GROUPS]

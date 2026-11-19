"""Progress bar components: plain values in, accessible HTML out (no analysis import)."""
from pfb_app.ui.wizard.bars import (
    STATUS_CHIPS, StageRow, approval_bar_html, card_bar_html, compact_bar_html,
    nav_percent_html, nav_turn_html, overall_bar_html, stage_list_html,
)


def test_overall_bar_says_the_percentage_and_the_time():
    h = overall_bar_html(63, "残り約12分（目安）", "running")
    assert "全体の進み具合 63%" in h and "残り約12分（目安）" in h
    assert '<progress' in h and 'max="100"' in h and 'value="63"' in h and "進行中" in h
    assert "残り" not in overall_bar_html(63, None, "running")


def test_every_status_has_its_word():
    for status, (word, _) in STATUS_CHIPS.items():
        assert word in stage_list_html([StageRow("行動の記録", status, 30, True, "")])
    assert {w for w, _ in STATUS_CHIPS.values()} == {
        "済み", "進行中", "待機中", "順番待ち", "止まっています"}


def test_status_chip_map_is_exact():
    assert STATUS_CHIPS == {
        "done": ("済み", "done"), "running": ("進行中", "processing"),
        "queued": ("待機中", "insufficient"), "waiting_gpu": ("順番待ち", "waiting_gpu"),
        "stopped": ("止まっています", "stopped")}


def test_running_stage_has_its_own_bar_and_note():
    h = stage_list_html([StageRow("準備", "done", 100, True, ""),
                         StageRow("行動の記録", "running", 30, True, "12 / 40 区間")])
    assert h.count("<progress") == 1 and "12 / 40 区間" in h
    assert h.startswith("<ol") and h.count("<li") == 2


def test_waiting_and_stopped_stages_also_get_a_bar():
    for status in ("waiting_gpu", "stopped"):
        h = stage_list_html([StageRow("行動の記録", status, 40, True, "")])
        assert h.count("<progress") == 1 and f"pfb-bar--{status}" in h
    assert "<progress" not in stage_list_html([StageRow("準備", "queued", 0, True, "")])


def test_unknown_progress_is_shown_moving_not_striped():
    h = stage_list_html([StageRow("行動の記録", "running", 0, False, "進んでいます")])
    assert "pfb-bar--moving" in h and "進んでいます" in h
    assert 'aria-valuetext="進んでいます"' in h
    known = stage_list_html([StageRow("行動の記録", "running", 30, True, "")])
    assert "pfb-bar--moving" not in known and "aria-valuetext" not in known


def test_every_bar_repeats_its_text_in_the_aria_label():
    h = overall_bar_html(63, None, "running")
    assert 'aria-label="全体の進み具合 63%"' in h
    assert 'aria-label="1班 63%"' in compact_bar_html("1班", 63, "行動の記録", "running")
    assert 'aria-label="行動の記録・63%"' in card_bar_html(63, "行動の記録", "running")


def test_card_compact_nav_and_approval():
    assert "行動の記録・63%" in card_bar_html(63, "行動の記録", "running")
    c = compact_bar_html("1班", 63, "行動の記録", "waiting_gpu")
    assert "1班" in c and "63%" in c and "pfb-bar--waiting_gpu" in c
    assert "63%" in nav_percent_html(63)
    assert nav_percent_html(63) == (
        '<span class="pfb-navpct"><span class="pfb-sr">進み具合</span>63%</span>')
    a = approval_bar_html(1, 3)
    assert "承認 1 / 全 3" in a and 'value="1"' in a and 'max="3"' in a
    assert "<progress" not in approval_bar_html(0, 0)
    assert "承認 0 / 全 0" in approval_bar_html(0, 0)


def test_approval_bar_keeps_the_group_label():
    a = approval_bar_html(2, 4, "<1班>")
    assert '<span class="pfb-counter__group">&lt;1班&gt;</span>' in a
    assert "pfb-counter__group" not in approval_bar_html(2, 4)


def test_text_is_escaped_and_clamped():
    assert "&lt;b&gt;" in compact_bar_html("<b>", 150, "x", "running")
    assert 'value="100"' in compact_bar_html("a", 150, "x", "running")
    assert 'value="0"' in overall_bar_html(-5, None, "running")
    assert "&lt;i&gt;" in stage_list_html([StageRow("<i>", "running", 5, True, "<i>")])
    assert "<i>" not in stage_list_html([StageRow("<i>", "running", 5, True, "<i>")])


def test_no_internal_words_in_markup():
    h = overall_bar_html(5, None, "running") + stage_list_html(
        [StageRow("行動の記録", "running", 5, True, "")])
    assert "job" not in h and "engine" not in h


def test_the_home_bar_always_has_a_chip_word():
    for status, (word, _) in STATUS_CHIPS.items():
        h = card_bar_html(40, "行動の記録", status)
        label = h[:h.index("</p>")]
        assert word in label and "行動の記録・40%" in label, status


def test_waiting_for_the_coach_says_your_turn_without_a_percentage():
    o = overall_bar_html(40, None, "your_turn")
    assert "あなたの番" in o and "%" not in o and "<progress" in o
    assert 'aria-label="全体の進み具合 あなたの番"' in o
    c = card_bar_html(40, "人の選定", "your_turn")
    assert "あなたの番" in c and "人の選定" in c and "%" not in c
    k = compact_bar_html("1班", 40, "人の選定", "your_turn")
    assert "あなたの番" in k and "%" not in k
    assert nav_turn_html() == '<span class="pfb-navpct">あなたの番</span>'

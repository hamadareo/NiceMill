"""The roster input widget (AppTest): tabs, preview, the role table and validation."""
import json

from streamlit.proto.WidgetStates_pb2 import WidgetState
from streamlit.testing.v1 import AppTest

from pfb_app.ui.wizard.roster_editor import roles_from_table


def script():
    import streamlit as st

    from pfb_app.ui.wizard.roster_editor import roster_input

    r = roster_input("rost", "sports", ["背番号", "ポジション"])
    st.session_state["out"] = None if r is None else {
        "roles": r.roles, "rows": len(r.rows), "place_camera": r.place_camera}


def app() -> AppTest:
    return AppTest.from_function(script, default_timeout=60).run()


def role_editor(at: AppTest):
    found = [df for df in at.dataframe if df.proto.id and "rost_roles" in df.proto.id]
    assert len(found) == 1
    return found[0]


def edit_roles(at: AppTest, changes: dict[int, str]) -> AppTest:
    """Change role selects of the data editor, as the browser would.

    The only use of AppTest internals in this file: streamlit 1.65's AppTest has no setter for
    st.data_editor, so the editor's widget state (the JSON the browser sends) is added to the
    next run's widget states through the private `_tree` / `_run`. If a streamlit upgrade
    breaks this helper, the pure label -> role step stays covered by test_roles_from_table."""
    states = at._tree.get_widget_states()
    ws = WidgetState(id=role_editor(at).proto.id, string_value=json.dumps({
        "edited_rows": {str(i): {"役割": label} for i, label in changes.items()},
        "added_rows": [], "deleted_rows": []}))
    states.widgets.append(ws)
    return at._run(states)


def test_roster_editor_tabs_and_caption():
    at = app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["ファイルから", "貼り付け"]
    assert any(c.value == "背番号、ポジションの列があると、名前を自動で推定できます"
               for c in at.caption)
    assert at.session_state["out"] is None


def test_roster_editor_role_change():
    at = app()
    at.text_area(key="rost_text").input("背番号,名前,メモ\n10,山田,赤\n7,佐藤,青\n").run()
    assert not at.exception
    assert at.session_state["out"]["roles"] == {"背番号": "number", "名前": "name",
                                                "メモ": "ignore"}
    editor = role_editor(at)
    assert list(editor.value["役割"]) == ["背番号", "名前", "使わない"]
    at = edit_roles(at, {2: "外見"})
    assert not at.exception
    assert at.session_state["out"]["roles"] == {"背番号": "number", "名前": "name",
                                                "メモ": "appearance"}
    # a repeated role is an error and nothing is returned
    at = edit_roles(at, {2: "名前"})
    assert at.session_state["out"] is None
    assert [e.value for e in at.error] == ["同じ役割の列が2つあります: 名前（名前、メモ）"]


def test_roster_editor_place_column_camera():
    at = app()
    at.text_area(key="rost_text").input("名前,メモ,初期位置（カメラ2）\na,x,左\n").run()
    assert at.session_state["out"]["place_camera"] == {"初期位置（カメラ2）": "cam2"}
    at = edit_roles(at, {1: "初期位置"})
    assert at.session_state["out"] is not None
    assert at.session_state["out"]["place_camera"] == {"メモ": "cam1", "初期位置（カメラ2）": "cam2"}


def test_roster_editor_file_upload_shift_jis_and_preview():
    at = app()
    rows = "".join(f"{i},選手{i}\r\n" for i in range(1, 9))
    data = ("Ｎｏ．,氏名\r\n" + rows).encode("cp932")
    at.file_uploader[0].set_value(("名簿.csv", data, "text/csv"))
    at.run()
    assert not at.exception
    assert at.session_state["out"] == {"roles": {"Ｎｏ．": "number", "氏名": "name"}, "rows": 8,
                                       "place_camera": {}}
    previews = [df for df in at.dataframe if not df.proto.id
                or "rost_roles" not in df.proto.id]
    assert len(previews[0].value) == 5
    assert any("8人分を読み取りました" in c.value for c in at.caption)


def test_roster_editor_errors():
    at = app()
    at.text_area(key="rost_text").input("ポジション,メモ\nFW,x\n").run()
    assert at.session_state["out"] is None
    assert [e.value for e in at.error] == ["名前か背番号の列を選んでください"]
    at.file_uploader[0].set_value(("名簿.xlsx", b"broken", "application/octet-stream"))
    at.run()
    assert at.session_state["out"] is None
    assert any("Excelの名簿を読めませんでした" in e.value for e in at.error)


def test_roles_from_table():
    rows = [{"列": "No", "例": "10", "役割": "背番号"}, {"列": "メモ", "例": "", "役割": "外見"},
            {"列": "x", "例": "", "役割": None}, {"列": "y", "例": "", "役割": "???"}]
    assert roles_from_table(rows) == {"No": "number", "メモ": "appearance", "x": "ignore",
                                      "y": "ignore"}


def test_roster_editor_empty_roster_warning():
    at = app()
    at.text_area(key="rost_text").input("名前,背番号\n").run()
    assert at.session_state["out"] is None
    assert [w.value for w in at.warning][0].startswith("選手名簿に人が見つかりませんでした")

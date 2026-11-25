"""「団体の設定」 on the PDF step: the organisation's name, logo and colour for the PDFs and
the one-person pages, and the 訂正のご連絡先 of the 「あとから訂正」 handouts
(services.organisation does the checks and the storing)."""
import streamlit as st

from pfb_app.services import organisation as org
from pfb_app.ui.wizard import theme

TITLE = "団体の設定"
GUIDE = ("PDFと一人ずつのページの上に、団体の名前とロゴが入ります。"
         "見出しの色は、読みやすい色の中から選べます。")
SAVE = "団体の設定を保存"
CONTACT = "訂正のご連絡先（入れなくてもかまいません）"
CONTACT_HELP = ("「あとから訂正」の作業のPDFに「訂正のご連絡先：…」として入ります。"
                "入れないと「団体の設定で連絡先を入力」と出ます。")
SAVED = "保存しました。次に受け取るPDFから使われます。"


def render(settings) -> None:
    current = org.load(settings)
    with st.expander(TITLE):
        st.caption(theme.L(GUIDE))
        with st.form("pfb_org_form", border=False):
            name = st.text_input("団体の名前（入れなくてもかまいません）", value=current.name,
                                 max_chars=org.MAX_NAME, key="pfb_org_name")
            keys = list(org.ACCENTS)
            accent = st.radio("見出しの色", keys, index=keys.index(current.accent),
                              format_func=lambda k: org.ACCENTS[k][0], horizontal=True,
                              key="pfb_org_accent")
            logo = st.file_uploader(
                f"ロゴ（PNG・JPEG・SVG、{org.MAX_LOGO_BYTES // 1024}KBまで）",
                type=["png", "jpg", "jpeg", "svg"], key="pfb_org_logo")
            remove = st.checkbox("今のロゴを外す", disabled=current.logo is None,
                                 key="pfb_org_remove")
            contact = st.text_input(CONTACT, value=current.contact, max_chars=org.MAX_CONTACT,
                                    key="pfb_org_contact", help=CONTACT_HELP,
                                    placeholder="例：〇〇クラブ 事務局 03-0000-0000")
            saved = st.form_submit_button(SAVE)
        if saved:
            try:
                org.save(settings, name=name, accent=accent,
                         logo=logo.getvalue() if logo is not None else None,
                         logo_name=logo.name if logo is not None else None, remove_logo=remove,
                         contact=contact)
            except org.OrgError as e:
                st.error(str(e))
            else:
                st.success(SAVED)
                current = org.load(settings)
        if current.logo is not None and current.logo.suffix == ".png":
            st.image(str(current.logo), width=120, caption="今のロゴ")
        elif current.logo is not None:
            st.caption("今のロゴ：SVG")

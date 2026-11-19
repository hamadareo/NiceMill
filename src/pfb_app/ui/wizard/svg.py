"""Inline SVG drawn in Python, shown as an image.

st.html sanitises with the HTML profile only, so an <svg> element itself is dropped; a data
URI <img> is kept. The SVG therefore carries its colours as values (the theme's TOKENS),
and its words go into `alt` (and a <title> inside).
"""
import base64
from html import escape


def img(svg: str, alt: str, cls: str = "") -> str:
    """An <img> showing `svg`, with `alt` as its text alternative."""
    data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    klass = f' class="{escape(cls)}"' if cls else ""
    return (f'<img{klass} src="data:image/svg+xml;base64,{data}" alt="{escape(alt)}" '
            'role="img" decoding="async">')


def doc(width: float, height: float, body: str, title: str) -> str:
    """A standalone SVG document of the given size (its viewBox), titled."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}" '
            f'width="{width:g}" height="{height:g}"><title>{escape(title)}</title>{body}</svg>')

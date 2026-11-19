"""The sample work's pictures, drawn in Python: no photo, no video frame, no face.

A scene is drawn the way a coach draws a play: a half court in chalk on the green board,
the players as round magnets with their numbers (one team blue, the other white), the
player the moment is about ringed in dashed chalk, a pass as a dashed arrow and a run as a
solid one. A player's picture is a plain figure in the team's colour with the number on it.
"""
from html import escape

from pfb_app.services.sample_work import SampleWork, Scene
from pfb_app.ui.wizard import svg
from pfb_app.ui.wizard.theme import TOKENS

W, H = 300.0, 220.0          # the half court's drawing size
_PAD = 12.0
TEAM = {"blue": (TOKENS["sky"], "#FFFFFF"), "white": ("#FFFFFF", TOKENS["board"])}
_CHALK, _CHALK_2, _BOARD = TOKENS["chalk"], TOKENS["chalk-2"], TOKENS["board"]


def _xy(x: float, y: float) -> tuple[float, float]:
    return _PAD + x / 100 * (W - 2 * _PAD), _PAD + y / 100 * (H - 2 * _PAD)


def _court() -> str:
    line = f'fill="none" stroke="{_CHALK_2}" stroke-width="2" stroke-linecap="round"'
    left, right, top, bottom = _PAD, W - _PAD, _PAD, H - _PAD
    hx, hy = W / 2, top + 22
    return (f'<rect width="{W:g}" height="{H:g}" rx="10" fill="{_BOARD}"/>'
            f'<rect x="{left:g}" y="{top:g}" width="{right - left:g}" height="{bottom - top:g}" '
            f'rx="2" {line}/>'
            f'<rect x="{hx - 46:g}" y="{top:g}" width="92" height="88" {line}/>'
            f'<path d="M{hx - 46:g} {top + 88:g} A46 46 0 0 0 {hx + 46:g} {top + 88:g}" {line}/>'
            f'<path d="M{left + 16:g} {top:g} L{left + 16:g} {top + 48:g} '
            f'A124 124 0 0 0 {right - 16:g} {top + 48:g} L{right - 16:g} {top:g}" {line}/>'
            f'<line x1="{hx - 18:g}" y1="{top + 12:g}" x2="{hx + 18:g}" y2="{top + 12:g}" '
            f'stroke="{_CHALK}" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{hx:g}" cy="{hy:g}" r="6" fill="none" stroke="{_CHALK}" '
            'stroke-width="2"/>')


def _arrow(kind: str, start: tuple[float, float], end: tuple[float, float]) -> str:
    (x1, y1), (x2, y2) = _xy(*start), _xy(*end)
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    gap = 15 if kind == "pass" else 4          # a pass ends at the receiver's magnet
    tx, ty = x2 - ux * gap, y2 - uy * gap
    sx, sy = (x1 + ux * 15, y1 + uy * 15) if kind == "pass" else (x1, y1)
    dash = ' stroke-dasharray="7 6"' if kind == "pass" else ""
    head_l = (tx - ux * 9 - uy * 5, ty - uy * 9 + ux * 5)
    head_r = (tx - ux * 9 + uy * 5, ty - uy * 9 - ux * 5)
    return (f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{_CHALK}" '
            f'stroke-width="2.5" stroke-linecap="round"{dash}/>'
            f'<path d="M{head_l[0]:.1f} {head_l[1]:.1f} L{tx:.1f} {ty:.1f} '
            f'L{head_r[0]:.1f} {head_r[1]:.1f}" fill="none" stroke="{_CHALK}" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')


def _magnet(cx: float, cy: float, team: str, number: int, focus: bool) -> str:
    fill, ink = TEAM[team]
    ring = (f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="19" fill="none" stroke="{_CHALK}" '
            'stroke-width="2.5" stroke-dasharray="4 4"/>' if focus else "")
    return (f'{ring}<circle cx="{cx:.1f}" cy="{cy:.1f}" r="12" fill="{fill}" '
            f'stroke="{_BOARD}" stroke-width="2"/>'
            f'<text x="{cx:.1f}" y="{cy + 4.5:.1f}" text-anchor="middle" font-size="13" '
            f'font-weight="700" font-family="sans-serif" fill="{ink}">{number}</text>')


def scene_svg(work: SampleWork, scene: Scene, title: str) -> str:
    """One moment (or the whole court) as a chalk drawing."""
    body = [_court()]
    if scene.arrow is not None:
        body.append(_arrow(scene.arrow.kind, scene.arrow.start, scene.arrow.end))
    for spot in scene.spots:
        body.append(_magnet(*_xy(spot.x, spot.y), work.team_of(spot.id),
                            work.number_of(spot.id), spot.id == scene.focus))
    return svg.doc(W, H, "".join(body), title)


def scene_img(work: SampleWork, scene: Scene, alt: str) -> str:
    return svg.img(scene_svg(work, scene, alt), alt, "pfb-sample__scene")


def player_svg(team: str, number: int, title: str) -> str:
    """A plain figure in the team's colour, the number on the shirt (no face)."""
    fill, ink = TEAM[team]
    body = (f'<rect width="64" height="64" rx="14" fill="{TOKENS["mist"]}"/>'
            f'<circle cx="32" cy="20" r="10" fill="{TOKENS["chalk-2"]}" '
            f'stroke="{TOKENS["muted"]}" stroke-width="1.5"/>'
            f'<path d="M12 60 C12 42 20 34 32 34 C44 34 52 42 52 60 Z" fill="{fill}" '
            f'stroke="{TOKENS["muted"]}" stroke-width="1.5"/>'
            f'<text x="32" y="54" text-anchor="middle" font-size="15" font-weight="700" '
            f'font-family="sans-serif" fill="{ink}">{escape(str(number))}</text>')
    return svg.doc(64, 64, body, title)


def player_img(team: str, number: int, alt: str) -> str:
    return svg.img(player_svg(team, number, alt), alt, "pfb-sample__face")

"""python -m pfb_app.review_video --settings S.yaml SUBMISSION_ID --out DIR [--subject N ...]

Writes, into DIR (not into the app's data), one highlight video per person
(`NN_<name>.mp4`) and the review player (`player/review_player.html` + the video). Opens the
database read only (mode=ro). `--credit` adds an attribution line to the end card and the player (repeatable),
for example the CC BY line of a public video.

Exit status: 0 = something was made, 1 = nothing was made, 2 = the arguments are wrong (a plain
message on stderr, nothing is written)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pfb_app.config import Settings, load_settings
from pfb_app.review_video import service


def _fail(message: str) -> int:
    print(f"エラー: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pfb_app.review_video")
    ap.add_argument("submission_id", type=int)
    ap.add_argument("--settings", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subject", type=int, action="append", default=None)
    ap.add_argument("--credit", action="append", default=[])
    ap.add_argument("--claim-cues", type=Path, default=None,
                    help="pose cues (an external pose tool (not included)): weakens action sentences the cues do not carry")
    ap.add_argument("--eval-texts", type=Path, default=None,
                    help="evaluation texts of everyone (evalgen.py): shown in the player, grouped by team")
    ap.add_argument("--pose-file", type=Path, default=None,
                    help="body poses (an external pose tool (not included)): the 'hands above the head / crouching' words over the heads")
    ap.add_argument("--no-court-estimate", action="store_true",
                    help="do not estimate the court's sidelines from the picture (no red line unless the court is registered)")
    ap.add_argument("--court-reg", type=Path, default=None,
                    help="a court registration (docs/court_reg/*.json: four corners read by a person on one picture); the camera movement is followed by the machine")
    ap.add_argument("--no-player", action="store_true")
    ap.add_argument("--ball", type=Path, default=None,
                    help="ball detections from an external ball-detection tool (not included): draws the ball holder (an estimate) as a red ring")
    ap.add_argument("--others", action="store_true",
                    help="draw thin grey boxes on everyone else in the highlight videos "
                         "(off: referees, bench and onlookers are not boxed)")
    a = ap.parse_args(argv)
    if a.out.exists() and not a.out.is_dir():
        return _fail(f"--out は、フォルダの名前にしてください（{a.out} はファイルです）。")
    for flag, path in (("--settings", a.settings), ("--ball", a.ball), ("--claim-cues", a.claim_cues), ("--eval-texts", a.eval_texts), ("--pose-file", a.pose_file), ("--court-reg", a.court_reg)):
        if path is not None and not path.is_file():
            return _fail(f"{flag} のファイルが見つかりません: {path}")
    from pfb_app.review_video import edit   # noqa: PLC0415
    edit.SHOW_OTHERS = a.others
    settings = load_settings(a.settings) if a.settings else Settings()
    try:
        repo = service.open_readonly(settings)      # read only: no schema step, no WAL switch
    except FileNotFoundError as e:
        return _fail(f"{e}（--settings でNiceMillの設定ファイルを渡してください）")
    try:
        repo.get_submission(a.submission_id)
    except KeyError:
        return _fail(f"解析（ID {a.submission_id}）が、このデータベースにありません。")
    if a.claim_cues is None:
        print("ヒント: --claim-cues を付けると、動作の確認（文の動作が姿勢の手がかりと合うか）をして、"
              "合わない文を断定しない形にします。付けないときは、文はAIの記録のままです。", file=sys.stderr)
    t0 = time.time()
    result = service.make_all(repo, settings, a.submission_id, a.out, subject_ids=a.subject,
                              credit_lines=a.credit, with_player=not a.no_player,
                              ball_file=str(a.ball) if a.ball else None,
                              claim_cues=str(a.claim_cues) if a.claim_cues else None,
                              eval_texts=str(a.eval_texts) if a.eval_texts else None,
                              pose_file=str(a.pose_file) if a.pose_file else None,
                              estimate_court=not a.no_court_estimate,
                              court_reg=str(a.court_reg) if a.court_reg else None)
    result["seconds_to_make"] = round(time.time() - t0, 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    made = any("file" in v for v in result["videos"]) or bool(result.get("player"))
    if not made:
        print("エラー: 作れたものがありません（理由は notes と videos の skipped / error）。", file=sys.stderr)
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())

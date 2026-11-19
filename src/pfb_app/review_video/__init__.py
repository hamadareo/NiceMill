"""振り返り用の編集動画と振り返りプレーヤー.

The model's output (each person's recorded behaviour, the moments it cites, the drawn person
boxes, the generated sentences) is used to EDIT the original match video with ordinary video
tools (cut, subtitle, box, marker). Nothing here generates or alters footage with a model, and
the video never leaves the machine: the files are local only.

- `model`   : Package / Person / Moment (the data every output is made from)
- `collect` : the app's database -> Package (`build_package`)
- `edit`    : a person's highlight video (`render_highlight`)
- `player`  : one local HTML + the video (`build_player`)
- `service` : the calls a screen / worker / CLI makes
- `__main__`: python -m pfb_app.review_video
"""

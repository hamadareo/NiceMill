# HTTPS で開く（tailscale serve）

PFB アプリを `https://myhost.example.ts.net/` で開けるようにする手順です。
アプリ自体は `127.0.0.1` だけで待ち受け、Tailscale がその前に立って HTTPS にします。
証明書の更新も Tailscale が自動で行います。tailnet の外からは今までどおり見えません。

## HTTPS にすると使えるようになるもの

| 機能 | http（今） | https |
|---|---|---|
| 解析が終わったときの端末の通知（`ui/wizard/notify.py`） | 使えない（音だけ） | 使える（ホームの「終わったら音で知らせる」をオンにすると許可を聞く） |
| 送信中に画面を消さない（Wake Lock、`ui/wizard/big_upload.py`） | 使えない（スマホの画面が消えると送信が止まり、戻ると続きから） | 使える |
| ホーム画面に追加（PWA） | 使えない | 使える（別作業） |
| 途中から続けて送る大きな動画の送信（`/pfb-upload`） | 使える | 使える（同じアドレスで動く） |
| 会社や学校の IT の確認 | 「暗号化なし」と指摘される | 通る |

通信は今も Tailscale（WireGuard）の中で暗号化されていますが、ブラウザーからは http に見えるため、上の機能が止められています。

## 事前に一度だけ（管理者）

1. Tailscale の管理画面（DNS のページ）で **HTTPS Certificates** をオンにする。
   - 証明書を取ると、マシン名 `myhost.example.ts.net` が公開の Certificate Transparency ログに載ります（中身や IP は載りません）。気になる場合はマシン名を変えてから行います。
2. サーバーで、アプリを動かすユーザーに serve の設定を許す: `sudo tailscale set --operator=user`
3. 確認（何も変えません）: `scripts/https_serve.sh check`
   - 2026-10-23 時点の結果: 「HTTPS certificates are off」「this user may not change」の 2 つが NG。どちらも上の 1 と 2 で直ります。

## 切り替え（デプロイの一部：利用者の承認を得てから）

```bash
cd ~/pfb_app
scripts/stop.sh                                   # 解析中のものがないことを先に確かめる（stop.sh の注意書き）
PFB_UI_HOST=127.0.0.1 scripts/start.sh            # 画面は 127.0.0.1:8501 だけで待ち受け
scripts/https_serve.sh on                         # https:443 -> http://127.0.0.1:8501 を追加
curl -s -o /dev/null -w '%{http_code}\n' https://myhost.example.ts.net/   # 200
```

- `https_serve.sh on` は、マシン全体で共有の tailscale serve の設定を書き換えます（デプロイの一部で、承認が要ります）。
  指定した HTTPS ポート（既定 443）の `/` がすでにほかのもの（`http://127.0.0.1:8501` 以外）に使われていれば、
  何も変えずに止まります。同じ割り当てがすでにあれば、そのままにします。
- `https_serve.sh off` は、`/` がちょうど `http://127.0.0.1:8501` のときだけ外します（ほかのものは外しません）。
- `check` は、アプリが 127.0.0.1 以外でも待ち受けていると NG にします（`PFB_UI_HOST=127.0.0.1` で起動し直す）。
- 以前のアドレス `http://<server-address>:8501` は使えなくなります。利用者にブックマークの変更を伝えてください。
- systemd で常駐させる場合は、ユニットを書くときに同じく指定します: `PFB_UI_HOST=127.0.0.1 scripts/install_user_services.sh --enable`（`--host 127.0.0.1` でも同じ。`PFB_UI_PORT` / `--port` も読みます）。アドレスはユニットに書き込まれるので、その後は毎回つける必要はありません。

## 元に戻す

```bash
scripts/https_serve.sh off                        # 443 の割り当てだけを外す
scripts/stop.sh && scripts/start.sh               # 既定（<server-address>:8501、http）に戻る
```

## 試すとき

テスト用の作業ツリーでは、自分のポート（8640〜8649）で `127.0.0.1` に立てます。
**下の `on` / `off` も、マシン全体で共有の tailscale serve の設定を変えます。テスト用のポートでも、
本番の切り替えと同じく承認を得てから行います**（`check` は何も変えません）。

```bash
PFB_CONFIG=<テスト用の設定> .venv/bin/streamlit run src/pfb_app/ui/serve.py \
    --server.address 127.0.0.1 --server.port 8640 --server.headless true
scripts/https_serve.sh check --port 8640
# 証明書と operator がそろっていれば、本番の 443 とは別のポートで:
scripts/https_serve.sh on --port 8640 --https-port 8443   # https://myhost.example.ts.net:8443/
scripts/https_serve.sh off --https-port 8443
```

2026-10-23 の phone-first レーンでは、上の check が NG（証明書オフ、operator なし）だったため、
tailscale serve での確認はしていません。`127.0.0.1:8640` での動作（画面、`/pfb-upload`、
途中からの再開）は SSH のトンネル越しに確かめています。

## 仕組みのメモ

- 画面と送信の窓口は同じサーバーです（`src/pfb_app/ui/serve.py` の `st.App`）。`tailscale serve`
  の割り当ては 1 つで足ります。
- 送信の窓口は、ページが渡す署名つきの合言葉（24 時間）と、送り元のサイトが同じことを確かめます
  （`services/chunked_upload.py`、`ui/upload_api.py`）。`tailscale serve` は元のホスト名を
  `X-Forwarded-Host` で渡すので、127.0.0.1 から来たときだけ同じサイトとして扱います。
- **ログインはまだありません。画面を開ける人（tailnet の中の端末から、このアドレスに届く人）なら、
  だれでも動画を送れます。** 合言葉は「画面を開いた」ことの印で、本人の確認ではありません。
  ログイン（監査 #7）が入るまでは、tailnet に入れる人を絞ることが守りになります。
- `PFB_UI_ENTRY=app scripts/start.sh` で、以前と同じ `ui/app.py` だけの起動にもできます（このときは
  Streamlit の送信欄に戻ります）。
- systemd で動かすとき（`scripts/install_user_services.sh`）も同じです。画面のユニットは既定で
  `ui/serve.py` を起動します（`--entry app` で `ui/app.py` だけ）。HTTPS のときは
  `--host 127.0.0.1` を付けて書き直してください。

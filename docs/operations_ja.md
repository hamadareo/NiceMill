# ナイスミル　運用ガイド（管理者向け）

サーバー（ナイスミルが動いているパソコン）を、開発者がいなくても動かし続けるための手引きです。
コマンドは、サーバーに入って（`ssh user@server`）から実行します。アプリのフォルダは `~/pfb_app` です。

## 1. 開く・元気かどうかを知る

- 画面: ブラウザーで `http://<server-address>:8501` を開きます（Tailscale につながっている端末から）。
- 元気かどうか: 画面の「管理用の画面」→「処理状況」→「運用の状態」を見ます。または、サーバーで次を実行します。

```bash
~/pfb_app/scripts/ops/check_ops.sh
```

`[OK]` だけなら問題ありません。`[注意]` は確かめたほうがよいこと、`[要対応]` は対応が必要なことです。
画面が開けないときでも、`cat ~/pfb_app/data/ops_status.txt` で、5分ごとに更新される最新の状態が読めます。

## 2. 起動方法（自動起動）

- **自動起動**: サーバーを再起動したり停電から戻ったりしても、ナイスミルが自分で起動します（systemd という仕組み）。止まっても自動で再起動します。
- **手動（tmux）**: 自動起動を入れる前の方法です。再起動すると止まったままになります。

どちらの方法で動いているかは `check_ops.sh` の最初の行に出ます。二つを同時に使ってはいけません（二重に動くと壊れます）。切り替えは次のスクリプトだけで行います（先に何が起きるかを見せるだけで、`--apply` を付けるまで何も変えません）。

```bash
~/pfb_app/scripts/ops/install_services.sh install              # 予定を見る
loginctl enable-linger user                                   # 一度だけ（再起動後も動かす設定）
~/pfb_app/scripts/ops/install_services.sh install --apply --stop-tmux
~/pfb_app/scripts/ops/install_services.sh uninstall --apply --start-tmux   # 元の手動方式に戻す
```

うまく起動しなかったときは、自動で元に戻ります。

## 3. 止める・始める・再起動する

自動起動のとき（普段はこちら）:

```bash
systemctl --user restart pfb-worker pfb-ui     # 再起動（解析中なら終わるまで最大3分待ちます）
systemctl --user stop pfb-ui pfb-worker        # 止める
systemctl --user start pfb-worker pfb-ui       # 始める
```

手動（tmux）のとき: `~/pfb_app/scripts/stop.sh` で止め、`PFB_CONFIG=~/pfb_app/config/settings.yaml ~/pfb_app/scripts/start.sh` で始めます。
解析の途中で止めても、動画や承認は消えません。止めた解析は、起動後に自動でやり直します。

## 4. 困ったとき（よくある3つ）

1. **画面が開かない** → `check_ops.sh` を実行。「画面が応答しません」なら `systemctl --user restart pfb-ui`。1分待って開き直します。Tailscale につながっているかも確かめてください。
2. **解析が進まない（「ワーカーが止まっています」）** → `systemctl --user restart pfb-worker`。2〜3分で「運用の状態」が「動いています」に変わります。変わらないときは `journalctl --user -u pfb-worker -n 50` の最後の行を、連絡先に送ってください。
3. **「保存場所の空きが少ない」と出る** → 画面で、使い終わった作業を削除します（動画のコピーなどが消えて空きが増えます）。空きが約15GBを切ると、新しい動画や書類を受け付けません。

停電・再起動のあとは、数分待ってから `check_ops.sh` を実行してください。自動起動が入っていれば、何もしなくても元に戻ります。

## 5. バックアップと復元

- データベース（文章・承認の記録）のコピーは、毎晩 3:30 頃に自動で作ります。場所は `~/pfb_app/backups/`。標準では、直近14日分と、そのあと4週間分（週に1つ）を残します。
- **動画・画像・書類のファイルはバックアップしません**（大きいため）。元の動画は受け取りフォルダや撮影した端末に残してください。
- 一覧・確認・復元の試験（本物のデータには触りません。月に一度の実行をおすすめします）:

```bash
~/pfb_app/scripts/backup.sh list
~/pfb_app/scripts/backup.sh verify
~/pfb_app/scripts/backup.sh rehearse
```

- 復元（データベースを昔の状態に戻す。最後の手段です）:

```bash
systemctl --user stop pfb-ui pfb-worker
sleep 120                                             # 止めてから約2分待つ（早いと「動いています」と断られます）
~/pfb_app/scripts/restore.sh app-20261114-033000      # list で出た名前
systemctl --user start pfb-worker pfb-ui
```

戻す前のデータベースは `data/app.db.before-restore-…` として残ります。

## 6. ログインなしで使えるモード

いまは「ログインなし」で動いています。**Tailscale につながっている端末から、このアドレスを開ける人は誰でも使えます。**
このアドレスを、公開のリンクや、つながりのない人に教えないでください。また、インターネットに公開する設定（Funnel など）は、ログインを有効にするまで使わないでください。

## 6-2. 「ききたいことを聞く」の答えを作るAI

この画面は、同じ機械の中で動いている文章用のAI（OpenAI互換のサーバー）に質問を送ります。外のサービスには送りません（アドレスは localhost だけ受け付けます）。設定がないと、画面は「いまは質問に答える準備ができていません」と出し、ほかの画面には影響しません。

`config/settings.yaml` に次を書くと使えます（書かなければ使えない状態のままです）。

```yaml
ask_server: http://127.0.0.1:1234/v1   # 文章用のAIのサーバー
ask_model: モデルの名前                     # 載せてあるモデルの名前。省略すると、サーバーが最初に返す名前
ask_timeout_sec: 120                      # 1回の返事を待つ秒数
```

- 使うモデルは、先にサーバーに載せておいてください（LM Studio のように、載せていないモデルも一覧に出るサーバーでは、`ask_model` に載せたモデルの名前を書くのが確実です）。
- 考えてから答えるモデルは、1回の返事に十数秒かかることがあります。
- 質問と答えは保存しません。ログにも質問の中身は書きません。

## 7. 連絡先

（ここに、開発担当の連絡先を書いてください）
連絡するときは、`check_ops.sh` の出力と、`journalctl --user -u pfb-worker -n 50` の最後の行を添えると、早く直せます。

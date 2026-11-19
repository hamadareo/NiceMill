"""待ち時間の目安が実測とどれくらい合うかを、記録のコピーで確かめる（読むだけ）。

使い方:  PYTHONPATH=src python scripts/eta_backtest.py <app.db のコピー>

本番のデータベースそのものではなく、sqlite の backup で作ったコピーを渡してください。
各グループを1つずつ除き、残りの実績から「動画の長さを使った目安」と「これまでの中央値」の
2通りで、そのグループの解析時間の合計を当てて、実測との差を表にします。
"""
import sqlite3
import statistics
import sys

from pfb_app.services import eta
from pfb_app.storage.db import Repo


def main(path: str) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    repo = Repo(conn, None)
    for engine in ("pfb3x3", "paper", "document"):
        checks = eta.backtest(repo, engine)
        if not checks:
            continue
        print(f"[{engine}] {len(checks)} groups")
        print("  group   actual(s)  by_length(s)  by_median(s)  err_len  err_med")
        for c in checks:
            print(f"  {c.submission_id:5d}  {c.actual:9.0f}  {c.by_length:12.0f}"
                  f"  {c.by_median:12.0f}  {c.error_length:7.0%}  {c.error_median:7.0%}")
        print("  median error: by length %.0f%%, by median %.0f%%" % (
            100 * statistics.median(c.error_length for c in checks),
            100 * statistics.median(c.error_median for c in checks)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))

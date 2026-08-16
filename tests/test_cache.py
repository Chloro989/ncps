"""
一度読んだものを覚えておく仕組みの試験。

    python tests/test_cache.py

大事なのは二つ。速くなることと、**書き直したら読み直すこと**。
古いものを返し続ける覚え書きは、無いほうがましである。
"""

import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion import cache

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


print("== 覚えて引く ==")
memo = cache.Memo(limit=3)
calls = []


def make(value):
    def made():
        calls.append(value)
        return value
    return made


first, remembered = memo.fetch("あ", make(1))
check("最初は作る", first == 1 and not remembered)
again, remembered = memo.fetch("あ", make(99))
check("二度目は覚えていたものを返す", again == 1 and remembered)
check("二度目は作らない", calls == [1], str(calls))

check("当たりと外れを数える",
      memo.state()["当たり"] == 1 and memo.state()["外れ"] == 1,
      str(memo.state()))


print("\n== 溢れさせる ==")
small = cache.Memo(limit=3)
for number in range(5):
    small.put(f"鍵{number}", number)
check("上限を守る", small.state()["覚えている数"] == 3,
      str(small.state()))
check("古いものから捨てる", small.get("鍵0") is None)
check("新しいものは残る", small.get("鍵4") == 4)

# 引いたものは新しい扱いにする。よく使うものが捨てられないように
order = cache.Memo(limit=2)
order.put("古", 1)
order.put("新", 2)
order.get("古")              # 触ったので新しい扱いになる
order.put("もっと新", 3)
check("触ったものは残る", order.get("古") == 1)
check("触らなかったものが捨てられる", order.get("新") is None)

memo.clear()
check("忘れられる", memo.state()["覚えている数"] == 0)


print("\n== 鍵の作り方 ==")
check("同じ本文は同じ鍵", cache.digest("あいう") == cache.digest("あいう"))
check("違う本文は違う鍵", cache.digest("あいう") != cache.digest("あいえ"))
check("鍵は短くまとめる", len(cache.digest("あ")) == 16)

with TemporaryDirectory() as folder:
    path = Path(folder) / "原稿.txt"
    path.write_text("　最初の本文である。", encoding="utf-8")
    before = cache.file_key(path)
    check("同じファイルは同じ鍵", cache.file_key(path) == before)

    # ここが要。書き直したのに古いものを返し続けると、
    # 直した原稿を直す前の姿で論評することになる
    time.sleep(0.01)
    path.write_text("　書き直した本文である。長さも変えておく。",
                   encoding="utf-8")
    after = cache.file_key(path)
    check("書き直せば鍵が変わる", after != before,
          f"{before} 対 {after}")

    other = Path(folder) / "別の原稿.txt"
    other.write_text("　最初の本文である。", encoding="utf-8")
    check("中身が同じでも別のファイルは別の鍵",
          cache.file_key(other) != cache.file_key(path))

    check("道は絶対のものにする", Path(cache.file_key(path)[0]).is_absolute())

print("\n== 画面からの求めに対応する鍵 ==")
check("貼り付けは中身で見分ける",
      cache.key_for({"text": "あ"}) != cache.key_for({"text": "い"}))
check("同じ貼り付けは同じ鍵",
      cache.key_for({"text": "あ"}) == cache.key_for({"text": " あ "}),
      str(cache.key_for({"text": " あ "})))
check("置き場のものは名前で見分ける",
      cache.key_for({"name": "甲.txt"}) != cache.key_for({"name": "乙.txt"}))
check("貼り付けがあればそちらを優先する",
      cache.key_for({"text": "あ", "name": "甲.txt"})[0] == "貼り付け")
check("空の貼り付けは置き場を見る",
      cache.key_for({"text": "   ", "name": "甲.txt"})[0] == "置き場")


print("\n== 全部忘れる ==")
cache.manuscripts.put("試験", "何か")
cache.structures.put("試験", "何か")
cache.forget()
check("原稿を忘れる", cache.manuscripts.get("試験") is None)
check("解析も忘れる", cache.structures.get("試験") is None)
check("様子を訊ける", set(cache.state()) == {"原稿", "解析"},
      str(cache.state()))


print("\n== 同時に触られても壊れない ==")
# 画面は複数の求めを同時に捌く。錠が無いと覚え書きが壊れる
import threading

shared = cache.Memo(limit=50)
errors = []


def hammer(seed):
    try:
        for number in range(200):
            shared.put(f"鍵{(seed * 200 + number) % 60}", number)
            shared.get(f"鍵{number % 60}")
    except Exception as problem:      # noqa: BLE001 — 何が来ても記録する
        errors.append(problem)


threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
check("壊れない", not errors, str(errors[:2]))
check("上限は守られたまま", shared.state()["覚えている数"] <= 50,
      str(shared.state()))


print(f"\n{passed}件通過 / {failed}件失敗")
sys.exit(1 if failed else 0)

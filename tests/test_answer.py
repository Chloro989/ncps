"""
答えの照合と添削ファイルの試験。

    python tests/test_answer.py

段落番号の実在だけを見る検査は弱い。Qwen2.5-3B に104段落の原稿を
読ませた答えでは、番号はすべて実在して検査は素通りしたが、
引用14件のうち8件が別の段落の中身で、うち3件は本文に無い文だった。
この試験は、その取り違えを捕まえ続けるためのもの。
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion.answer import (MARK_BAD, MARK_OK, annotate, attach,
                              find_quotes, report_quotes)
from centurion.manuscript import Manuscript

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


work = Manuscript("\n\n".join([
    "　改札を抜けると、売店の灯りがまだついていた。",
    "　雨は上がっていたが、路面には水たまりが残っていた。",
    "　彼女は「行こう」と言って、先に歩き出した。",
    "　橋の上で立ち止まり、水路の暗がりを覗きこんだ。",
]))

print("== 引用の照合 ==")
quotes = find_quotes("[0] 「改札を抜けると、売店の灯りがまだついていた。」", work)
check("一致する引用を通す", len(quotes) == 1 and quotes[0].ok)
check("一致なら出どころを探さない", quotes[0].invented is False)

quotes = find_quotes("[3] 「改札を抜けると、売店の灯りがまだついていた。」", work)
check("番号違いを見つける", len(quotes) == 1 and not quotes[0].ok)
check("本当の出どころを示す", quotes[0].home == [0], str(quotes[0].home))
check("捏造とは区別する", quotes[0].invented is False)
check("表示に本当の番号が入る", "[0]" in str(quotes[0]), str(quotes[0]))

quotes = find_quotes("[1] 「冷蔵庫の扉を開けると、豆腐が音を立てていた。」", work)
check("本文に無い引用を見つける", quotes[0].invented)
check("捏造だと分かる言葉で伝える", "本文に無い" in str(quotes[0]))

quotes = find_quotes("[2] 「彼女は「行こう」と言って、先に歩き出した。」", work)
check("入れ子の括弧を扱える", len(quotes) == 1 and quotes[0].ok,
      str(quotes[0]) if quotes else "拾えず")

check("観点の名前は引用として数えない",
      find_quotes("段落 [0] に「【一度きり】」の観点を当てる", work) == [])
# 日本語は語をそのまま鉤括弧で括る。短いものまで引用として突き合わせると
# 正しい指摘に×が付く。実際に作中語を括った指摘3件へ誤って×を付けていた
check("短い語の括りは引用として数えない",
      find_quotes("[0] ~ [3]: 「あっちゃぐり」の規則とその背景。", work) == [],
      str(find_quotes("[0] ~ [3]: 「あっちゃぐり」の規則。", work)))
check("十分に長い引用は数える",
      len(find_quotes("[0] 「改札を抜けると、売店の灯りがまだ」", work)) == 1)
check("番号が無い行は数えない", find_quotes("「改札を抜けると」", work) == [])
check("引用が無い行は数えない", find_quotes("[0] は良い。", work) == [])
check("短すぎる引用は数えない", find_quotes("[0] 「あ」", work) == [])
check("存在しない番号でも落ちない",
      find_quotes("[999] 「改札を抜けると、売店の灯りが」", work)[0].ok is False)

print("\n== まとめの文 ==")
mixed = ("[0] 「改札を抜けると、売店の灯りがまだついていた。」\n"
         "[1] 「冷蔵庫の扉を開けると、豆腐が音を立てていた。」\n"
         "[3] 「雨は上がっていたが、路面には水たまりが残っていた。」")
summary = report_quotes(find_quotes(mixed, work))
check("件数を数える", "3件中 一致1件 / 不一致2件" in summary, summary[:40])
check("捏造の数を伝える", "1件は本文に存在しない" in summary)
check("引用が無ければそう言う",
      "引用が見当たらない" in report_quotes([]))

print("\n== 段落への振り分け ==")
preamble, notes = attach("まず全体について。\n[1] は冗長である。\n"
                         "[0] と [3] は響き合っている。", work)
check("番号の無い行は前置きへ", preamble == ["まず全体について。"])
check("番号のある行は段落へ", notes[1] == ["[1] は冗長である。"])
check("複数の番号があれば両方に付く",
      notes[0] == notes[3] == ["[0] と [3] は響き合っている。"])
check("範囲外の番号は前置きへ",
      attach("[999] は冗長である。", work)[0] == ["[999] は冗長である。"])

# 範囲指定を両端に貼ると、離れた二箇所に同じ指摘が現れて紛らわしい
_, spanned = attach("[0] ~ [3]: 前半の流れについて。", work)
check("範囲は先頭にだけ貼る", set(spanned) == {0}, str(sorted(spanned)))
for mark in ("〜", "～", "-"):
    _, spanned = attach(f"[1] {mark} [3]: 中ほどについて。", work)
    check(f"区切りが {mark} でも範囲と分かる", set(spanned) == {1},
          str(sorted(spanned)))

print("\n== 添削ファイル ==")
# 何のモードで、どのモデルに解かせた添削なのかが残っていないと、
# 溜まった添削を並べて比べられない
RECORDS = [("日付", "2026-08-12 13:00"), ("モード", "発想"),
           ("モデル", "claude-sonnet-5 (API)"), ("観点", "視点／熱量"),
           ("空の項目", "")]
text = annotate(mixed, work, records=RECORDS)
check("題が入る", text.startswith("# "))
check("モードが入る", "# モード: 発想" in text)
check("モデルが入る", "# モデル: claude-sonnet-5 (API)" in text)
check("日付が入る", "# 日付: 2026-08-12 13:00" in text)
check("観点が入る", "視点／熱量" in text)
check("中身の無い項目は出さない", "空の項目" not in text)
check("記録は照合より先に来る",
      text.index("# モード") < text.index("# 引用の照合"))
check("記録が無くても作れる", annotate(mixed, work).startswith("# "))
check("照合の件数が入る", "一致1件" in text)
check("捨てるべき指摘があると警告する", "捨てること" in text)
check("本文が全段落そのまま入る",
      all(p.text in text for p in work.paragraphs))
check("段落番号が付く", "[0] " in text and "[3] " in text)
check("一致した指摘に印が付く",
      any(line.strip().startswith(MARK_OK) for line in text.splitlines()))
check("不一致の指摘に別の印が付く",
      any(line.strip().startswith(MARK_BAD) for line in text.splitlines()))

lines = text.splitlines()
place = next(i for i, line in enumerate(lines) if line.startswith("[1] "))
check("指摘は本文のすぐ下に来る",
      lines[place + 1].strip().startswith(MARK_BAD),
      lines[place + 1][:30])

plain = annotate("全体としてよく書けている。", work)
check("番号の無い答えでも添削を作れる", "段落を指していない指摘" in plain)
check("そのときも本文は全部入る",
      all(p.text in plain for p in work.paragraphs))

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

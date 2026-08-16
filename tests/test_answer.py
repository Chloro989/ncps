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

from centurion.answer import (MARK_BAD, MARK_OK, anchoring, annotate, attach,
                              check_scores, claimed_total, find_quotes,
                              report_anchoring, report_quotes, scores)
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

print("\n== 錨の数 ==")
# 引用の照合は「引用があるもの」しか見られない。どこも指さない指摘ばかりの
# 答えは、照合を素通りして「不一致0件」と出てしまう。実際にそうなった
anchored, total = anchoring(
    "[0] は良い。\n[1] も良い。\n全体として、静寂と血流の揺らぎがある。", work)
check("指摘の総数を数える", total == 3, str(total))
check("原稿を指している数を数える", anchored == 2, str(anchored))

anchored, total = anchoring(
    "静寂と微細な血流の揺らぎに集約される。\n"
    "未完成の呪文が読者の潜在的記憶に埋め込まれる。\n"
    "言葉にならない時間の裂け目である。", work)
check("どこも指していなければ0", anchored == 0)
summary = report_anchoring(anchored, total)
check("割合を出す", "0/3件 (0%)" in summary, summary[:40])
check("半分を割れば警告する", "半分も原稿を指していない" in summary)
check("どう直せばよいかを言う", "観点を減らす" in summary)

anchored, total = anchoring("[0] も [1] も良い。[3] は冗長。", work)
check("錨があれば警告しない",
      "半分も" not in report_anchoring(anchored, total))
check("見出しは数えない", anchoring("### 熱量\n[0] は良い。", work)[1] == 1)
check("短い行は数えない", anchoring("うん\n[0] は良いと思われる。", work)[1] == 1)
check("空の答えなら0件", anchoring("", work) == (0, 0))
check("0件なら何も言わない", report_anchoring(0, 0) == "")

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

print("\n== 採点の検算 ==")
# 実測: LFM2.5-1.2B が 4+3+5+5+4+5+4=30 を「合計点 16/35」と書いた。
# 引用の照合と同じで、これは本文を読まずに確かめられる
AXES = ("構成・プロット", "キャラクター", "文章・文体", "描写",
        "対話", "テーマ・主題", "世界観・設定")
SOUND = """# 講評
## 1. 構成・プロット: 4/5
## 2. キャラクター: 3/5
## 3. 文章・文体: 5/5
## 4. 描写: 5/5
## 5. 対話: 4/5
## 6. テーマ・主題: 5/5
## 7. 世界観・設定: 4/5
## 総評
合計点 30/35。"""

found = scores(SOUND)
check("七観点すべて拾う", len(found) == 7, str(len(found)))
check("番号を拾う", [n for n, _, _ in found] == list(range(1, 8)))
check("観点名を拾う", found[0][1] == "構成・プロット")
check("点数を拾う", [p for _, _, p in found] == [4, 3, 5, 5, 4, 5, 4])
check("合計を拾う", claimed_total(SOUND) == 30)

check("合っていれば一行だけ", len(check_scores(SOUND, AXES)) == 1,
      str(check_scores(SOUND, AXES)))
check("合計を出す", "合計 30点" in check_scores(SOUND, AXES)[0])

wrong = SOUND.replace("合計点 30/35", "合計点 16/35")
lines = check_scores(wrong, AXES)
check("合計の食い違いを見つける",
      any("合計が合わない" in line for line in lines), str(lines))
check("正しい和を示す", any("和は30点" in line for line in lines))
check("書かれた合計も示す", any("16点" in line for line in lines))

lines = check_scores(SOUND.replace("\n合計点 30/35。", ""), AXES)
check("合計が無ければ言う",
      any("合計点が書かれていない" in line for line in lines), str(lines))

short = "\n".join(SOUND.splitlines()[:4]) + "\n合計点 7/35。"
lines = check_scores(short, AXES)
check("抜けた観点を名指しする",
      any("描写" in line and "抜けている" in line for line in lines),
      str(lines))

over = SOUND.replace("## 4. 描写: 5/5", "## 4. 描写: 7/5")
check("範囲外の点数を見つける",
      any("範囲外" in line for line in check_scores(over, AXES)))

check("採点の形でなければそう言う",
      "採点の形になっていない" in check_scores("よく書けている。", AXES)[0])

# 見出し記号やコロンの揺れで取りこぼすと、検算が黙って素通りする
for shape in ["### 1. 構成・プロット：4/5", "1. 構成・プロット: 4/5",
              "## 1．構成・プロット: 4 / 5", "**1. 構成・プロット**: 4/5"]:
    check(f"{shape[:14]}… の形も読める", len(scores(shape)) == 1, shape)
check("全角の合計も読める", claimed_total("合計点 30／35") == 30)
check("合計点の直後に語が続いても読める",
      claimed_total("合計点: 21/35。作品は…") == 21)

# 添削ファイルの見出しにも出す
graded = annotate(wrong, work, axes=AXES)
check("添削ファイルに検算が入る", "合計が合わない" in graded)
check("採点でなければ検算しない", "合計が合わない" not in annotate(wrong, work))


print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

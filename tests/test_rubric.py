"""
厳しさと、外部ファイルからの文面の試験。

    python tests/test_rubric.py

見ているのは二つ。

  1. 厳しさが規則を実際に入れ替えているか
     選べるようにしても中身が同じなら、選ばせる意味がない
  2. prompts/ に置いたファイルが本当に使われるか
     「編集できます」と書いておいて効かないのが一番悪い
"""

import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion import review, rubric, wording
from centurion.manuscript import Manuscript
from centurion.review import IDEA, REVIEW, SCORE, build_prompt

FIXTURES = HERE / "fixtures"
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


novel = Manuscript.load(FIXTURES / "sample_novel.txt")
chunk = novel.chunks(size=6000, overlap=1)[0]
lenses = review.lenses_for(REVIEW)[:3]


print("== ルーブリックの形 ==")
check("厳しさは三段階", rubric.SEVERITIES == ("育成", "標準", "厳格"))
check("観点は七つ", len(rubric.AXES) == 7, str(len(rubric.AXES)))
for severity in rubric.SEVERITIES:
    levels = rubric.LEVELS[severity]
    check(f"{severity}に七観点そろっている",
          set(levels) == set(rubric.AXES),
          str(set(rubric.AXES) - set(levels)))
    check(f"{severity}はどの観点も五段階",
          all(len(levels[axis]) == 5 for axis in rubric.AXES),
          str({a: len(levels[a]) for a in rubric.AXES
               if len(levels[a]) != 5}))
    check(f"{severity}に空の水準が無い",
          all(text.strip() for axis in rubric.AXES
              for text in levels[axis]))

# 育成と厳格は基準の置き方が逆。同じ文面なら選ばせる意味がない
check("育成と厳格の水準はすべて違う",
      all(rubric.LEVELS["育成"][axis] != rubric.LEVELS["厳格"][axis]
          for axis in rubric.AXES))
check("育成は評価3を健闘とする",
      "十分な水準" in rubric.LEVELS["育成"]["構成・プロット"][2])
check("厳格は評価1を商業水準とする",
      "商業作品の水準" in rubric.LEVELS["厳格"]["構成・プロット"][0])


print("\n== 厳しさが規則を入れ替える ==")
heads = {severity: build_prompt(chunk, lenses, mode=REVIEW,
                                severity=severity, directory=str(HERE / "無"))[0]
         for severity in rubric.SEVERITIES}
check("三つとも違う文面", len(set(heads.values())) == 3)
check("育成は良い点を先に挙げさせる", "良かった点を必ず" in heads["育成"])
check("厳格は迷ったら低い方を選ばせる", "低い方の評価" in heads["厳格"])
check("厳格は破綻の無さを加点にしない", "加点理由にならない" in heads["厳格"])
check("育成は忖度をしないと言わない",
      "忖度" not in heads["育成"])
check("標準は忖度をしないと言う", "忖度" in heads["標準"])

# 幻覚を防ぐ規則は、どの厳しさでも消えてはならない。
# 育成で「良い点を先に」を足したときに、この行を落とすと
# 段落番号の検査が働かなくなる
for severity in rubric.SEVERITIES:
    check(f"{severity}でも段落番号を求める",
          "段落番号" in heads[severity])
    check(f"{severity}でも本文に無い要素を禁じる",
          "本文に無い要素" in heads[severity])

try:
    build_prompt(chunk, lenses, mode=REVIEW, severity="超厳格")
    refused = ""
except ValueError as problem:
    refused = str(problem)
check("知らない厳しさは断る", "厳しさは" in refused, refused)
check("使える厳しさを教える", "育成" in refused and "厳格" in refused)


print("\n== 採点モード ==")
head, body = build_prompt(chunk, (), mode=SCORE, severity="厳格",
                          directory=str(HERE / "無"))
check("七観点すべてが入る",
      all(axis in head for axis in rubric.AXES),
      str([a for a in rubric.AXES if a not in head]))
check("五段階の水準が入る", head.count("  5: ") == 7,
      str(head.count("  5: ")))
check("出力形式を指定する", "X/5" in head)
check("観点のくじ引きを使わない", "今回の観点は次の" not in head)
check("見えていない部分を補わせない", "想像で補わない" in head)
check("本文には段落番号が付く", "[0]" in body)

# 採点は塊ごとに渡ることがある。全体を見たと誤解させない
check("範囲が限られていることを伝える", "渡された範囲だけ" in head)

# 作者の弁明は査読と採点には渡さない
for mode, severity in ((REVIEW, "標準"), (SCORE, "標準")):
    _, said = build_prompt(chunk, lenses, mode=mode, severity=severity,
                           note="ここは笑わせるつもりで書きました",
                           directory=str(HERE / "無"))
    check(f"{mode}では作者の補足を渡さない", "笑わせるつもり" not in said)
_, said = build_prompt(chunk, review.lenses_for(IDEA)[:3], mode=IDEA,
                       note="ここは笑わせるつもりで書きました",
                       directory=str(HERE / "無"))
check("発想では作者の補足を渡す", "笑わせるつもり" in said)


print("\n== 観点ファイルの読み書き ==")
rows = wording.parse_lenses("""
# これは読み飛ばす行

削除 | 構造 | 発想,査読 | 何が壊れるかを述べよ。
分岐 | 不在 | 発想 | 選ばなかった道を挙げよ。
""")
check("二行読める", len(rows) == 2, str(len(rows)))
check("鍵が取れる", rows[0][0] == "削除")
check("群が取れる", rows[0][1] == "構造")
check("モードが組になる", rows[0][2] == ("発想", "査読"))
check("片方だけのモードも読める", rows[1][2] == ("発想",))
check("問いが取れる", rows[0][3] == "何が壊れるかを述べよ。")


def broken(text):
    try:
        wording.parse_lenses(text)
    except ValueError as problem:
        return str(problem)
    return ""


check("欄が足りなければ断る", "4つに区切る" in broken("削除 | 構造 | 発想"))
check("何行目かを言う", "2 行目" in broken("\n削除 | 構造 | 発想"))
check("空の欄を断る", "空の欄" in broken("削除 |  | 発想 | 問い。"))
check("モードが空なら断る", "モードが空" in broken("削除 | 構造 |  | 問い。"))
check("一つも無ければ断る", "1つも無い" in broken("# 註だけ\n\n"))

# 書き出したものを読み直して同じに戻らないと、
# 「prompts で書き出して編集する」という道筋が成り立たない
text = wording.format_lenses(review.BUILTIN_LENSES)
again = wording.parse_lenses(text)
check("書き出して読み直すと元に戻る",
      again == [(l.key, l.group, tuple(l.modes), l.question)
                for l in review.BUILTIN_LENSES],
      f"{len(again)}件 / {len(review.BUILTIN_LENSES)}件")
check("書き出しに書式の説明が入る", "縦棒" in text or "|" in text)


print("\n== prompts/ が効く ==")
with tempfile.TemporaryDirectory() as folder:
    written, skipped = review.export_wording(folder)
    names = {path.stem for path in written}
    check("八つ書き出す", len(written) == 8, str(sorted(names)))
    check("発想を書き出す", "発想" in names)
    check("厳しさごとに書き出す",
          {"査読-育成", "査読-標準", "査読-厳格",
           "採点-育成", "採点-標準", "採点-厳格"} <= names,
          str(sorted(names)))
    check("観点を書き出す", "観点" in names)
    check("初回は何も飛ばさない", not skipped)

    # 二度目は触らない。編集したものを消さないため
    written2, skipped2 = review.export_wording(folder)
    check("二度目は上書きしない", not written2 and len(skipped2) == 8)
    written3, _ = review.export_wording(folder, force=True)
    check("--force なら上書きする", len(written3) == 8)

    # 書き換えたものが実際に使われるか
    target = Path(folder) / "査読-標準.txt"
    target.write_text(target.read_text(encoding="utf-8")
                      + "- 手で足した規則。\n", encoding="utf-8")
    head, _ = build_prompt(chunk, lenses, mode=REVIEW, directory=folder)
    check("編集した規則が使われる", "手で足した規則" in head)

    # 触っていない厳しさは既定のまま
    other, _ = build_prompt(chunk, lenses, mode=REVIEW, severity="厳格",
                            directory=folder)
    check("触っていない厳しさは変わらない", "手で足した規則" not in other)

    # 観点ファイルの差し替え
    saved = list(review.LENSES)
    try:
        (Path(folder) / "観点.txt").write_text(
            "手作り | 実験 | 発想,査読 | 手で足した観点。\n",
            encoding="utf-8")
        check("観点を読み込む", review.load_wording(folder))
        check("読み込んだ観点だけになる",
              [l.key for l in review.LENSES] == ["手作り"],
              str([l.key for l in review.LENSES]))
        check("鍵の索引も入れ替わる", "手作り" in review.LENS_BY_KEY
              and "削除" not in review.LENS_BY_KEY)
        check("群も入れ替わる", review.GROUPS == ["実験"],
              str(review.GROUPS))
    finally:
        review.set_lenses(saved)
    check("戻せる", [l.key for l in review.LENSES]
          == [l.key for l in saved])

with tempfile.TemporaryDirectory() as empty:
    check("観点ファイルが無ければ何もしない",
          review.load_wording(empty) is False)
    head, _ = build_prompt(chunk, lenses, mode=REVIEW, directory=empty)
    check("無ければ既定の文面を使う", "忖度" in head)


print("\n== 手元の prompts/ が壊れていない ==")
# 使う人が編集したものを、試験のたびに読み直す。
# 書式を壊したまま気づかずに使い続けるのを防ぐ
if wording.HOME.exists():
    try:
        rows = wording.load_lenses()
        check("観点.txt が読める", rows is None or len(rows) > 0)
    except ValueError as problem:
        check("観点.txt が読める", False, str(problem))
    for name in ["発想"] + [f"{mode}-{severity}"
                            for mode in (REVIEW, SCORE)
                            for severity in rubric.SEVERITIES]:
        text = wording.read(name)
        check(f"{name}.txt が空でない", text is None or len(text) > 20,
              f"{name}.txt")
else:
    check("prompts/ はまだ無い (main.py prompts で作れる)", True)


print(f"\n{passed}件通過 / {failed}件失敗")
sys.exit(1 if failed else 0)

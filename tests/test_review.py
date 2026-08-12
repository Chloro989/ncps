"""
観点とプロンプト組み立ての試験。

    python tests/test_review.py

一番大事なのは段落番号の検査で、これが効かないと
「実在しない箇所への指摘」を素通りさせてしまう。
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion.manuscript import Manuscript
from centurion.review import (GROUPS, IDEA_RULES, LENSES, NEED, REVIEW_RULES,
                              build_prompt, check_citations, choose_lenses,
                              citations, describe, needs, number_paragraphs,
                              resolve, suggest_lenses, survey)

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
aozora = Manuscript.load(FIXTURES / "aozora_akuma.txt")

print("== 観点 ==")
check("鍵が重複していない", len({l.key for l in LENSES}) == len(LENSES))
check("群が複数ある", len(GROUPS) >= 5, str(GROUPS))
check("すべて操作として書かれている",
      all(l.question.rstrip().endswith(("せよ。", "述べよ。", "挙げよ。",
                                        "示せ。", "問え。", "書け。",
                                        "判じよ。", "分けよ。", "探せ。"))
          for l in LENSES),
      str([l.key for l in LENSES
           if not l.question.rstrip().endswith(
               ("せよ。", "述べよ。", "挙げよ。", "示せ。", "問え。",
                "書け。", "判じよ。", "分けよ。", "探せ。"))]))
check("願望だけの言い回しを含まない",
      not any(word in l.question
              for l in LENSES
              for word in ("独創的に", "面白く", "深みを", "丁寧に")))

print("\n== 観点の選び方 ==")
rng = random.Random(1)
picked = choose_lenses(rng, count=3)
check("指定した数だけ選ぶ", len(picked) == 3)
check("同じ観点を二度選ばない", len({l.key for l in picked}) == 3)
check("群を散らす", len({l.group for l in picked}) == 3,
      str([l.group for l in picked]))
check("同じ種なら同じ観点",
      [l.key for l in choose_lenses(random.Random(9))]
      == [l.key for l in choose_lenses(random.Random(9))])
check("読み直すたびに入れ替わる",
      len({tuple(sorted(l.key for l in choose_lenses(rng)))
           for _ in range(30)}) > 10)
check("群を絞れる",
      all(l.group == "不在" for l in choose_lenses(rng, 2, groups=["不在"])))
check("求める数が多すぎても落ちない",
      len(choose_lenses(rng, count=99)) == len(LENSES))
check("群を絞ったうえで数が多くても落ちない",
      len(choose_lenses(rng, count=99, groups=["熱量"])) == 2)

print("\n== 原稿を測る ==")
named = Manuscript("\n\n".join([
    "　川口さんは大山町の坂を下りていった。冷たい風が吹いていた。",
    "　佐藤くんが振り返った。遠くで鐘が鳴った。匂いがした。",
    "　光が差した。硬い石畳に足を取られた。彼は転んだ。",
]))
nameless = Manuscript("\n\n".join([
    "　彼は坂を下りていった。",
    "　男が振り返った。",
    "　誰かが何かを言った。",
]))
check("名前のある原稿は高く出る",
      survey(named.paragraphs)["名前"] > survey(nameless.paragraphs)["名前"],
      f"{survey(named.paragraphs)['名前']:.2f} 対 "
      f"{survey(nameless.paragraphs)['名前']:.2f}")
check("感覚が多い原稿は高く出る",
      survey(named.paragraphs)["感覚"] > survey(nameless.paragraphs)["感覚"])
check("すべて0〜1に収まる",
      all(0 <= value <= 1 for value in survey(named.paragraphs).values()),
      str(survey(named.paragraphs)))
check("段落が無くても落ちない",
      all(0 <= v <= 1 for v in survey([]).values()))

talky = Manuscript("\n\n".join(["「行こう」", "「うん」", "「まだ早い」",
                                "　彼は黙っていた。"]))
check("会話文の割合を測る", survey(talky.paragraphs)["会話"] == 0.75,
      str(survey(talky.paragraphs)["会話"]))

events = Manuscript("\n\n".join([
    "　雨が降った。傘をさした。歩き出した。",
    "　風が吹いた。木が鳴った。空が暗くなった。",
]))
check("出来事を語る割合を測る",
      survey(events.paragraphs)["出来事"] == 1.0)

rutty = Manuscript("　宇宙の神秘は永遠の深淵に似ている。静寂が彼方に。")
check("常套語の濃さを測る", survey(rutty.paragraphs)["轍"] > 0.5,
      str(survey(rutty.paragraphs)["轍"]))
check("常套語が無ければ0", survey(events.paragraphs)["轍"] == 0.0)

print("\n== 必要度 ==")
score, measured = needs(nameless.paragraphs)
check("すべての観点に値が付く", set(score) == {l.key for l in LENSES})
check("名前の無い原稿では固有が高い", score["固有"] > 0.7,
      f"{score['固有']:.2f}")
check("名前のある原稿では固有が下がる",
      needs(named.paragraphs)[0]["固有"] < score["固有"])
check("感覚の揃った原稿では感覚が下がる",
      needs(named.paragraphs)[0]["感覚"] < needs(nameless.paragraphs)[0]["感覚"])
check("常套語の多い原稿では既視が高い",
      needs(rutty.paragraphs)[0]["既視"] > needs(events.paragraphs)[0]["既視"])
check("値の無い観点は中央に置く",
      needs(named.paragraphs)[0]["重心"] == 0.5)
check("実測を一行で見せられる", "名前" in describe(measured))

print("\n== 実測から観点を選ぶ ==")
picked, measured = suggest_lenses(nameless.paragraphs, count=3,
                                  rng=random.Random(1))
check("指定した数だけ選ぶ", len(picked) == 3)
check("群を散らす", len({l.group for l in picked}) == 3,
      str([l.group for l in picked]))
check("必要度の高いものが入る",
      "固有" in {l.key for l in picked} or "分岐" in {l.key for l in picked},
      str([l.key for l in picked]))
check("実測も返す", "名前" in measured)
check("同じ種なら同じ観点",
      [l.key for l in suggest_lenses(nameless.paragraphs,
                                     rng=random.Random(2))[0]]
      == [l.key for l in suggest_lenses(nameless.paragraphs,
                                        rng=random.Random(2))[0]])
check("揺らぎがあるので毎回は同じでない",
      len({tuple(l.key for l in suggest_lenses(nameless.paragraphs,
                                               rng=random.Random(s))[0])
           for s in range(20)}) > 1)
check("原稿が違えば選ばれる観点も違う",
      {l.key for l in suggest_lenses(named.paragraphs, count=4,
                                     rng=random.Random(3), jitter=0)[0]}
      != {l.key for l in suggest_lenses(nameless.paragraphs, count=4,
                                        rng=random.Random(3), jitter=0)[0]})
check("求める数が多すぎても落ちない",
      len(suggest_lenses(named.paragraphs, count=99)[0]) == len(LENSES))
check("段落が無くても落ちない", len(suggest_lenses([], count=2)[0]) == 2)

print("\n== 本文の番号 ==")
numbered = number_paragraphs(novel.paragraphs)
check("全段落に番号が付く", numbered.count("[") == len(novel.paragraphs))
check("番号は段落の通し番号と一致する",
      numbered.startswith(f"[0] {novel.paragraphs[0].text[:6]}"))

chunks = aozora.chunks(size=500, overlap=1)
marked = number_paragraphs(chunks[1].paragraphs,
                           {p.index for p in chunks[1].paragraphs[1:]})
check("持ち越した段落に印が付く", marked.splitlines()[0].startswith("[")
      and "＞" in marked.splitlines()[0])
check("担当する段落には印が付かない", "＞" not in marked.splitlines()[1])

print("\n== プロンプトの組み立て ==")
lenses = choose_lenses(random.Random(3), 2)
head, body = build_prompt(chunks[1], lenses, mode="発想",
                          title=aozora.title, author=aozora.author,
                          note="語り手の距離感を試している",
                          place="2つ目の範囲")
check("観点が本文に入る", all(l.question[:12] in head for l in lenses))
check("観点の数を明示する", f"次の{len(lenses)}つだけ" in head)
check("発想モードの規則が入る", IDEA_RULES[0] in head)
check("査読モードの規則は入らない", REVIEW_RULES[3] not in head)
check("題と著者が入る", "「悪魔」" in body and "芥川龍之介" in body)
check("作者の補足が入る", "語り手の距離感" in body)
check("重なりの説明が入る", "重なり" in body)
check("段落番号つきの本文が入る",
      f"[{chunks[1].paragraphs[0].index}]" in body)

head_review, body_review = build_prompt(chunks[0], lenses, mode="査読",
                                        title=aozora.title)
check("査読モードの規則が入る", REVIEW_RULES[3] in head_review)
check("査読モードでは発想の規則が入らない", IDEA_RULES[0] not in head_review)
check("査読モードでは作者の補足を渡さない",
      "補足" not in build_prompt(chunks[0], lenses, mode="査読",
                                note="狙いはこうです")[1])

print("\n== 段落番号の検査 ==")
answer = ("[3] の描写は効いている。一方 [9999] は冗長で、"
          "[0] との対応も薄い。")
real, missing, outside = check_citations(answer, aozora)
check("実在する番号を拾う", real == [3, 0], str(real))
check("存在しない番号を見つける", missing == [9999], str(missing))
check("範囲外という概念が無ければ空", outside == [])

allowed = {p.index for p in chunks[1].paragraphs[chunks[1].carried:]}
real, missing, outside = check_citations(answer, aozora, allowed=allowed)
check("担当範囲の外を指した指摘を見つける", len(outside) >= 1,
      f"{real} / {outside}")

check("番号が無ければ空", citations("よい文章である。") == [])
check("番号を並べて拾える", citations("[1][2] [30]") == [1, 2, 30])

print("\n== 番号を本文に戻す ==")
restored = resolve("[3] が要点である。", aozora)
check("番号のあとに本文が付く",
      aozora.paragraphs[3].text[:10] in restored, restored[:60])
check("存在しない番号は印を付けて残す", "[9999?]" in resolve("[9999]", aozora))

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

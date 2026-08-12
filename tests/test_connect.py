"""
遠い二つを繋ぐ仕組みの試験。

    python tests/test_connect.py

正解の分かっている作品で、作者が架けた橋を拾えるかを確かめたところ、
無作為に対を選ぶ方式では5回試して0件だった(候補1165対から6個)。
そこから、稀な語の反復を手がかりにする方式に変えている。
この試験は、その手がかりが効いていることを繋ぎ止めるためのもの。
"""

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion.connect import (DREAM_WORK, GENERIC, MIN_CHARS, MOTIF_TIMES,
                               build_chain_prompt, build_connection_prompt,
                               content_words, distant_pairs, once_only,
                               overlap, recurrences)
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


def build(*paragraphs):
    return Manuscript("\n\n".join(paragraphs))


# 仕掛けを埋めた原稿。「万華鏡」を端と端に置き、間を埋め草で伸ばす
FILLER = [
    "　彼は坂を下りて改札を抜けた。売店の灯りがまだついていて、"
    "並んだ雑誌の表紙だけが妙に明るく見えた。",
    "　雨は上がっていたが、路面には水たまりが残っていた。"
    "靴の先が濡れるのを避けながら歩いた。",
    "　駅前の食堂で、遅い夕飯を済ませることにした。"
    "券売機の一番左の釦を押した。",
    "　店主は無言で丼を置き、また新聞に戻っていった。"
    "壁の時計だけが音を立てていた。",
    "　外に出ると、風が思ったより冷たくなっていた。"
    "襟を立てて、歩幅を少し広げた。",
    "　バスの時刻表を眺めて、歩いて帰ることに決めた。"
    "次の便まで二十分あった。",
    "　橋の上で立ち止まり、水路の暗がりを覗きこんだ。"
    "水面には何も映っていなかった。",
    "　遠くで踏切の音がして、それから静かになった。"
    "自転車が一台、脇を追い越していった。",
]
planted = build(
    "　祖母の簞笥の奥に万華鏡が仕舞われていた。"
    "覗くと硝子の欠片が鳴って、模様がゆっくりと崩れていった。",
    *FILLER,
    "　病室の窓辺に万華鏡が置いてあった。"
    "誰が持ってきたのか、看護師に訊いても分からないままだった。")

print("== 語の取り出し ==")
check("漢字2文字以上を拾う", "万華鏡" in content_words("祖母の万華鏡である。"))
check("カタカナ2文字以上を拾う", "ガラス" in content_words("ガラスの音がした。"))
check("一文字は拾わない", "鳥" not in content_words("鳥が鳴いた。"))
check("重なりは同じ文で1.0", overlap("硝子が鳴った。", "硝子が鳴った。") == 1.0)
check("共通語が無ければ0", overlap("硝子が鳴った。", "電車が走った。") == 0.0)
check("空文なら0", overlap("", "硝子") == 0.0)

print("\n== 反復 ==")
# 実距離の下限はこの節では切っておく。合成原稿は700文字ほどしかなく、
# 下限を効かせると何も残らない。下限そのものは次の節で確かめる
found = recurrences(planted, min_chars=0)
words = [item.word for item in found]
check("端と端に置いた語を見つける", "万華鏡" in words, str(words))
check("埋め草の語は上位に来ない", words[0] == "万華鏡", str(words[:3]))
motif = next(item for item in found if item.word == "万華鏡")
check("現れる段落を全部持つ",
      [p.index for p in motif.paragraphs] == [0, 9],
      str([p.index for p in motif.paragraphs]))
check("隔たりを測る", motif.gap > 0.8, f"{motif.gap:.2f}")
check("回数が少なければ反復と呼ぶ", motif.kind == "反復")
check("対に変換できる",
      motif.pair().left.index == 0 and motif.pair().right.index == 9)
check("表示に段落番号が入る", "[0]" in str(motif) and "[9]" in str(motif))

near = build("万華鏡を覗いた。", "万華鏡を仕舞った。", *FILLER)
check("近すぎる反復は拾わない",
      "万華鏡" not in [i.word for i in recurrences(near, min_chars=0)])

generic = build("　彼は自分のことを話しはじめた。長い話だった。", *FILLER,
                "　最後に自分の名前だけを言い残していった。")
check("一般語は除く",
      "自分" not in [i.word for i in recurrences(generic, min_chars=0)])
check("一般語の一覧が空でない", len(GENERIC) > 20)

many = build(*[f"　レースの話を{i}度目にした日のことである。" for i in range(6)],
             *FILLER)
found_many = [i for i in recurrences(many, min_chars=0)
              if i.word == "レース"]
check("何度も出る語は主題と呼ぶ",
      found_many and found_many[0].kind == "主題",
      str([(i.word, i.times, i.kind) for i in found_many]))
check("反復と主題の境目が定まっている", MOTIF_TIMES >= 2)

print("\n== 実距離の下限 ==")
# 割合だけで隔たりを測ると短い作品で破綻する。太宰治「I can speak」
# (1949文字・15段落)では「酔漢」が[8]と[12]に出て隔たり27%と表示されたが、
# 実際には4段落しか離れておらず、しかも同じ場面の同じ人物だった
short = build("　万華鏡を覗いた。硝子の欠片がゆっくりと崩れていった。",
              *FILLER,
              "　もう一度その万華鏡を覗いた。今度は何も鳴らなかった。")
check("下限を切れば拾う",
      "万華鏡" in [i.word for i in recurrences(short, min_chars=0)])
check("下限を効かせれば落ちる",
      "万華鏡" not in [i.word for i in recurrences(short, min_chars=1500)],
      f"原稿は{len(short.text)}文字しかない")
check("既定の下限が働いている",
      "万華鏡" not in [i.word for i in recurrences(short)])
check("下限の既定値が定まっている", MIN_CHARS >= 1000, str(MIN_CHARS))
check("短い原稿では対も出ない",
      distant_pairs(short, count=5, rng=random.Random(0)) == [])
check("反復は実距離を持つ",
      recurrences(short, min_chars=0)[0].chars < 1500)

print("\n== 遠い対 ==")
pairs = distant_pairs(planted, count=3, rng=random.Random(0), min_chars=0)
check("求めた数だけ返す", len(pairs) == 3, str(len(pairs)))
check("隔たりの条件を満たす", all(p.gap >= 0.2 for p in pairs))
check("重なりの条件を満たす", all(p.overlap <= 0.12 for p in pairs))
check("左が先、右が後", all(p.left.index < p.right.index for p in pairs))
check("同じ段落を使い回さない",
      len({p.left.index for p in pairs} | {p.right.index for p in pairs}) == 6)
check("同じ種なら同じ対",
      [(p.left.index, p.right.index)
       for p in distant_pairs(planted, 3, rng=random.Random(4))]
      == [(p.left.index, p.right.index)
          for p in distant_pairs(planted, 3, rng=random.Random(4))])
check("短い段落は使わない",
      all(len(p.left.text) >= 30 and len(p.right.text) >= 30
          for p in pairs))
check("段落が足りなければ空", distant_pairs(build("短い。")) == [])
check("空の原稿でも落ちない", distant_pairs(Manuscript("")) == [])
check("表示に隔たりが入る", "隔たり" in str(pairs[0]))

print("\n== 一度きりの語 ==")
once = once_only(planted)
check("一度しか出ない語を拾う", any(word == "簞笥" for word, _ in once),
      str([w for w, _ in once][:8]))
check("二度出る語は入らない", all(word != "万華鏡" for word, _ in once))
check("段落番号が付く", all(isinstance(index, int) for _, index in once))

print("\n== プロンプト ==")
head, body = build_connection_prompt(planted, motif.pair(),
                                     note="祖母の話を軸にしています",
                                     extra=DREAM_WORK[:1])
check("両方の段落が入る",
      planted.paragraphs[0].text[:12] in body
      and planted.paragraphs[9].text[:12] in body)
check("段落番号が入る", "[0]" in body and "[9]" in body)
check("隔たりを伝える", "離れており" in head)
check("代償を書かせる", "失うか" in head)
check("褒めさせない", "褒めない" in head)
check("外から設定を持ち込ませない", "持ち込まない" in head)
check("繋がらない判断を許す", "繋がらないと判断" in head)
check("夢の作業を足せる", DREAM_WORK[0][0] in head)
check("作者の補足が入る", "祖母の話を軸" in body)

head, body = build_chain_prompt(planted, planted.paragraphs[0], steps=5)
check("歩数を伝える", "5歩" in head)
check("理由を書かせる", "なぜそこへ移った" in head)
check("作品へ戻させる", "この作品へ戻す" in head)
check("戻せない判断を許す", "戻せない" in head)
check("元の段落が入る", planted.paragraphs[0].text[:12] in body)

print("\n== 夢の作業 ==")
check("四つある", len(DREAM_WORK) == 4)
check("鍵が重複しない", len({key for key, _ in DREAM_WORK}) == 4)
check("すべて操作として書かれている",
      all(question.rstrip().endswith(("述べよ。", "書け。", "置き換えよ。",
                                      "通してみよ。"))
          for _, question in DREAM_WORK),
      str([k for k, q in DREAM_WORK
           if not q.rstrip().endswith(("述べよ。", "書け。", "置き換えよ。",
                                       "通してみよ。"))]))

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

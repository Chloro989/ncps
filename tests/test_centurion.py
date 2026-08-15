"""
センチュリオンの、モデルを読まずに確かめられる部分の試験。

    python tests/test_centurion.py

GPUの要る部分(生成そのもの)は Colab で確かめる。
ここで見るのは、抑圧が狙った場所だけで働くか、文末で切れるか、
プロンプトが組み上がるか、コマンドラインの引数が通るか。
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
except ImportError:
    # 原稿を読ませるだけの人は torch を入れない。
    # この試験は小説を書く側のものなので、無いなら飛ばす
    print("torch が無いので、小説を書く側の試験は飛ばす")
    print("0件通過 / 0件失敗")
    raise SystemExit(0)

from centurion.generate import (BranchDiverter, ENTROPY_GATE,
                                SUPPRESS_STRENGTH, SUPPRESS_TOP_K, trim)
from centurion.prompts import (BAN_COUNT, PREFILL, RUT_WORDS, STANCE_CLAUSES,
                               STANCE_COUNT, build_fluid)
from centurion.__main__ import build_parser

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


class FakeTokenizer:
    def decode(self, ids):
        return f"<{ids[0]}>"


def flat_scores(size, peak=None, height=20.0):
    """一様なスコア。peak を指定すると、そこだけ尖ってエントロピーが下がる"""
    scores = torch.zeros(1, size)
    if peak is not None:
        scores[0, peak] = height
    return scores


print("== 文末で切る ==")
check("途中で切れた文を落とす",
      trim("そうですね、雨が降った。傘を持って") == "そうですね、雨が降った。")
check("文末で終わっていればそのまま",
      trim("そうですね、雨が降った。") == "そうですね、雨が降った。")
check("鉤括弧の閉じも文末として扱う",
      trim("彼は「行こう」と言った「またね」") == "彼は「行こう」と言った「またね」")
check("文末が一つも無ければ丸ごと残す",
      trim("そうですね") == "そうですね")

print("\n== 分岐点での抑圧 ==")
diverter = BranchDiverter(FakeTokenizer())

# 尖った分布 = 迷っていない = 抑圧しない
sharp = flat_scores(1000, peak=7)
before = sharp.clone()
diverter.reset()
result = diverter(None, sharp)
check("迷っていない箇所では何もしない", torch.equal(result, before),
      f"エントロピーが{ENTROPY_GATE}未満のはず")
check("抑圧の記録も残らない", diverter.diverted == [])

# 平らな分布 = 分岐点 = 上位を押し下げる
flat = flat_scores(1000)
flat[0, 3] = 1.0        # わずかに上位2つを作る
flat[0, 5] = 0.9
diverter.reset()
result = diverter(None, flat.clone())
check("分岐点では上位を押し下げる",
      abs(float(result[0, 3]) - (1.0 - SUPPRESS_STRENGTH)) < 1e-5,
      f"{float(result[0, 3])}")
check(f"押し下げるのは上位{SUPPRESS_TOP_K}個だけ",
      abs(float(result[0, 5]) - (0.9 - SUPPRESS_STRENGTH)) < 1e-5
      and abs(float(result[0, 9])) < 1e-5)
check("抑圧した候補を記録する", len(diverter.diverted) == 1,
      str(diverter.diverted))
check("reset で記録が消える",
      (diverter.reset(), diverter.diverted == [])[1])

# 押し下げ幅が量子化されていないこと(round を使うと選択圧が伝わらない)
fine = flat_scores(1000)
fine[0, 2] = 0.123456
diverter.reset()
result = diverter(None, fine.clone())
check("押し下げ幅を丸めない",
      abs(float(result[0, 2]) - (0.123456 - SUPPRESS_STRENGTH)) < 1e-6)

print("\n== 流動プロンプト ==")
seen_stance, seen_banned = set(), set()
for _ in range(200):
    prompt, stance, banned = build_fluid()
    seen_stance.update(stance)
    seen_banned.update(banned)
    if len(stance) != STANCE_COUNT or len(banned) != BAN_COUNT:
        break
check(f"姿勢を{STANCE_COUNT}つ選ぶ", len(stance) == STANCE_COUNT)
check(f"禁止語を{BAN_COUNT}つ選ぶ", len(banned) == BAN_COUNT)
check("同じ姿勢を二度選ばない", len(set(stance)) == STANCE_COUNT)
check("すべての姿勢が出うる", seen_stance == set(STANCE_CLAUSES),
      f"{len(seen_stance)}/{len(STANCE_CLAUSES)}")
check("すべての禁止語が出うる", seen_banned == set(RUT_WORDS),
      f"{len(seen_banned)}/{len(RUT_WORDS)}")
check("選んだ姿勢が本文に入る", all(s in prompt for s in stance))
check("選んだ禁止語が本文に入る", all(b in prompt for b in banned))
check("人格の書き出しは常に同じ",
      prompt.startswith("あなたはセンチュリオン。"))
check("長さが現行プロンプトに近い", 170 <= len(prompt) <= 230, str(len(prompt)))

print("\n== 乱数種 ==")
# 姿勢の抽選は Python の乱数、本文の抽選は torch の乱数。
# torch だけ固定しても同じ文章にはならない
first = [build_fluid(random.Random(11))[1] for _ in range(5)]
second = [build_fluid(random.Random(11))[1] for _ in range(5)]
check("同じ種なら同じ姿勢が出る", first == second)
check("違う種なら違う姿勢が出る",
      first != [build_fluid(random.Random(12))[1] for _ in range(5)])

stream = random.Random(11)
check("同じ種でも回すたびに姿勢は変わる",
      build_fluid(stream)[1] != build_fluid(stream)[1]
      or build_fluid(stream)[1] != build_fluid(stream)[1])

print("\n== コマンドライン ==")
parser = build_parser()
args = parser.parse_args(["青色にまつわる話を聞かせて"])
check("お題を受け取る", args.topics == ["青色にまつわる話を聞かせて"])
check("抑圧は既定で入っている", not args.no_suppress)
check("流動プロンプトは既定で入っている", not args.fixed_prompt)
check("書き出しの既定は相槌", args.prefill == PREFILL)
check("文末で切るのが既定", not args.raw)
args = parser.parse_args(["朝の匂い", "沈黙", "--turns", "--seed", "7"])
check("会話にできる", args.turns and args.seed == 7)
args = parser.parse_args(["--chat", "--prefill", ""])
check("書き出しを空にできる", args.chat and args.prefill == "")

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

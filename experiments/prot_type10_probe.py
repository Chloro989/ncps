"""
Phase 10 の受け入れ試験。LLM を動かさずに回路だけを確かめる。

Phase 0 の教訓 — 入力に反応しない回路でも、動いているように見える。
生成に混ぜる前に、姿勢の選び方が次の5つを満たすか確かめる。

  1. 漂い     隣のターンで共有する句が、ランダムより高く固定より低い
  2. 多様性   会話をまたいで十分な種類の組み合わせが出る
  3. 反応     入力を変えたら選ぶ姿勢も変わる
  4. 再現性   同じ入力なら同じ選択になる
  5. 記憶     階段応答の緩和が2ターン以上ある

5 が要る理由: LTC を歩幅8で使うと漂いは0.91で CfC+LSTM と同じ値になるが、
階段応答の緩和は1.62しかなく記憶がほぼ無い。
漂いだけでは記憶のある回路と無い回路を区別できなかった。

回路の重みの種によって振る舞いが大きく違う(漂い0.72〜1.41、凍結0〜5本)。
後から良い種を選ぶのは結果を見てからの選択になるので、
ここに条件を先に書き、種0から順に探して最初に通ったものを使う。
prot_type10_duel.py は find_seed() を呼ぶだけで、種を指定しない。
"""
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from centurion_prompts import STANCE_CLAUSES, STANCE_COUNT
from centurion_turn import (
    SHARED_TURNS, TURNS, StanceCircuit, pick_by_circuit, response_features,
)

COUNT = len(STANCE_CLAUSES)
COMBINATIONS = COUNT * (COUNT - 1) // 2
CONVERSATION_COUNT = 16    # 本番と同じ会話数
SEEDS = 6                  # 構造の確認に使う種の数
SEARCH_LIMIT = 24          # 種を探す上限

# 合格条件。数字を見る前に決めてある。
# 漂いの下限はランダム選択の実測値(約0.45)に余裕を持たせたもの。
# 凍結2本は、16会話のうち姿勢が動かない会話が2本までなら自然という判断
DRIFT_MIN, DRIFT_MAX = 0.70, 1.60
VARIETY_MIN = 8
FROZEN_MAX = 2
RELAX_MIN = 2
# 最頻句が選択の何割に入っているか。第1回はこれを見ておらず、
# 「断定を恐れず」が92%に入った状態を組み合わせ8種として合格させた。
# 種類の数は、同じ句を軸にした8通りとバラバラな8通りを区別できない。
# ランダム選択なら1句あたり2/9=22%なので、5割を上限にする
SHARE_MAX = 0.50
STEP_HOLD = 3              # 段の前に一定入力を保つターン数
STEP_AFTER = 8             # 段のあと一定入力で追うターン数

# 第1回の実測応答。想像で作った文の池は本番と分布が違い
# (轍語率 -0.75σ / 豊かさ +1.09σ)、別の作動点で回路を測っていた。
# 実物を読んで、そこから引く
RESPONSE_FILE = (Path(__file__).resolve().parent.parent / "results"
                 / "centurion_turn_responses.txt")


def load_responses():
    """実測応答を「ターン番号 → 本文の並び」で読む。
    ターンごとに引くのは、長さや轍語率がターンによって偏るため"""
    by_turn = {}
    for line in RESPONSE_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "\t" not in line:
            continue
        number, body = line.split("\t", 1)
        by_turn.setdefault(int(number), []).append(body)
    return by_turn


RESPONSES = load_responses()


def wide_features(rng, turn):
    """特徴の取りうる範囲全体をなめる入力。構造の確認に使う。
    毎ターン独立なので、それでも隣のターンが似るなら
    似せているのは入力ではなく回路の記憶しかない"""
    return torch.tensor([turn / max(TURNS - 1, 1), rng.uniform(0.0, 2.5),
                         rng.uniform(0.5, 1.5), rng.uniform(0.5, 0.9)],
                        dtype=torch.float32)


def response_like_features(rng, turn):
    """第1回の実測応答から引いた入力。種を選ぶのに使う"""
    return response_features(rng.choice(RESPONSES[turn]), turn, TURNS)


def constant_features(level):
    """特徴を平均から level ぶんずらした一定入力"""
    return torch.tensor([0.5, 1.0 + level, 1.0 + level * 0.4,
                         0.7 - level * 0.1], dtype=torch.float32)


def relaxation(circuit, size=1.0):
    """階段応答の緩和ターン数。
    入力を切り替えたあとは一定に保つので、記憶がなければ次のターンで
    落ち着いて1になる。それより長く姿勢が動くなら記憶が働いている"""
    circuit.reset()
    for _ in range(STEP_HOLD):
        pick_by_circuit(circuit, constant_features(-size / 2))
    after = [pick_by_circuit(circuit, constant_features(size / 2))[0]
             for _ in range(STEP_AFTER)]
    settled = STEP_AFTER
    while settled > 1 and after[settled - 2] == after[STEP_AFTER - 1]:
        settled -= 1
    return settled


def walk(circuit, seed, sampler, shared=0):
    """会話ぶんの姿勢を選ぶ。shared に 1 を渡すと本番と同じ形になる —
    1ターン目は両条件で共通なので回路を通らず、選択は2ターン目から。
    本番では t ターン目の入力が t-1 ターン目の応答なので、
    回路が呼ばれる回数は TURNS - shared 回になる"""
    rng = random.Random(seed)
    picks = []
    for _ in range(CONVERSATION_COUNT):
        circuit.reset()
        chosen = []
        for turn in range(shared, TURNS):
            stance, _ = pick_by_circuit(circuit, sampler(rng, turn))
            chosen.append(tuple(sorted(stance)))
        picks.append(chosen)
    return picks


def measure(picks):
    """漂い(隣のターンで共有する句)、組み合わせの種類、凍結した会話の数、
    最頻句が選択の何割に入っているか"""
    steps = len(picks[0])
    drift = float(np.mean([len(set(c[i]) & set(c[i - 1]))
                           for c in picks for i in range(1, steps)]))
    variety = len({p for c in picks for p in c})
    frozen = sum(1 for c in picks if len(set(c)) == 1)
    counts = Counter(clause for c in picks for pair in c for clause in pair)
    total = sum(len(c) for c in picks)
    share = counts.most_common(1)[0][1] / total if total else 1.0
    return drift, variety, frozen, share


def build(seed):
    torch.manual_seed(seed)
    circuit = StanceCircuit(COUNT)
    circuit.settle()
    return circuit


def check(seed):
    """本番と同じ形(会話16本、1ターン目共通、回路の呼び出し3回)で測る"""
    circuit = build(seed)
    drift, variety, frozen, share = measure(
        walk(circuit, 4242, response_like_features, shared=SHARED_TURNS))
    return drift, variety, frozen, share, relaxation(circuit)


def passes(drift, variety, frozen, share, relax):
    return (DRIFT_MIN <= drift <= DRIFT_MAX and variety >= VARIETY_MIN
            and frozen <= FROZEN_MAX and share <= SHARE_MAX
            and relax >= RELAX_MIN)


def find_seed(limit=SEARCH_LIMIT):
    """条件を通す最初の種。0から順に探すので、結果を見てから選ぶ余地がない"""
    for seed in range(limit):
        if passes(*check(seed)):
            return seed
    raise RuntimeError(f"{limit}種のどれも条件を通らなかった")


def random_drift(trials=20000):
    rng = random.Random(0)
    return float(np.mean([
        len(set(rng.sample(STANCE_CLAUSES, STANCE_COUNT))
            & set(rng.sample(STANCE_CLAUSES, STANCE_COUNT)))
        for _ in range(trials)]))


def structure():
    """入力に反応するか、同じ入力で再現するかを、広い分布で確かめる"""
    baseline = random_drift()
    print(f"姿勢の句 {COUNT}個から{STANCE_COUNT}個 → 組み合わせ{COMBINATIONS}通り")
    print(f"ランダム選択の共有句 {baseline:.2f} / 固定なら {STANCE_COUNT}.00\n")
    print("== 広い分布で構造を確かめる ==")
    print(f"{'種':>4} {'漂い':>7} {'種類':>6} {'反応':>9} {'再現':>5} {'緩和':>6}")

    ok = True
    for seed in range(SEEDS):
        circuit = build(seed)
        picks = walk(circuit, 100 + seed, wide_features)
        drift, variety, _, _ = measure(picks)
        other = walk(circuit, 900 + seed, wide_features)
        differs = sum(1 for a, b in zip(picks, other)
                      for x, y in zip(a, b) if x != y)
        same = walk(circuit, 100 + seed, wide_features) == picks
        ok = ok and same
        slots = CONVERSATION_COUNT * TURNS
        print(f"{seed:>4} {drift:>7.2f} {variety:>6} {differs:>5}/{slots:<3} "
              f"{'○' if same else '×':>4} {relaxation(circuit):>6}")
    if not ok:
        print("× 同じ入力で選択が変わる。状態の持ち回しが壊れている")
    return ok, baseline


def main():
    ok, baseline = structure()

    print(f"\n== 実測応答で種を選ぶ (条件: 漂い{DRIFT_MIN}〜{DRIFT_MAX} / "
          f"種類{VARIETY_MIN}以上 / 凍結{FROZEN_MAX}本まで / "
          f"最頻句{SHARE_MAX:.0%}まで / 緩和{RELAX_MIN}以上) ==")
    print(f"{'種':>4} {'漂い':>7} {'種類':>6} {'凍結':>6} {'最頻句':>7} "
          f"{'緩和':>6} {'合否':>5}")
    chosen = None
    for seed in range(SEARCH_LIMIT):
        result = check(seed)
        drift, variety, frozen, share, relax = result
        good = passes(*result)
        if good and chosen is None:
            chosen = seed
        print(f"{seed:>4} {drift:>7.2f} {variety:>6} {frozen:>6} "
              f"{share:>6.0%} {relax:>6} {'○' if good else '×':>4}"
              f"{'  ← これを使う' if seed == chosen else ''}")

    if not ok:
        print("\n× 状態の持ち回しが壊れているので、種の選択は無意味")
    elif chosen is None:
        print(f"\n× {SEARCH_LIMIT}種のどれも条件を通らなかった")
    else:
        drift, variety, frozen, share, relax = check(chosen)
        print(f"\n○ 種{chosen} を使う。ランダム{baseline:.2f} < "
              f"漂い{drift:.2f} < 固定{STANCE_COUNT}.00、"
              f"種類{variety}、凍結{frozen}本、最頻句{share:.0%}、"
              f"緩和{relax}ターン")


if __name__ == "__main__":
    main()

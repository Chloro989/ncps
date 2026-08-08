"""
センチュリオン: 跳躍指標の検証 (Phase 3の締め)
ラベルを付けた centurion_samples.txt を読み、跳躍指標が人間の判定を
再現できるかを24件で確かめる。4件では決められなかったことをここで決める。

あわせて、抑圧強度に対して跳躍と崩壊がどう動くかの曲線も出す。
Phase 4 で回路に学ばせたい形が、この曲線に現れているはず。

判定欄が空のままなら何も検証せずに終わる。
スコアを見てからラベルを付けると検証にならないため。
"""

import re
from pathlib import Path

from centurion_score import (
    DEFAULT_WEIGHTS, build_baseline_vocab, measure, parse_samples,
    parse_trace, score,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SAMPLE_FILE = RESULTS / "centurion_samples.txt"
TRACE_FILE = RESULTS / "centurion_trace.txt"

# ===== 検証 =====
def evaluate(samples):
    """お題ごとの重心からの距離と、崩壊の指標を測る"""
    from centurion_embed import centroid, distance

    prompts = sorted({s["prompt"] for s in samples})
    centers, vocabs = {}, {}
    for prompt in prompts:
        texts = parse_trace(TRACE_FILE, prompt)
        if not texts:
            raise SystemExit(f"対照が見つからない: {prompt}")
        centers[prompt] = centroid(texts)
        vocabs[prompt] = build_baseline_vocab(texts)
        print(f"対照 {len(texts)}件: {prompt}")

    for s in samples:
        _, top, mean = distance(s["text"], centers[s["prompt"]])
        s["jump"] = top
        s["jump_mean"] = mean
        m = measure(s["text"], vocabs[s["prompt"]])
        _, _, s["breakdown"] = score(m, DEFAULT_WEIGHTS)
        s["metrics"] = m
    return samples


def report(samples):
    print(f"\n{'標本':<14} {'判定':<4} {'抑圧':>6} {'跳躍':>7} {'崩壊':>7}"
          f" {'ラテン':>6} {'反復':>5} {'逸脱':>5}")
    for s in samples:
        m = s["metrics"]
        print(f"{s['id']:<14} {s['label']:<4} {s['strength']:6.2f}"
              f" {s['jump']:7.4f} {s['breakdown']:7.2f}"
              f" {m['ラテン']:6.2f} {m['反復']:5.2f} {m['逸脱']:5.0f}")


def report_curve(samples):
    """抑圧強度ごとに、跳躍と崩壊がどう動くか"""
    print("\n" + "=" * 60)
    print("抑圧強度と、跳躍・崩壊の関係")
    print("=" * 60)
    print(f"{'抑圧強度':>8} {'件数':>4} {'跳躍(平均)':>11} {'崩壊(平均)':>11}")

    groups = {}
    for s in samples:
        groups.setdefault(round(s["strength"], 1), []).append(s)
    for strength in sorted(groups):
        group = groups[strength]
        jump = sum(s["jump"] for s in group) / len(group)
        broke = sum(s["breakdown"] for s in group) / len(group)
        print(f"{strength:8.2f} {len(group):4d} {jump:11.4f} {broke:11.2f}")


def check(samples):
    """当たりと外れが跳躍で分離するか"""
    hits = [s for s in samples if s["label"] == "○"]
    misses = [s for s in samples if s["label"] == "×"]

    print("\n" + "=" * 60)
    print("検証")
    print("=" * 60)
    if not hits or not misses:
        print("判定欄が埋まっていないため検証できない。")
        print("centurion_samples.txt の「判定:」に ○ か × を書いてから再実行すること。")
        return

    print(f"当たり {len(hits)}件 / 外れ {len(misses)}件")
    for name, key in (("跳躍(文の最大)", "jump"), ("跳躍(文の平均)", "jump_mean"),
                      ("崩壊(小さいほど良い)", "breakdown")):
        sign = -1 if key == "breakdown" else 1
        h = [sign * s[key] for s in hits]
        m = [sign * s[key] for s in misses]
        margin = min(h) - max(m)
        gap = sum(h) / len(h) - sum(m) / len(m)
        print(f"{name:<22} 余裕 {margin:+8.4f}   平均の差 {gap:+8.4f}")

    print("\n余裕が正なら、当たりが全て外れより上に並んでいる。")
    print("余裕が負でも平均の差が正なら、傾向はあるが例外があるということ。")

    report_band(hits, misses)


def spread(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def report_band(hits, misses):
    """跳躍に「ちょうどよい幅」があるという仮説を確かめる。
    遠すぎれば壊れ、近すぎれば轍に沈む — なら当たりは中央に集まるはず"""
    print("\n" + "=" * 60)
    print("仮説: 跳躍には最適な幅がある")
    print("=" * 60)

    h = sorted(s["jump"] for s in hits)
    m = sorted(s["jump"] for s in misses)
    hm, hs = spread(h)
    mm, ms = spread(m)
    print(f"当たり: 平均{hm:.4f} 標準偏差{hs:.4f} 範囲 {h[0]:.4f}〜{h[-1]:.4f}")
    print(f"外れ  : 平均{mm:.4f} 標準偏差{ms:.4f} 範囲 {m[0]:.4f}〜{m[-1]:.4f}")

    if ms <= hs:
        print("\n外れのほうが散らばっていない。帯の仮説は支持されない。")
        return

    print(f"\n外れのほうが {ms / hs:.2f}倍 散らばっている。両端に寄っている可能性。")

    # 中心からの距離で測り直す。ただし中心を当たりから決めれば当然当たりが有利なので、
    # 一件ずつ抜いて、残りから決めた中心で判定する(抜いた一件は中心の計算に使わない)
    correct = 0
    samples = [(s["jump"], True) for s in hits] + [(s["jump"], False) for s in misses]
    for i, (value, is_hit) in enumerate(samples):
        rest_hits = [v for j, (v, k) in enumerate(samples) if k and j != i]
        rest_miss = [v for j, (v, k) in enumerate(samples) if not k and j != i]
        center = sorted(rest_hits)[len(rest_hits) // 2]

        # 残りの標本だけで、中心からの距離のしきい値を決める
        cut = sorted(abs(v - center) for v in rest_hits)[int(len(rest_hits) * 0.75)]
        predicted = abs(value - center) <= cut
        correct += int(predicted == is_hit)

    total = len(samples)
    majority = max(len(hits), len(misses)) / total
    print(f"\n一件抜き検証の正答率: {correct / total:.1%} ({correct}/{total})")
    print(f"多数派に倒すだけの正答率: {majority:.1%}")
    if correct / total > majority:
        print("→ 帯として測れば、多数派に倒すより当たる。仮説は支持される。")
    else:
        print("→ 多数派に倒すのと変わらない。帯としても測れていない。")


def main():
    samples = parse_samples(SAMPLE_FILE)
    hits = sum(1 for s in samples if s["label"] == "○")
    misses = sum(1 for s in samples if s["label"] == "×")
    print(f"標本 {len(samples)}件 (当たり {hits}件 / 外れ {misses}件"
          f" / 未判定 {len(samples) - hits - misses}件)")

    unclear = [s["id"] for s in samples if not s["label"]]
    if unclear:
        print(f"未判定として除外: {', '.join(unclear)}")

    if not any(s["label"] for s in samples):
        print("\n判定欄が全て空。スコアを見てからラベルを付けると検証にならないため、")
        print("ここで止める。centurion_samples.txt の「判定:」を埋めてから再実行すること。")
        return

    evaluate(samples)
    report(samples)
    report_curve(samples)
    check(samples)


main()

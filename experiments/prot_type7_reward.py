"""
センチュリオン: 報酬の決定 (Phase 3 の締め)
単独の指標では判定を再現できなかった。× には2種類あるため:
  轍型   — 流暢だが跳んでいない。尤度は高く、埋め込み距離は小さい
  崩壊型 — 跳んだが読めない。尤度は低く、埋め込み距離は大きい
どちらの軸も、片側だけ見れば必ずもう片方を取り違える。

そこで複数の特徴を組み合わせ、当たりの集まる領域を推定する。
判定は一件抜き検証で行い、その標本を領域の推定から必ず除外する。
特徴の組み合わせを総当たりで試すので、24件では偶然当たるものが出る。
最良の組み合わせを選ぶこと自体が過適合になる点は、結果と併せて報告する。
"""

import itertools
from pathlib import Path

import numpy as np

from centurion_score import (
    DEFAULT_WEIGHTS, build_baseline_vocab, measure, parse_samples,
    parse_trace, score,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SAMPLE_FILE = RESULTS / "centurion_samples.txt"
TRACE_TEXT = RESULTS / "centurion_trace.txt"
FLUENCY_FILE = Path(__file__).resolve().parent / "centurion_fluency.npz"

# 当たりの領域をどこまで広く取るか(当たりの何割を含む距離まで許すか)
COVERAGE = 0.75


def collect():
    """埋め込みの跳躍、尤度、テキスト指標を1つの表にまとめる"""
    from centurion_embed import centroid, distance

    samples = [s for s in parse_samples(SAMPLE_FILE) if s["label"]]
    fluency = np.load(FLUENCY_FILE, allow_pickle=False)
    by_id = {i: n for n, i in enumerate(fluency["id"])}

    prompts = sorted({s["prompt"] for s in samples})
    centers, vocabs = {}, {}
    for prompt in prompts:
        texts = parse_trace(TRACE_TEXT, prompt)
        centers[prompt] = centroid(texts)
        vocabs[prompt] = build_baseline_vocab(texts)

    for s in samples:
        _, top, mean = distance(s["text"], centers[s["prompt"]])
        s["跳躍"] = top
        s["跳躍平均"] = mean

        row = by_id[s["id"]]
        s["尤度"] = float(fluency["mean"][row])
        s["下位尤度"] = float(fluency["p5"][row])

        m = measure(s["text"], vocabs[s["prompt"]])
        s["轍率"] = m["轍率"]
        s["崩壊"] = score(m, DEFAULT_WEIGHTS)[2]
    return samples


def loo_region(features, labels):
    """一件抜き検証。抜いた一件は領域の推定に一切使わない"""
    correct = 0
    for i in range(len(labels)):
        keep = np.arange(len(labels)) != i
        rest, rest_labels = features[keep], labels[keep]

        # 残りだけで標準化し、当たりの重心と許容距離を決める
        mean, std = rest.mean(axis=0), rest.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        scaled = (rest - mean) / std
        hits = scaled[rest_labels]
        center = hits.mean(axis=0)

        radius = np.quantile(np.linalg.norm(hits - center, axis=1), COVERAGE)
        target = (features[i] - mean) / std
        predicted = np.linalg.norm(target - center) <= radius
        correct += int(predicted == labels[i])
    return correct / len(labels)


def main():
    samples = collect()
    labels = np.array([s["label"] == "○" for s in samples])
    majority = max(labels.mean(), 1 - labels.mean())
    print(f"標本 {len(samples)}件 (当たり {labels.sum()} / 外れ {(~labels).sum()})")
    print(f"多数派に倒すだけの正答率: {majority:.1%}\n")

    names = ["跳躍", "跳躍平均", "尤度", "下位尤度", "轍率", "崩壊"]
    table = {n: np.array([s[n] for s in samples]) for n in names}

    print("=" * 62)
    print("特徴の組み合わせごとの正答率 (一件抜き検証)")
    print("=" * 62)

    results = []
    for size in (1, 2, 3):
        for combo in itertools.combinations(names, size):
            features = np.stack([table[n] for n in combo], axis=1)
            results.append((loo_region(features, labels), combo))

    results.sort(reverse=True)
    for accuracy, combo in results[:10]:
        mark = " ←" if accuracy > majority else ""
        print(f"{accuracy:6.1%}  {' + '.join(combo)}{mark}")

    best, combo = results[0]
    beat = sum(1 for a, _ in results if a > majority)
    print(f"\n最良: {' + '.join(combo)} で {best:.1%}")
    print(f"多数派を上回った組み合わせ: {beat} / {len(results)}")

    if best < majority + 0.15:
        print("\n多数派からの改善が小さい。24件の標本では、"
              "この差を実力と偶然に分けられない。")
        print("報酬として使うには、ラベルを増やして確かめ直す必要がある。")
    else:
        print("\n明確に上回っている。この組み合わせを報酬の土台にできる。")

    print(f"\n注意: {len(results)}通りを試して最良を選んでいる。"
          "選んだ時点で過適合しており、")
    print("この正答率は新しい標本での性能ではない。別の標本で確かめること。")


main()

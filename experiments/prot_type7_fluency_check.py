"""
センチュリオン: 対数尤度が判定を再現できるか (Phase 3c の検証)
centurion_fluency.npz を読み、下位の尤度で崩壊が、平均の尤度で跳躍が
分離するかを24件のラベルで確かめる。

判定の基準は「崩壊しているものを弾き、意味はわからないが読めるものを残す」。
これが正しければ、下位の尤度(局所の破れ)がラベルを分け、
平均の尤度(全体の意外性)は分けないはず — 意味が分からなくてもよいのだから。

しきい値は一件抜き検証で決める。24件で6つの統計量を試すので、
たまたま当たるものが出る。過適合を避けるため、その点も併せて報告する。
"""

import numpy as np

TRACE = "centurion_fluency.npz"
STATS = ("min", "p5", "p25", "median", "mean", "broken")

# 統計量の意味。対数尤度は高いほど「素のモデルにも予測できた」
MEANING = {
    "min": "最も予測できなかった1トークン",
    "p5": "下位5%",
    "p25": "下位25%",
    "median": "中央",
    "mean": "全体の平均",
    "broken": "-8以下のトークンの割合",
}


def load():
    data = np.load(TRACE, allow_pickle=False)
    hit = data["label"] == "○"
    miss = data["label"] == "×"
    return data, hit, miss


def report_distribution(data, hit, miss):
    print("=" * 68)
    print("統計量ごとの分布")
    print("=" * 68)
    print(f"{'統計量':<8} {'意味':<24} {'当たり':>16} {'外れ':>16}")
    for key in STATS:
        v = data[key]
        h, m = v[hit], v[miss]
        print(f"{key:<8} {MEANING[key]:<24}"
              f" {h.mean():7.2f}±{h.std():5.2f}"
              f" {m.mean():7.2f}±{m.std():5.2f}")


def loo_threshold(values, labels):
    """一件抜き検証。判定する標本を、しきい値の決定から除外する"""
    correct = 0
    for i in range(len(values)):
        keep = np.arange(len(values)) != i
        rest_v, rest_l = values[keep], labels[keep]

        # 残りの標本で最も当たるしきい値を探す(高い側が当たりと仮定)
        best, best_cut = -1, None
        for cut in np.unique(rest_v):
            acc = ((rest_v >= cut) == rest_l).mean()
            if acc > best:
                best, best_cut = acc, cut

        correct += int((values[i] >= best_cut) == labels[i])
    return correct / len(values)


def report_separation(data, hit, miss):
    print("\n" + "=" * 68)
    print("ラベルとの一致 (高い側を当たりと仮定)")
    print("=" * 68)

    labels = hit[hit | miss]
    majority = max(labels.mean(), 1 - labels.mean())
    print(f"多数派に倒すだけの正答率: {majority:.1%}\n")
    print(f"{'統計量':<8} {'平均の差':>9} {'一件抜き正答率':>14}")

    results = {}
    for key in STATS:
        v = data[key][hit | miss]
        gap = v[labels].mean() - v[~labels].mean()
        acc = loo_threshold(v, labels)
        results[key] = (gap, acc)
        mark = " ←" if acc > majority else ""
        print(f"{key:<8} {gap:+9.3f} {acc:14.1%}{mark}")
    return results, majority


def report_by_strength(data, hit, miss):
    """抑圧強度ごとに、尤度とラベルがどう動くか"""
    print("\n" + "=" * 68)
    print("抑圧強度との関係")
    print("=" * 68)
    print(f"{'抑圧':>6} {'件数':>4} {'当たり率':>8} {'下位5%':>8} {'平均':>8}")

    strength = np.round(data["strength"], 1)
    for value in np.unique(strength):
        pick = strength == value
        labeled = pick & (hit | miss)
        rate = hit[labeled].mean() if labeled.any() else float("nan")
        print(f"{value:6.1f} {pick.sum():4d} {rate:8.0%}"
              f" {data['p5'][pick].mean():8.2f} {data['mean'][pick].mean():8.2f}")


def conclude(results, majority):
    print("\n" + "=" * 68)
    print("判定")
    print("=" * 68)

    low = max(("min", "p5", "p25"), key=lambda k: results[k][1])
    print(f"局所の破れ側で最良: {low} 正答率 {results[low][1]:.1%}")
    print(f"全体の意外性 (mean): 正答率 {results['mean'][1]:.1%}"
          f" 平均の差 {results['mean'][0]:+.3f}")

    if results[low][1] > majority and results["mean"][1] <= majority:
        print("\n狙いどおり。局所の破れがラベルを分け、全体の意外性は分けない。")
        print("「読めるが意味は分からない」を許す基準と整合する。")
        print(f"報酬は {low} を門番にし、跳躍は別に測ればよい。")
    elif results[low][1] > majority:
        print("\n局所の破れは分離するが、全体の意外性も分離してしまっている。")
        print("2軸が絡んでおり、このままでは跳躍を伸ばす報酬にならない。")
    else:
        print("\n対数尤度でもラベルを分離できない。")
        print("崩壊の測り方をさらに考え直す必要がある。")

    print(f"\n注意: 24件で{len(STATS)}個の統計量を試している。"
          "偶然当たるものが1つは出る規模。")


def main():
    data, hit, miss = load()
    print(f"標本 {len(data['id'])}件 (当たり {hit.sum()} / 外れ {miss.sum()})")
    print(f"本文の長さ: 平均{data['tokens'].mean():.0f}トークン\n")

    report_distribution(data, hit, miss)
    results, majority = report_separation(data, hit, miss)
    report_by_strength(data, hit, miss)
    conclude(results, majority)


main()

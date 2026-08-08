"""
センチュリオン: トレースの読み解き (Phase 1の後処理)
Colabで採ったcenturion_trace.npzを開き、Phase 2以降が前提にしている
2つのことを確かめる:
  1. カスタムprocessorが min_p の前後どちらで呼ばれているか
  2. 特徴の実分布が、type6 の固定スケールと比べてどうだったか
"""

import numpy as np

TRACE_FILE = "centurion_trace.npz"
VOCAB_SIZE = 151936        # Qwen2.5-3B-Instruct
FEATURE_NAMES = ["entropy", "top1", "top5", "step"]


def report_order(data):
    """生きている候補の数から、processorの呼ばれる位置を判定する"""
    n = data["n_finite"]
    print("=" * 60)
    print("プロセッサ順序")
    print("=" * 60)
    print(f"語彙数: {VOCAB_SIZE}")
    print(f"生きている候補数: 最小{n.min()} 中央{int(np.median(n))} 最大{n.max()}")

    if n.min() >= VOCAB_SIZE:
        print("→ 全候補が生きている。processorは min_p より前に呼ばれている。")
        print("  エントロピーは素の分布に対する値。")
    else:
        cut = (n < VOCAB_SIZE).mean() * 100
        print(f"→ 候補が絞られている箇所が全体の {cut:.1f}%。")
        print("  processorは min_p より後に呼ばれている。")
        print("  type5/type6 の ENTROPY_GATE は、切り詰め後の分布に対する閾値だった。")


def report_stats(data):
    """Phase 2 の標準化に使う統計"""
    print("\n" + "=" * 60)
    print("特徴の実分布")
    print("=" * 60)
    print(f"{'特徴':<10} {'平均':>8} {'標準偏差':>9} {'最小':>8}"
          f" {'5%':>8} {'50%':>8} {'95%':>8} {'最大':>8}")

    for name in FEATURE_NAMES:
        v = data[name]
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        print(f"{name:<10} {v.mean():8.3f} {v.std():9.3f} {v.min():8.3f}"
              f" {p5:8.3f} {p50:8.3f} {p95:8.3f} {v.max():8.3f}")

    scaled = data["entropy"] / 5.0
    print(f"\ntype6 が使っていた entropy/5.0:")
    print(f"  平均 {scaled.mean():.3f}  標準偏差 {scaled.std():.3f}"
          f"  範囲 {scaled.min():.3f}〜{scaled.max():.3f}")
    print(f"  top1 と top5 の相関: "
          f"{np.corrcoef(data['top1'], data['top5'])[0, 1]:.3f}")


def report_gate(data):
    """type5 の ENTROPY_GATE=3.5 が実際どれだけの箇所を拾っていたか"""
    entropy = data["entropy"]
    print("\n" + "=" * 60)
    print("分岐点の割合 (判明3の裏取り)")
    print("=" * 60)
    for gate in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        rate = (entropy >= gate).mean() * 100
        print(f"エントロピー {gate:.1f} 以上: {rate:5.1f}%")


def main():
    data = np.load(TRACE_FILE)
    print(f"トレース: {len(data['entropy'])}行"
          f" ({len(np.unique(data['prompt_id']))}お題"
          f" × {len(np.unique(data['run_id']))}試行)\n")
    report_order(data)
    report_stats(data)
    report_gate(data)


main()

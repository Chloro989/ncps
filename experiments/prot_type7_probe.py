"""
センチュリオン: 回路単体の応答測定 (Phase 0 / Phase 2の合否判定)
LLMを使わず、合成した特徴列を回路に流して出力の可動域を測る。
定数入力を対照に引くことで、回路自身の自律過渡と入力への感度を切り分ける。
V1(type6の回路)とV2(修正版)を同じ入力で並べて比較する。
CPUで数秒で終わるので、回路を直すたびにこれを回す。
"""

import torch

from centurion_circuit import (
    CenturionCircuitV1, CenturionCircuitV2,
    FEATURE_NAMES, OUTPUT_COUNT_V1, OUTPUT_COUNT_V2,
    STRENGTH_MAX, TOP_K_MAX, WIDTH_MIN, WIDTH_MAX,
    decode_v2, load_stats,
)

# ===== 設定 =====
STEPS = 150            # 1回の生成に相当する長さ
SEEDS = 5              # 初期値を変えて何通り試すか
VERBOSE = False        # 各初期値の全テストを表示するか

# 特徴の素の値域。トレースの実測に合わせてある
FEATURE_RANGE = [(0.0, 6.5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

# 入力を振ったときの可動域が、定数入力のときより
# これ以上広がらなければ「入力に反応していない」とみなす
SENSITIVITY_FLOOR = 0.05

# 対照は特徴の平均で固定する。V2はこの点で定常化させてあるので、
# ここで動く分だけが純粋な自律過渡になる
BASELINE_TEST = "定数 平均"


# ===== 入力する特徴列 =====
def at(frac):
    """各特徴の値域を、共通の割合で指定する"""
    return [lo + frac * (hi - lo) for lo, hi in FEATURE_RANGE]


def constant_sequence(frac):
    """全特徴を固定し続ける。ここで動く分は回路の自律過渡"""
    return torch.tensor(at(frac)).repeat(STEPS, 1)


def mean_sequence(mean):
    """特徴の平均で固定した列。他の特徴を留めておく土台にも使う"""
    return torch.tensor(mean, dtype=torch.float32).repeat(STEPS, 1)


def step_sequence(index, mean):
    """特徴を1つだけ、途中で下限から上限へ跳ね上げる"""
    seq = mean_sequence(mean)
    lo, hi = FEATURE_RANGE[index]
    seq[:STEPS // 2, index] = lo
    seq[STEPS // 2:, index] = hi
    return seq


def sweep_sequence(index, mean):
    """特徴を1つだけ、下限から上限へゆっくり動かす"""
    seq = mean_sequence(mean)
    seq[:, index] = torch.linspace(*FEATURE_RANGE[index], STEPS)
    return seq


def random_walk_sequence():
    """実際の生成に近い、緩やかに揺れる入力"""
    span = torch.tensor([hi - lo for lo, hi in FEATURE_RANGE])
    lo = torch.tensor([lo for lo, _ in FEATURE_RANGE])
    start = lo + torch.rand(1, len(FEATURE_RANGE)) * span
    noise = torch.randn(STEPS, len(FEATURE_RANGE)).cumsum(dim=0) * 0.03 * span
    seq = start.repeat(STEPS, 1) + noise
    return torch.max(torch.min(seq, lo + span), lo)


def build_tests(mean):
    tests = [
        (BASELINE_TEST, mean_sequence(mean)),
        ("定数 下限", constant_sequence(0.0)),
        ("定数 上限", constant_sequence(1.0)),
        ("ランダムウォーク", random_walk_sequence()),
    ]
    for i, name in enumerate(FEATURE_NAMES):
        tests.append((f"ステップ {name}", step_sequence(i, mean)))
    for i, name in enumerate(FEATURE_NAMES):
        tests.append((f"掃引 {name}", sweep_sequence(i, mean)))
    return tests


def adapt(seq, version):
    """V1 は entropy を 5.0 で割った値を受け取る前提だった"""
    if version == 1:
        seq = seq.clone()
        seq[:, 0] = seq[:, 0] / 5.0
    return seq


# ===== 計測 =====
def run(circuit, seq):
    """特徴列を1ステップずつ流し、出力を積み上げる"""
    circuit.reset()
    outputs = []
    with torch.no_grad():
        for row in seq:
            outputs.append(circuit(row).clone())
    return torch.stack(outputs)


def control_span(out, version):
    """出力を実際の制御値に写像したときの可動域"""
    if version == 1:
        strength = out[:, 0] * STRENGTH_MAX
        top_k = (out[:, 1] * TOP_K_MAX).round().clamp(min=1)
        return ((strength.max() - strength.min()).item(),
                int(top_k.unique().numel()))

    gate, strength, width = decode_v2(out.T)
    effective = gate * strength
    return ((effective.max() - effective.min()).item(),
            (width.max() - width.min()).item())


def measure(circuit, tests, version):
    """各テストでの出力と可動域を集める"""
    result = {}
    for name, seq in tests:
        out = run(circuit, adapt(seq, version))
        spans = out.max(dim=0).values - out.min(dim=0).values
        result[name] = (out, spans)
    return result


def summarize(result, version):
    """定数入力を対照に引き、自律過渡と入力感度を分ける"""
    baseline = result[BASELINE_TEST][1]
    varying = [(out, spans) for name, (out, spans) in result.items()
               if not name.startswith("定数")]

    sens = torch.stack([(spans - baseline).clamp(min=0.0)
                        for _, spans in varying]).max(dim=0).values
    level = torch.stack([out.mean(dim=0) for out, _ in varying]).mean(dim=0)
    ctrl = [control_span(out, version) for out, _ in varying]

    return {
        "drift": baseline,
        "sens": sens,
        "level": level,
        "ctrl_strength": max(c[0] for c in ctrl),
        "ctrl_width": max(c[1] for c in ctrl),
    }


def report_detail(seed, result, version):
    print(f"\n--- V{version} seed={seed} ---")
    baseline = result[BASELINE_TEST][1]
    header = " ".join(f"{'幅' + str(i):>8}" for i in range(len(baseline)))
    print(f"{'テスト':<22}{header}")
    for name, (_, spans) in result.items():
        excess = (spans - baseline).clamp(min=0.0)
        cells = " ".join(f"{v.item():8.4f}" for v in excess)
        print(f"{name:<22}{cells}")


# ===== 判定 =====
def report_version(version, records):
    n = OUTPUT_COUNT_V1 if version == 1 else OUTPUT_COUNT_V2
    drift = torch.stack([r["drift"] for r in records]).max(dim=0).values
    sens = torch.stack([r["sens"] for r in records]).max(dim=0).values
    level = torch.stack([r["level"] for r in records])

    labels = (["強度", "幅"] if version == 1 else ["ゲート", "強度", "幅"])
    print(f"\n--- V{version} ---")
    print(f"{'出力':<8} {'自律過渡':>9} {'入力感度':>9} {'感度比':>8}"
          f" {'出力の居場所':>16}")
    for i in range(n):
        # 定常化が効いていると過渡がゼロになり、比が意味を失う
        ratio = ("過渡なし" if drift[i].item() < 1e-6
                 else f"{sens[i].item() / drift[i].item():.2f}")
        print(f"{labels[i]:<8} {drift[i].item():9.4f} {sens[i].item():9.4f}"
              f" {ratio:>8}"
              f" {level[:, i].min().item():7.3f}〜{level[:, i].max().item():.3f}")

    strength = max(r["ctrl_strength"] for r in records)
    width = max(r["ctrl_width"] for r in records)
    if version == 1:
        print(f"実効抑圧の可動域: {strength:.3f} (最大 {STRENGTH_MAX})")
        print(f"抑圧幅がとった値の種類: {int(width)}種類 (round による量子化)")
    else:
        print(f"実効抑圧の可動域: {strength:.3f} (最大 {STRENGTH_MAX})")
        print(f"実効幅の可動域: {width:.3f}"
              f" (可動範囲 {WIDTH_MAX - WIDTH_MIN})")

    return sens, drift


def verdict(sens_v1, drift_v1, sens_v2, drift_v2):
    print("\n" + "=" * 60)
    print("判定")
    print("=" * 60)

    # V2 は 強度(1) と 幅(2) が入力に反応していれば合格とする
    worst = min(sens_v2[1].item(), sens_v2[2].item())
    drift = max(drift_v2[1].item(), drift_v2[2].item())
    buried = drift > 1e-6 and worst < drift

    print(f"V1 の入力感度(最大): {sens_v1.max().item():.4f}"
          f" / 自律過渡 {drift_v1.max().item():.4f}")
    print(f"V2 の入力感度(最小、強度と幅): {worst:.4f}"
          f" / 自律過渡 {drift:.4f}")

    if worst > SENSITIVITY_FLOOR and not buried:
        print(f"\n合格。V2 は入力を {SENSITIVITY_FLOOR} 以上の幅で読んでおり、"
              "自律過渡に埋もれていない。")
        print("学習させる土台としては成立した。")
    elif worst > SENSITIVITY_FLOOR:
        print("\n不十分。可動域は足りているが、自律過渡に埋もれている。")
        print("初期状態の扱いか、入力射影のゲインを見直すこと。")
    else:
        print(f"\n不合格。V2 の可動域が {SENSITIVITY_FLOOR} に届かない。")
        print("入力射影のゲインと出力ゲインを上げること。")


# ===== 実行 =====
def main():
    mean, std, measured = load_stats()
    tests = build_tests(mean)
    source = "Phase 1 の実測" if measured else "暫定値"
    print(f"標準化統計: {source}")
    print(f"  平均 {[round(v, 3) for v in mean]}")
    print(f"  標準偏差 {[round(v, 3) for v in std]}")
    print(f"入力: {STEPS}ステップ × {len(tests)}テスト × {SEEDS}初期値")

    summaries = {1: [], 2: []}
    for version in (1, 2):
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            if version == 1:
                circuit = CenturionCircuitV1()
            else:
                circuit = CenturionCircuitV2((mean, std))
                circuit.settle()   # ゼロ状態からの立ち上がりを測定に混ぜない
            if seed == 0:
                n = sum(p.numel() for p in circuit.parameters())
                print(f"V{version} パラメータ数: {n}")

            result = measure(circuit, tests, version)
            if VERBOSE:
                report_detail(seed, result, version)
            summaries[version].append(summarize(result, version))

    print("\n" + "=" * 60)
    print("集計 (全初期値・全テストを通した最大値)")
    print("=" * 60)
    sens_v1, drift_v1 = report_version(1, summaries[1])
    sens_v2, drift_v2 = report_version(2, summaries[2])
    verdict(sens_v1, drift_v1, sens_v2, drift_v2)


main()

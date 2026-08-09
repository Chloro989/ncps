"""
センチュリオン: 学習済み回路が何をしているかを見る
盲検で差が出なかった。回路が実際にどう振る舞っているかを、
Phase 1 の実トレースを流して確かめる。

見たいのは1点 — ゲートが働いているかどうか。
type5 は「エントロピー3.5以上でのみ、上位2個を-2.0」という硬い門で
跳躍を生んだ。type6 は門を外して一律に抑圧し、失敗した。
学習済み回路が門を持たず一律に抑圧しているなら、type6 に戻っている。
"""

import numpy as np
import torch

from centurion_circuit import (
    CenturionCircuitV2, decode_v2, load_stats, unpack,
)

CHECKPOINT = "centurion_circuit.pt"
TRACE = "centurion_trace.npz"
GATE_REFERENCE = 3.5      # type5 が使っていたエントロピーの門


def run(circuit, features):
    """実トレースを流し、ゲート・強度・幅の推移を集める"""
    circuit.reset()
    rows = []
    with torch.no_grad():
        for row in features:
            control = circuit(torch.tensor(row, dtype=torch.float32))
            gate, strength, width = decode_v2(control)
            rows.append((gate.item(), strength.item(), width.item()))
    return np.array(rows)


def describe(name, values):
    print(f"{name:<10} 平均{values.mean():7.3f} 標準偏差{values.std():7.3f}"
          f" 範囲 {values.min():.3f}〜{values.max():.3f}")


def main():
    data = np.load(TRACE)
    mean, std, _ = load_stats(TRACE)

    # 1試行ぶん(150トークン)を流す
    length = 150
    features = np.stack([data["entropy"][:length], data["top1"][:length],
                         data["top5"][:length], data["step"][:length]], axis=1)
    entropy = features[:, 0]

    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    trained = unpack(CenturionCircuitV2((saved["mean"], saved["std"])),
                     saved["theta"])
    torch.manual_seed(0)
    untrained = CenturionCircuitV2((mean, std))
    untrained.settle()

    for name, circuit in (("学習済み", trained), ("未学習", untrained)):
        out = run(circuit, features)
        gate, strength, width = out[:, 0], out[:, 1], out[:, 2]
        effective = gate * strength

        print(f"\n--- {name} ---")
        describe("ゲート", gate)
        describe("抑圧強度", strength)
        describe("実効抑圧", effective)
        describe("実効幅", width)

        # 門として働いているか。分岐点とそれ以外で実効抑圧が変わるか
        branch = entropy >= GATE_REFERENCE
        print(f"エントロピー{GATE_REFERENCE}以上: {branch.mean():.0%}のトークン")
        print(f"  分岐点での実効抑圧   {effective[branch].mean():.3f}")
        print(f"  それ以外での実効抑圧 {effective[~branch].mean():.3f}")
        ratio = effective[branch].mean() / max(effective[~branch].mean(), 1e-9)
        print(f"  比 {ratio:.2f}倍")

        correlation = np.corrcoef(entropy, effective)[0, 1]
        print(f"エントロピーと実効抑圧の相関: {correlation:+.3f}")

    print("\n" + "=" * 56)
    print("読み方")
    print("=" * 56)
    print("比が1.0付近なら、門は働いておらず一律に抑圧している。")
    print("type5 は分岐点だけを叩いて跳躍を生み、type6 は一律に叩いて失敗した。")
    print("学習済みの比が1.0付近なら、type6 と同じことをしていることになる。")


main()

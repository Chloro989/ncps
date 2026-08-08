"""
センチュリオン: 抑圧を制御する回路の定義 (Phase 2)
probe・学習ループ・評価の全てから import して使う共有モジュール。
実行しても何も起きない。計測は prot_type7_probe.py 側で行う。

V1 は type6 の回路をそのまま写したもの(比較用)。
V2 は Phase 0 の診断を受けた修正版。直したのは4点:
  1. 特徴を実測統計で標準化し、射影層を通してから回路に入れる
     (生特徴は分散が小さく、CfCの入力重みも小さいため信号が埋もれていた)
  2. 出力に学習可能なゲインとバイアスを掛け、sigmoidの可動域を使い切る
     (V1は0.46〜0.60の帯にしか居なかった)
  3. 抑圧幅の round() を廃止し、順位に沿って滑らかに減衰する重みにする
     (整数への量子化で情報が死に、進化戦略の探索面も平坦になっていた)
  4. 初期状態を学習可能にする
     (V1は自律過渡が入力応答より大きく、初期状態に振り回されていた)
さらに出力を3つにして、type5 の ENTROPY_GATE 相当を回路に学ばせる。
"""

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ===== 回路の構成 =====
NEURON_COUNT = 16
FEATURE_COUNT = 4

OUTPUT_COUNT_V1 = 2      # 抑圧強度, 抑圧幅
OUTPUT_COUNT_V2 = 3      # ゲート, 抑圧強度, 抑圧幅

PROJ_DIM = 8             # 特徴を一度この次元に持ち上げてから回路に入れる
INPUT_GAIN = 2.0         # 射影後の信号をtanhの効く範囲まで押し上げる
OUTPUT_GAIN_INIT = 4.0   # sigmoidに入る前のゲインの初期値

# ===== 抑圧の可動域 =====
STRENGTH_MAX = 4.0       # 抑圧の最大値
TOP_K_MAX = 4            # V1が使っていた抑圧候補数の上限
SUPPRESS_SPAN = 8        # V2が触れる上位候補の数(実効幅は重みが決める)
WIDTH_MIN = 0.5          # 実効幅の下限(ほぼ1位だけを叩く)
WIDTH_MAX = 4.0          # 実効幅の上限(上位に広く掛ける)

FEATURE_NAMES = ["entropy", "top1", "top5", "step"]

# Phase 1 のトレースが無いときに使う暫定統計。
# centurion_trace.npz が手に入り次第 load_stats() で実測値に差し替わる。
PROVISIONAL_MEAN = [2.0, 0.50, 0.85, 0.50]
PROVISIONAL_STD = [1.0, 0.25, 0.15, 0.29]


def load_stats(path="centurion_trace.npz"):
    """Phase 1 のトレースから標準化統計を読む。無ければ暫定値を返す"""
    try:
        data = np.load(path)
    except OSError:
        return list(PROVISIONAL_MEAN), list(PROVISIONAL_STD), False

    mean, std = [], []
    for name in FEATURE_NAMES:
        v = data[name]
        mean.append(float(v.mean()))
        # 分散がほぼ無い特徴で割って発散させないための下限
        std.append(max(float(v.std()), 1e-3))
    return mean, std, True


# ===== V1: type6 の回路(比較用) =====
class CenturionCircuitV1(nn.Module):
    """線虫の神経回路構造を模した、抑圧を制御する小さな回路"""

    def __init__(self):
        super().__init__()
        wiring = AutoNCP(NEURON_COUNT, OUTPUT_COUNT_V1)
        self.rnn = CfC(FEATURE_COUNT, wiring, batch_first=True)
        self.state = None

    def reset(self):
        self.state = None

    def forward(self, features):
        # V1は entropy を 5.0 で割るだけの固定スケールだった
        x = features.view(1, 1, -1)
        out, self.state = self.rnn(x, self.state)
        return torch.sigmoid(out.view(-1))


# ===== V2: 修正版 =====
class CenturionCircuitV2(nn.Module):
    """特徴を標準化して読み、ゲート付きで抑圧を制御する回路"""

    def __init__(self, stats=None):
        super().__init__()
        mean, std = stats if stats else (PROVISIONAL_MEAN, PROVISIONAL_STD)
        self.register_buffer("feat_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(std, dtype=torch.float32))

        self.proj = nn.Linear(FEATURE_COUNT, PROJ_DIM)
        wiring = AutoNCP(NEURON_COUNT, OUTPUT_COUNT_V2)
        self.rnn = CfC(PROJ_DIM, wiring, batch_first=True)

        # 出力の可動域を広げるための、学習可能なゲインとバイアス
        self.gain = nn.Parameter(
            torch.full((OUTPUT_COUNT_V2,), OUTPUT_GAIN_INIT))
        self.bias = nn.Parameter(torch.zeros(OUTPUT_COUNT_V2))

        # 初期状態を学習させ、自律過渡を制御下に置く
        self.init_state = nn.Parameter(torch.zeros(1, NEURON_COUNT))
        self.state = None

    def reset(self):
        # Parameterをそのまま代入するとstateがパラメータとして登録されてしまうので、
        # clone()でテンソルにして渡す(学習時の勾配の接続は保たれる)
        self.state = self.init_state.clone()

    def settle(self, steps=200):
        """平均的な入力に対する定常状態を初期状態にする。
        ゼロ状態からの立ち上がりは入力応答と同じ大きさになるため、
        そこから始めると何に反応しているのか分からなくなる"""
        with torch.no_grad():
            self.state = torch.zeros(1, NEURON_COUNT,
                                     device=self.feat_mean.device)
            for _ in range(steps):
                self.forward(self.feat_mean)
            self.init_state.data.copy_(self.state)
        self.reset()

    def forward(self, features):
        # 標準化してから射影する。LayerNormは使わない —
        # 特徴間で正規化すると「全体的に高い/低い」という情報が消えてしまう
        x = (features - self.feat_mean) / self.feat_std
        x = torch.tanh(self.proj(x) * INPUT_GAIN)

        out, self.state = self.rnn(x.view(1, 1, -1), self.state)
        return torch.sigmoid(out.view(-1) * self.gain + self.bias)


# ===== 制御値への変換 =====
def decode_v2(control):
    """回路の出力を、意味のある制御値に読み替える"""
    gate = control[0]
    strength = control[1] * STRENGTH_MAX
    width = WIDTH_MIN + control[2] * (WIDTH_MAX - WIDTH_MIN)
    return gate, strength, width


def soft_mask(width, span=SUPPRESS_SPAN, device=None):
    """順位に沿って滑らかに減衰する抑圧の重み。widthが実効幅を決める"""
    ranks = torch.arange(span, dtype=torch.float32, device=device)
    return torch.exp(-ranks / width)


def apply_suppression(scores, control):
    """回路の出力に従って上位候補を抑圧する。round()による量子化はしない"""
    gate, strength, width = decode_v2(control)
    effective = gate * strength
    if effective.item() <= 0.0:
        return scores, effective, width

    _, indices = scores[0].topk(SUPPRESS_SPAN)
    mask = soft_mask(width, device=scores.device)
    scores[0, indices] -= effective * mask
    return scores, effective, width

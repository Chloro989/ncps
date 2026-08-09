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
# ===== V3: type5 の構造に閉じ込めた回路 =====
# V2は自由度を与えすぎた。学習の結果、門をまったく使わず
# 分岐点でもそれ以外でも同じ強さ(比0.96)で、上位8個すべてに広く抑圧していた。
# これは type5 の正反対で、失敗した type6 と同じ振る舞い。
# 報酬のどの項も「分岐点だけを叩け」と要求していなかったのが原因。
#
# V3では選択性を報酬に頼らず構造で保証する。
# 抑圧の上限をエントロピーの単調関数で押さえ、
# 回路はその内側で強さと幅を調整することしかできない。
# 低エントロピーの箇所を叩くことは、回路がどう学んでも不可能になる。
# 閾値の下限は3.0にする。エントロピーの中央値が1.93なので、
# これ以下に下げられると門が開きっぱなしになり選択性が消える
THRESHOLD_MIN = 3.0
THRESHOLD_MAX = 5.0
THRESHOLD_INIT = 3.5      # type5 が使っていた値

# 門の切れ味は学習させない。定数にする。
# 最初これを学習対象にしたところ、パラメータを大きく摂動すると
# 切れ味が0に潰れて門が平坦になり、選択性が消えることが分かった。
# 前回の学習は報酬の穴を突いて退化解に落ちたので、ここも必ず突かれる
SHARPNESS = 4.0

V3_STRENGTH_MIN = 1.0     # type5 の 2.0 を挟む範囲
V3_STRENGTH_MAX = 3.0
V3_WIDTH_MIN = 0.3        # 実効的に上位1〜2個
V3_WIDTH_MAX = 1.5        # 実効的に上位3〜4個。type5 の「上位2個」に対応する幅

# 学習の出発点。盲検で12戦12勝した type5 の設定に合わせる。
# 未学習V3は分岐点での実効抑圧が0.48しかなく、type5の2.0に遠く及ばなかった。
# 検証済みの点から始めるほうが、偶然そこへ辿り着くのを待つより確か
TYPE5_GATE_MOD = 0.95     # 門が開いているときはほぼ全開にする
TYPE5_STRENGTH = 2.1      # type5 の 2.0 相当
TYPE5_WIDTH = 0.5         # 減衰が速く、実効的に上位2個になる幅


class CenturionCircuitV3(CenturionCircuitV2):
    """エントロピーの門を構造として持つ回路。
    門の閾値と切れ味も学習するが、門そのものは外せない"""

    def __init__(self, stats=None):
        super().__init__(stats)
        # 閾値はsigmoidで範囲に写す。初期値がちょうど THRESHOLD_INIT になるようにする
        span = THRESHOLD_MAX - THRESHOLD_MIN
        start = (THRESHOLD_INIT - THRESHOLD_MIN) / span
        self.raw_threshold = nn.Parameter(
            torch.tensor(float(np.log(start / (1 - start)))))

    def gate_from_entropy(self, entropy):
        """エントロピーだけで決まる、抑圧の上限。回路はこれを超えられない。
        切れ味が定数で、閾値もsigmoidで範囲に閉じているので、
        パラメータをどう動かしても門を平坦にはできない"""
        threshold = (THRESHOLD_MIN
                     + torch.sigmoid(self.raw_threshold)
                     * (THRESHOLD_MAX - THRESHOLD_MIN))
        return torch.sigmoid((entropy - threshold) * SHARPNESS)

    def match_type5(self, rounds=4):
        """平均的な入力に対する出力が type5 の設定になるよう、バイアスを合わせる。
        バイアスを変えると定常状態も動くので、数回繰り返して寄せる"""
        # 回路と同じデバイスに置く。CPUで作るとGPU実行時に落ちる
        targets = torch.tensor([
            TYPE5_GATE_MOD,
            (TYPE5_STRENGTH - V3_STRENGTH_MIN) / (V3_STRENGTH_MAX - V3_STRENGTH_MIN),
            (TYPE5_WIDTH - V3_WIDTH_MIN) / (V3_WIDTH_MAX - V3_WIDTH_MIN),
        ], device=self.bias.device, dtype=self.bias.dtype).clamp(1e-3, 1 - 1e-3)

        for _ in range(rounds):
            self.settle()
            with torch.no_grad():
                raw = self.forward(self.feat_mean).clamp(1e-3, 1 - 1e-3)
                # sigmoidの逆関数で、必要なバイアスのずれを求める
                self.bias += torch.logit(targets) - torch.logit(raw)
        self.settle()

    def control(self, features):
        """実効抑圧と実効幅を返す。features[0] は素のエントロピー"""
        raw = self.forward(features)          # V2と同じ3出力 (0〜1)
        gate = self.gate_from_entropy(features[0]) * raw[0]
        strength = V3_STRENGTH_MIN + raw[1] * (V3_STRENGTH_MAX - V3_STRENGTH_MIN)
        width = V3_WIDTH_MIN + raw[2] * (V3_WIDTH_MAX - V3_WIDTH_MIN)
        return gate * strength, width


def apply_values(scores, effective, width):
    """決まった強さと幅で上位候補を抑圧する"""
    _, indices = scores[0].topk(SUPPRESS_SPAN)
    mask = soft_mask(width, device=scores.device)
    scores[0, indices] -= effective * mask
    return scores


# ===== 学習で動かすパラメータ =====
# ncps は配線マスク (sparsity_mask) もParameterとして持っている。
# これは線虫由来の結線そのもので、摂動すると回路の構造が壊れる。
# 進化戦略の対象から必ず外すこと。非連続でもあるので
# parameters_to_vector がそのままでは通らない
MASK_NAME = "sparsity_mask"


def trainable_parameters(circuit):
    """配線マスクを除いた、学習してよいパラメータ"""
    return [p for name, p in circuit.named_parameters() if MASK_NAME not in name]


def pack(circuit):
    """学習対象を1本のベクトルにまとめる"""
    return torch.cat([p.detach().reshape(-1)
                      for p in trainable_parameters(circuit)])


def unpack(circuit, vector):
    """ベクトルを回路に書き戻す。配線マスクには触れない"""
    offset = 0
    with torch.no_grad():
        for p in trainable_parameters(circuit):
            count = p.numel()
            p.copy_(vector[offset:offset + count].view_as(p))
            offset += count
    return circuit


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
    """回路の出力に従って上位候補を抑圧する。round()による量子化はしない。
    gate も strength も sigmoid 由来で常に正なので、早期打ち切りはしない —
    ここで .item() を呼ぶと生成のたびにGPU同期が入り、学習ループが遅くなる"""
    gate, strength, width = decode_v2(control)
    effective = gate * strength

    _, indices = scores[0].topk(SUPPRESS_SPAN)
    mask = soft_mask(width, device=scores.device)
    scores[0, indices] -= effective * mask
    return scores, effective, width

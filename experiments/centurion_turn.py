"""
センチュリオン: 複数ターンの会話と、姿勢の漂い (Phase 10)

Phase 9 で流動プロンプトが勝った(20勝12敗)。ただし句の選択は
生成ごとに独立で、そこには状態がない。回路の出番もない。

複数ターンにすると状態が生まれる。ターンをまたいで姿勢が漂い、
前のターンの語り方が次のターンに影響する。時定数が入力依存である
液体回路が、初めて意味を持つ設計になる。

回路が勝つべき相手は固定プロンプトではなくランダム流動である。
姿勢が変わるだけならランダムで足りる。回路が加えられるのは
**変化の連続性** — ランダムは毎ターン独立に飛ぶが、
回路は記憶を持つので前のターンを引き継いで漂う。
仮説は「連続した漂いは一貫した語り手として読め、独立な飛びは不安定に読める」。

学習は使わない。報酬は5回作ろうとして作れなかった。
連続性は CfC の記憶による構造的性質として持たせる。
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from centurion_prompts import (
    CORE_CLAUSES, PREFILL, RUT_WORDS, STANCE_CLAUSES, USER_PROMPTS,
    STANCE_COUNT,
)
from centurion_score import words

NEURON_COUNT = 32          # 出力が9つあるので Phase 7 の16では足りない
FEATURE_COUNT = 4
PROJ_DIM = 8
INPUT_GAIN = 2.0
OUTPUT_GAIN_INIT = 4.0

# CfC の default モードには状態を保つ項がない
# (new_hidden = ff1*(1-t) + t*ff2 で、どちらも毎ステップ再計算される)。
# 素のままだと隣のターンで共有する句が0.44で、ランダム選択の0.46と変わらなかった。
# mixed_memory は LSTM を挟んで状態を保つ。これで0.98になり、
# 固定(2.00)とランダム(0.46)の間の「漂い」になる。
# 歩幅も入力の強さも記憶にはならない(順位は倍率に対して不変なため)
MIXED_MEMORY = True

# 特徴の暫定的な正規化。実測統計を取っていないので、
# 中央がおよそ0になるようにだけ揃えてある。
# Phase 0 の教訓 — 正規化しない特徴を入れると回路が入力を読めなくなる
FEATURE_MEAN = [0.5, 1.0, 1.0, 0.7]
FEATURE_STD = [0.3, 1.0, 0.4, 0.2]


class StanceCircuit(nn.Module):
    """会話の状態から、次のターンで使う姿勢の点数を出す回路。
    CenturionCircuitV2 と同じ作りだが出力数が姿勢の句数になる"""

    def __init__(self, output_count):
        super().__init__()
        self.register_buffer("feat_mean",
                            torch.tensor(FEATURE_MEAN, dtype=torch.float32))
        self.register_buffer("feat_std",
                             torch.tensor(FEATURE_STD, dtype=torch.float32))
        self.proj = nn.Linear(FEATURE_COUNT, PROJ_DIM)
        wiring = AutoNCP(NEURON_COUNT, output_count)
        self.rnn = CfC(PROJ_DIM, wiring, batch_first=True,
                       mixed_memory=MIXED_MEMORY)
        self.gain = nn.Parameter(
            torch.full((output_count,), OUTPUT_GAIN_INIT))
        self.bias = nn.Parameter(torch.zeros(output_count))
        self.state = None
        self.init_state = None
        self.baseline = None

        # 配線マスクは非連続で、学習対象でもない。連続化して扱いを揃える
        for parameter in self.parameters():
            parameter.data = parameter.data.contiguous()

    def blank(self):
        """まだ何も見ていない状態。混合記憶では隠れ状態とセルの2つ組になる"""
        zero = torch.zeros(1, NEURON_COUNT, device=self.feat_mean.device)
        return [zero, zero.clone()] if MIXED_MEMORY else zero

    @staticmethod
    def copy_state(state):
        """混合記憶では CfC が (隠れ状態, セル) の組を返してくる"""
        if isinstance(state, (list, tuple)):
            return [s.clone() for s in state]
        return state.clone()

    def reset(self):
        self.state = self.copy_state(self.init_state)

    def settle(self, features=None, steps=200):
        """与えた入力に対する定常状態を初期状態にし、そのときの出力を
        基準値として覚える。ゼロ状態からの立ち上がりを会話に混ぜないため。

        基準値が要る理由: 出力ごとに固有のバイアスがあり、順位がそれに
        支配されて12会話で2種類しか姿勢が出なかった。
        水準ではなく基準値からのずれで順位を決めれば固有のバイアスが消える。

        基準を会話ごとに置き直す理由: 想定した特徴の平均が実測とずれて
        いると、偏差の大部分が毎ターン同じ定数になる。実際に偏差の可動域
        0.29に対して2位と3位の差は0.03しかなく、16会話中14本で姿勢が
        凍結した。会話が始まった場所に基準を置けばオフセットが消え、
        まだ緩和しきっていない状態としての記憶だけが残る"""
        if features is None:
            features = self.feat_mean
        with torch.no_grad():
            self.state = self.blank()
            for _ in range(steps):
                out = self.forward(features)
            self.init_state = self.copy_state(self.state)
            self.baseline = out.clone()
        self.reset()

    def forward(self, features):
        x = (features - self.feat_mean) / self.feat_std
        x = torch.tanh(self.proj(x) * INPUT_GAIN)
        out, self.state = self.rnn(x.view(1, 1, -1), self.state)
        return torch.sigmoid(out.view(-1) * self.gain + self.bias)


def response_features(text, turn, total_turns, rut_words=RUT_WORDS):
    """直前の応答から、回路に渡す4つの数値を作る。
    1ターン目は前の応答がないので中央値を入れる。

    轍語は部分一致で数える。centurion_score.words() は漢字2文字以上を
    拾う作りなので「星」と「囁く」が漏れ、「宇宙的」も別語になってしまう。
    Phase 9 の答え合わせで使った数え方(text.count)に揃えた"""
    progress = turn / max(total_turns - 1, 1)
    if not text:
        return torch.tensor([progress] + FEATURE_MEAN[1:],
                            dtype=torch.float32)

    rut = sum(text.count(word) for word in rut_words) / len(text) * 100
    length = len(text) / 150.0
    content = words(text)
    richness = len(set(content)) / len(content) if content else 0.0
    return torch.tensor([progress, rut, length, richness],
                        dtype=torch.float32)


def compose(stance, banned):
    """姿勢と禁止語からシステムプロンプトを組む。流動版と同じ並び"""
    parts = [CORE_CLAUSES[0], CORE_CLAUSES[1]]
    parts.extend(stance)
    parts.append(
        "・".join(banned) + "といった言葉と、その常套的な連想には頼らない。")
    parts.extend(CORE_CLAUSES[2:])
    return "".join(parts)


def pick_by_circuit(circuit, features, count=STANCE_COUNT):
    """回路の点数が高い姿勢を選ぶ。記憶があるので前のターンを引き継ぐ。
    順位は基準値からのずれで決める — 出力固有のバイアスを外すため"""
    with torch.no_grad():
        scores = circuit(features)
    baseline = 0.0 if circuit.baseline is None else circuit.baseline
    deviation = scores - baseline
    order = deviation.argsort(descending=True)[:count].tolist()
    return [STANCE_CLAUSES[i] for i in order], deviation


def build_conversation(tokenizer, system_prompt, history, user_prompt):
    """これまでのやり取りを含めた前置きを組む。
    history は (お題, 応答) の並び"""
    messages = [{"role": "system", "content": system_prompt}]
    for past_prompt, past_reply in history:
        messages.append({"role": "user", "content": past_prompt})
        messages.append({"role": "assistant", "content": past_reply})
    messages.append({"role": "user", "content": user_prompt})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) + PREFILL


TURNS = 4
SHARED_TURNS = 1           # 1ターン目は両条件で共通にして、始まりを揃える

# 会話の流れ。同じお題の並びを両条件に与えるので、違いは姿勢の選び方だけになる。
#
# 手で組むと自分の好みでお題の並びを選んでしまうので、巡回でずらして作る。
# ずらし幅を全て違えると、16お題それぞれが4つのターン位置に
# ちょうど1回ずつ現れる。ターン位置による効果(1ターン目が有利など)が
# お題の偏りと混ざらない。会話の中でお題が重複することもない。
TURN_OFFSETS = [0, 5, 9, 13]
CONVERSATIONS = [
    [USER_PROMPTS[(index + offset) % len(USER_PROMPTS)]
     for offset in TURN_OFFSETS]
    for index in range(len(USER_PROMPTS))
]

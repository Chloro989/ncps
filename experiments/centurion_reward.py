"""
センチュリオン: 学習の報酬 (Phase 4)
Phase 3 で「人間の判定を再現する指標」を作る道は行き詰まった。
41通り試して最良66.7%(多数派54.2%)では、報酬にしても回路が何を学ぶか制御できない。
そこで原理から報酬を定義し、結果の良し悪しは人間の判定で見ることにした。

Phase 3 で確実に分かった構造をそのまま形にしている:
  × には2種類ある — 轍型(流暢だが跳んでいない)と崩壊型(跳んだが読めない)
  だから報酬は両側を持つ。跳躍は頭打ちにし、それ以上遠ざかっても得をさせない。

1回目の学習の結果を受けて、目盛りを3つ直した:
  跳躍の天井が低すぎ、第1世代から飽和して勾配を出していなかった
  尤度の床が甘すぎ、壊れた出力(-2.71)を一度も罰していなかった
  抑圧の帯が事実上の主動力になっていた。事前知識にすぎないので弱める

さらに崩壊を2段で見るようにした。平均だけでは
「ものだい」「だろうい」のような局所の破れが埋もれる。
対比較でも、途中切断された出力は勝ち、破格のある出力だけが負けた。
"""

import numpy as np
import torch

from centurion_score import persona_break, rut_rate, unterminated

# ===== 報酬の形 =====
# 跳躍の頭打ち。1回目は0.13で第1世代から飽和していたので上げた
JUMP_CEILING = 0.18

# 全体の対数尤度がこれを割ったら崩壊とみなす。
# 24件の実測で床ごとの発火率を調べた結果:
#   -2.8 は判別が最も良い(当たりの15%・外れの45%を罰する)が、
#   実際に壊れた出力が-2.71で、すぐ上をすり抜けていた
#   -2.2 は当たりの54%を罰してしまい、良い出力まで潰す
# 拾えることを優先して -2.6 にする(当たりの31%・外れの55%)
FLUENCY_FLOOR = -2.6

# 局所の破れ。下位25%がこれを割ったら、文のどこかが壊れている。
# 平均は保っているのに一部だけ破綻している出力を拾う。
# 実測では当たりの15%・外れの36%に発火し、誤爆が最も少なかった
LOCAL_PERCENTILE = 25
LOCAL_FLOOR = -4.5

# 抑圧強度の当たりを付ける。実測で最も当たり率が高かった帯
STRENGTH_TARGET = 2.1

W_JUMP = 1.0          # 轍から出た分の報酬 (0〜1に正規化済み)
W_BREAK = 2.0         # 全体の崩壊。跳躍より重くして、壊すくらいなら跳ばせない
W_LOCAL = 1.5         # 局所の破れ。対比較で負けた唯一の原因がこれだった
W_RUT = 0.3           # 轍語彙 (100文字あたりの回数、実測で0〜4程度)
W_PERSONA = 0.5       # 助力の申し出やAIへの言及
W_UNTERMINATED = 0.15 # 途中切断。破格より軽い — 切断された出力も対比較で勝っている
W_BAND = 0.02         # 抑圧強度の帯。1回目で役目を終えたので大きく下げる


def token_logprobs(model, tokenizer, prefix, body):
    """素のモデルが本文の各トークンをどれだけ予測できたか"""
    if not body.strip():
        return None

    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full_ids = tokenizer(prefix + body, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        logits = model(full_ids).logits[0].float()

    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    picked = logprobs.gather(1, full_ids[0, 1:].unsqueeze(-1)).squeeze(-1)

    values = picked[prefix_ids.shape[1] - 1:]
    return values.cpu().numpy() if values.numel() else None


def jump_distance(text, center, embed_fn, split_fn):
    """轍の重心から、最も遠ざかった一文の距離。跳躍は一箇所で起きる"""
    parts = split_fn(text)
    if not parts:
        return 0.0
    return float((1.0 - (embed_fn(parts) @ center)).max())


def compute(text, center, strength_mean, model, tokenizer, prefix,
            embed_fn, split_fn, prefill=""):
    """1つの出力の報酬と、その内訳を返す"""
    body = text[len(prefill):] if prefill and text.startswith(prefill) else text

    jump = min(jump_distance(text, center, embed_fn, split_fn), JUMP_CEILING)
    jump /= JUMP_CEILING

    values = token_logprobs(model, tokenizer, prefix, body)
    if values is None:
        fluency, local = FLUENCY_FLOOR - 5.0, LOCAL_FLOOR - 5.0
    else:
        fluency = float(values.mean())
        local = float(np.percentile(values, LOCAL_PERCENTILE))

    breakdown = max(0.0, FLUENCY_FLOOR - fluency)
    local_break = max(0.0, LOCAL_FLOOR - local)

    rut = rut_rate(text)
    persona = persona_break(text)
    cut = unterminated(text)
    band = abs(strength_mean - STRENGTH_TARGET)

    total = (W_JUMP * jump
             - W_BREAK * breakdown
             - W_LOCAL * local_break
             - W_RUT * rut
             - W_PERSONA * persona
             - W_UNTERMINATED * cut
             - W_BAND * band)

    return total, {
        "跳躍": jump, "尤度": fluency, "崩壊": breakdown,
        "局所": local_break, "轍率": rut, "逸脱": persona,
        "切断": cut, "抑圧": strength_mean,
    }


def summarize(records):
    """内訳の平均。学習が何を伸ばして何を削ったかを見る"""
    if not records:
        return {}
    return {k: float(np.mean([r[k] for r in records])) for k in records[0]}

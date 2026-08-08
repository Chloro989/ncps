"""
センチュリオン: 意味的な跳躍の測定 (Phase 3b)
語の一致では概念の飛距離を測れない(判明2)ため、文埋め込みで測る。

考え方: 無制御で生成させた出力の集まりが「轍」の位置を示している。
その重心からどれだけ離れたかが跳躍。文章全体で平均すると跳躍が薄まるので、
文ごとに測って最大値も見る — 発想の飛躍は文章の一箇所で起きるため。

崩壊した文も重心から遠ざかるが、それは centurion_score.py 側で別に罰する。
ここでは距離だけを測り、良し悪しの判断は混ぜない。
"""

import re

import torch
from transformers import AutoModel, AutoTokenizer

# 多言語e5。日本語に強く、CPUでも回る大きさ
MODEL_NAME = "intfloat/multilingual-e5-small"
PREFIX = "query: "        # e5 はこの接頭辞を付ける前提で学習されている
MAX_LENGTH = 512

SENTENCE_SPLIT = re.compile(r"(?<=[。！？])")

_cache = {}


def load():
    """モデルは重いので一度だけ読む"""
    if not _cache:
        _cache["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_NAME)
        _cache["model"] = AutoModel.from_pretrained(MODEL_NAME).eval()
    return _cache["tokenizer"], _cache["model"]


def embed(texts):
    """文字列の集まりを、長さ1に揃えたベクトルに変換する"""
    tokenizer, model = load()
    batch = tokenizer([PREFIX + t for t in texts],
                      padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        out = model(**batch).last_hidden_state

    # パディングを除いた平均をとる
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, dim=-1)


def sentences(text):
    """句点で切って文に分ける。短すぎる断片は落とす"""
    parts = [s.strip() for s in SENTENCE_SPLIT.split(text)]
    return [s for s in parts if len(s) >= 10]


def centroid(texts):
    """轍の位置。無制御の出力群の重心"""
    return torch.nn.functional.normalize(embed(texts).mean(dim=0), dim=-1)


def distance(text, center):
    """重心からの距離を、文章全体と文ごとの両方で測る"""
    whole = 1.0 - (embed([text])[0] @ center).item()

    parts = sentences(text)
    if not parts:
        return whole, whole, whole

    dists = 1.0 - (embed(parts) @ center)
    return whole, dists.max().item(), dists.mean().item()


def measure_jump(texts, baseline_texts):
    """複数の出力について、跳躍の距離をまとめて測る"""
    center = centroid(baseline_texts)
    return {key: distance(text, center) for key, text in texts.items()}

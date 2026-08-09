"""
センチュリオン: 表現への介入 (Phase 7)

ロジットを操作する機構が2つとも落ちた。
抑圧は読みやすさを上げるが轍を増やし(相関+0.862)、
未来エントロピーは読みやすさを下げて轍に効かない(相関-0.085)。

共通点は「次の1トークンをどれにするか」しか変えられないこと。
轍が概念レベルにあるなら(判明2)、出力の直前では遅い。
そこで中間層の隠れ状態に介入する。概念が宿っている層が違う。

方向ベクトルの作り方には注意が要る。人手ラベルをそのまま使うと
お題の偏り(冬は1○7△)を拾ってしまい、「轍らしさ」ではなく
「朝・道具らしさ」を学ぶ。お題の中で対比を取ってから平均する。

標本は轍語彙率で自動的に二分して増やす。この指標は人間の○/△の
判定と一致することが確認できている(p=0.029)。
ただし指標で作って指標で測ると循環するので、最終判定は必ず盲検で行う。
"""

import re
from pathlib import Path

import numpy as np
import torch

from centurion_score import build_rut_vocab, parse_trace, rut_rate_vocab

RESULTS = Path(__file__).resolve().parent.parent / "results"

# 本文を集める先。書式が違うので個別に読む
TRACE_FILE = "centurion_trace.txt"
BLOCK_FILES = ["centurion_eval.txt", "centurion_eval_type5.txt",
               "centurion_eval_future.txt", "centurion_samples.txt"]
ALPHA_FILE = "centurion_alpha.txt"

PREFILL = "そうですね、"


def collect_texts(prompts):
    """手元の結果ファイルから、お題つきの本文を集める"""
    found = {prompt: [] for prompt in prompts}

    trace = RESULTS / TRACE_FILE
    if trace.exists():
        for prompt in prompts:
            found[prompt].extend(parse_trace(trace, prompt))

    # 標本ファイル形式: 区切り線 → お題 → 判定 → 本文
    for name in BLOCK_FILES:
        path = RESULTS / name
        if not path.exists():
            continue
        for block in re.split(r"^-{20,}$", path.read_text(encoding="utf-8"),
                              flags=re.MULTILINE):
            head = re.search(r"^お題: (.+)$", block, re.MULTILINE)
            if not head or head.group(1).strip() not in found:
                continue
            lines = block.splitlines()
            start = next((i for i, l in enumerate(lines)
                          if l.startswith("判定:")), None)
            if start is None:
                continue
            body = "".join(l.strip() for l in lines[start + 1:]
                           if l.strip() and not l.startswith("["))
            if body:
                found[head.group(1).strip()].append(body)

    # αの掃引ファイル形式: [お題 #n] → 本文 → (轍… 尤度…)
    path = RESULTS / ALPHA_FILE
    if path.exists():
        for prompt, body in re.findall(
                r"\[(.+?) #\d+\]\n(.+?)\n\(轍", path.read_text(encoding="utf-8"),
                re.DOTALL):
            if prompt in found:
                found[prompt].append(body.strip())

    return found


def hidden_at_layer(model, tokenizer, prefix, body, layer):
    """本文のトークン位置における、指定した層の隠れ状態の平均"""
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full = tokenizer(prefix + body, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model(full, output_hidden_states=True)
    states = output.hidden_states[layer][0]
    return states[prefix_ids.shape[1]:].float().mean(dim=0).cpu()


def build_vector(model, tokenizer, prompts, build_prefix, layer,
                 min_per_side=3):
    """轍に沈んでいる側から出ている側を引いた方向を返す。
    お題ごとに中央値で二分し、お題内の差を平均する —
    お題の成分が方向に混ざらないようにするため"""
    trace = RESULTS / TRACE_FILE
    vocabs = {p: build_rut_vocab(parse_trace(trace, p), p) for p in prompts}
    texts = collect_texts(prompts)

    directions, report = [], []
    for prompt in prompts:
        bodies = texts[prompt]
        if len(bodies) < min_per_side * 2:
            report.append((prompt, len(bodies), 0, 0))
            continue

        rates = np.array([rut_rate_vocab(PREFILL + b, vocabs[prompt])
                          for b in bodies])
        middle = np.median(rates)
        deep = [b for b, r in zip(bodies, rates) if r > middle]
        shallow = [b for b, r in zip(bodies, rates) if r <= middle]
        if len(deep) < min_per_side or len(shallow) < min_per_side:
            report.append((prompt, len(bodies), len(deep), len(shallow)))
            continue

        prefix = build_prefix(prompt)
        deep_mean = torch.stack([hidden_at_layer(model, tokenizer, prefix, b, layer)
                                 for b in deep]).mean(dim=0)
        shallow_mean = torch.stack([hidden_at_layer(model, tokenizer, prefix, b, layer)
                                    for b in shallow]).mean(dim=0)
        directions.append(deep_mean - shallow_mean)
        report.append((prompt, len(bodies), len(deep), len(shallow)))

    if not directions:
        raise SystemExit("方向ベクトルを作るだけの本文が集まらない")

    vector = torch.stack(directions).mean(dim=0)
    return vector / vector.norm(), report


class Steering:
    """指定した層の隠れ状態から、方向ベクトルを差し引く"""

    def __init__(self, model, layer, vector, strength):
        self.vector = vector.to(model.device)
        self.strength = strength
        self.handle = model.model.layers[layer - 1].register_forward_hook(self)

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        shifted = hidden - self.strength * self.vector.to(hidden.dtype)
        if isinstance(output, tuple):
            return (shifted,) + output[1:]
        return shifted

    def remove(self):
        self.handle.remove()

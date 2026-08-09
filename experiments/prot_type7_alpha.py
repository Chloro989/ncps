"""
センチュリオン: α の用量反応 (Phase 6)

α=1 で轍語彙が41%減った(p=0.045)。ただし比較を3つ行っているので
単独では弱い証拠でしかない。α を振って単調に減るなら、
機構が因果であることの強い証拠になり、同時に働かせどころも決まる。

人手のラベル付けは要らない。轍語彙率が人間の○/△の判定と
一致することが確認できた(p=0.029)ため、指標だけで測れる。

崩壊も同時に見る。α を上げれば「先が開く」候補を選びすぎて
文が壊れるはずで、その境目が知りたい。
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import centurion_future as future
from centurion_score import build_rut_vocab, find_data, parse_trace, rut_rate_vocab

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_alpha.txt"

ALPHAS = [0.0, 1.0, 2.0, 3.0, 5.0]
RUNS = 3
MAX_TOKENS = 150
PREFILL = "そうですね、"
TOP_N = 6
MIN_P = 0.05

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)

USER_PROMPTS = [
    "青色にまつわる話を聞かせて",
    "朝の匂いについて書いて",
    "忘れられた道具の話をして",
    "冬の終わりをどう感じる",
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    return tokenizer, model


def build_inputs(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) + PREFILL
    return tokenizer(prompt, return_tensors="pt").to("cuda"), prompt


def mean_logprob(model, tokenizer, prefix, body):
    """素のモデルが本文をどれだけ予測できたか。崩壊の目安"""
    if not body.strip():
        return -10.0
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full = tokenizer(prefix + body, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        logits = model(full).logits[0].float()
    picked = torch.log_softmax(logits[:-1], dim=-1).gather(
        1, full[0, 1:].unsqueeze(-1)).squeeze(-1)
    values = picked[prefix_ids.shape[1] - 1:]
    return float(values.mean()) if values.numel() else -10.0


def main():
    tokenizer, model = load_model()
    trace = find_data("centurion_trace.txt")
    vocabs = {p: build_rut_vocab(parse_trace(trace, p), p) for p in USER_PROMPTS}

    records = []
    for alpha in ALPHAS:
        began = time.time()
        for prompt in USER_PROMPTS:
            inputs, prefix = build_inputs(tokenizer, prompt)
            for index in range(RUNS):
                # 同じ乱数で始めれば、αの違いだけが出力を変える
                torch.manual_seed(1000 + index)
                tokens = future.generate(
                    model, tokenizer, inputs.input_ids, MAX_TOKENS,
                    greedy=False, top_n=TOP_N, alpha=alpha, min_p=MIN_P)
                body = tokenizer.decode(tokens, skip_special_tokens=True)
                text = PREFILL + body
                records.append({
                    "alpha": alpha, "prompt": prompt, "run": index,
                    "rut": rut_rate_vocab(text, vocabs[prompt]),
                    "logprob": mean_logprob(model, tokenizer, prefix, body),
                    "length": len(text),
                    "text": text,
                })
        print(f"α={alpha}: {time.time() - began:.0f}秒")

    print(f"\n{'α':>5}{'轍語彙率':>10}{'尤度':>10}{'文字数':>8}")
    for alpha in ALPHAS:
        group = [r for r in records if r["alpha"] == alpha]
        print(f"{alpha:5.1f}{np.mean([r['rut'] for r in group]):10.2f}"
              f"{np.mean([r['logprob'] for r in group]):10.2f}"
              f"{np.mean([r['length'] for r in group]):8.0f}")

    print(f"\n{'α':>5}" + "".join(f"{p[:6]:>10}" for p in USER_PROMPTS))
    for alpha in ALPHAS:
        cells = []
        for prompt in USER_PROMPTS:
            group = [r for r in records
                     if r["alpha"] == alpha and r["prompt"] == prompt]
            cells.append(f"{np.mean([r['rut'] for r in group]):.2f}")
        print(f"{alpha:5.1f}" + "".join(f"{c:>10}" for c in cells))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"αの用量反応 / 上位{TOP_N}候補を先読み / {RUNS}試行\n")
        f.write(f"轍語彙: お題ごとに無制御トレースから実測\n\n")
        for alpha in ALPHAS:
            group = [r for r in records if r["alpha"] == alpha]
            f.write("=" * 60 + "\n")
            f.write(f"α={alpha}  轍語彙率 {np.mean([r['rut'] for r in group]):.2f}"
                    f"  尤度 {np.mean([r['logprob'] for r in group]):.2f}\n")
            f.write("=" * 60 + "\n\n")
            for record in group:
                f.write(f"[{record['prompt']} #{record['run']}]\n")
                f.write(record["text"].strip() + "\n")
                f.write(f"(轍{record['rut']:.2f} 尤度{record['logprob']:.2f})\n\n")

    print(f"\n完了: {OUTPUT_FILE}")


main()

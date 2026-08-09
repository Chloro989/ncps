"""
センチュリオン: 表現への介入の用量反応 (Phase 7)

轍の方向ベクトルを中間層で差し引き、強さを振って効果を見る。
αの掃引と同じ設計 — 同じ乱数種で対にし、轍語彙率と尤度の両方を測る。
人手のラベル付けは不要。効くと分かってから盲検にかける。

強さは隠れ状態のノルムに対する割合で指定する。
絶対値で指定すると層によって意味が変わってしまう。
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from centurion_score import build_rut_vocab, find_data, parse_trace, rut_rate_vocab
from centurion_steer import Steering, build_vector, hidden_at_layer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_steer.txt"
VECTOR_FILE = "centurion_steer.pt"

LAYER = 24             # 全36層のうち2/3の深さ。概念が乗っているとされる帯

# 隠れ状態ノルムに対する割合(%)。
# 最初 0.5〜4% で組んだが、表現への介入は加えるベクトルのノルムが
# 残差ストリームの10〜100%程度でようやく振る舞いが変わるのが通例で、
# 4%では閾値以下だった可能性が高い。壊れるところまで振って境目を見る
STRENGTHS = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
RUNS = 3
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0

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


def build_prefix(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True) + PREFILL


def mean_logprob(model, tokenizer, prefix, body):
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
    print(f"層数 {model.config.num_hidden_layers} / 介入する層 {LAYER}")

    # 同じ層のベクトルが保存済みなら作り直さない。
    # 材料集めに212件の順伝播がかかるので、強さや層を変えて試すたびに
    # やり直すのは無駄になる
    saved = Path(VECTOR_FILE)
    if saved.exists():
        data = torch.load(saved, map_location="cpu", weights_only=False)
        if data.get("layer") == LAYER:
            vector, scale = data["vector"], data["scale"]
            print(f"保存済みの方向ベクトルを使う (層{LAYER})")
        else:
            saved = None
    else:
        saved = None

    if saved is None:
        vector, report = build_vector(
            model, tokenizer, USER_PROMPTS,
            lambda p: build_prefix(tokenizer, p), LAYER)
        print("\n方向ベクトルの材料")
        for prompt, total, deep, shallow in report:
            print(f"  {prompt[:14]:<16} 本文{total:4d}件"
                  f" → 轍が深い{deep} / 浅い{shallow}")

        # 隠れ状態の大きさを測り、強さをその割合で指定できるようにする
        sample = hidden_at_layer(model, tokenizer,
                                 build_prefix(tokenizer, USER_PROMPTS[0]),
                                 "青は静かな色です。", LAYER)
        scale = float(sample.norm())
        torch.save({"vector": vector, "layer": LAYER, "scale": scale},
                   VECTOR_FILE)

    print(f"隠れ状態のノルム {scale:.1f} / 方向ベクトルは単位長")

    trace = find_data("centurion_trace.txt")
    vocabs = {p: build_rut_vocab(parse_trace(trace, p), p) for p in USER_PROMPTS}

    records = []
    for percent in STRENGTHS:
        strength = scale * percent / 100.0
        began = time.time()
        hook = Steering(model, LAYER, vector, strength) if percent else None
        try:
            for prompt in USER_PROMPTS:
                prefix = build_prefix(tokenizer, prompt)
                inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
                for index in range(RUNS):
                    torch.manual_seed(1000 + index)
                    output = model.generate(
                        **inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
                        temperature=1.0, min_p=MIN_P, top_p=TOP_P)
                    body = tokenizer.decode(
                        output[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True)
                    text = PREFILL + body
                    records.append({
                        "percent": percent, "prompt": prompt, "run": index,
                        "rut": rut_rate_vocab(text, vocabs[prompt]),
                        "text": text, "body": body, "prefix": prefix,
                    })
        finally:
            if hook is not None:
                hook.remove()
        print(f"強さ{percent}%: {time.time() - began:.0f}秒")

    # 尤度は介入なしの素のモデルで測る。介入したまま測ると自分で自分を褒める
    for record in records:
        record["logprob"] = mean_logprob(model, tokenizer,
                                         record["prefix"], record["body"])

    print(f"\n{'強さ%':>6}{'轍語彙率':>10}{'標準偏差':>10}{'尤度':>9}")
    for percent in STRENGTHS:
        group = [r for r in records if r["percent"] == percent]
        rut = np.array([r["rut"] for r in group])
        print(f"{percent:6.1f}{rut.mean():10.2f}{rut.std():10.2f}"
              f"{np.mean([r['logprob'] for r in group]):9.2f}")

    base = {(r["prompt"], r["run"]): r["rut"]
            for r in records if r["percent"] == 0.0}
    print("\n同じ乱数種での対比較 (介入なしと比べて轍語彙が減った件数)")
    for percent in STRENGTHS[1:]:
        wins = sum(1 for r in records if r["percent"] == percent
                   and r["rut"] < base[(r["prompt"], r["run"])])
        losses = sum(1 for r in records if r["percent"] == percent
                     and r["rut"] > base[(r["prompt"], r["run"])])
        print(f"  {percent}%: {wins}勝 {losses}敗")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"表現への介入 / 層{LAYER} / {RUNS}試行\n")
        f.write("強さは隠れ状態のノルムに対する割合\n\n")
        for percent in STRENGTHS:
            group = [r for r in records if r["percent"] == percent]
            f.write("=" * 60 + "\n")
            f.write(f"強さ{percent}%  轍語彙率 "
                    f"{np.mean([r['rut'] for r in group]):.2f}"
                    f"  尤度 {np.mean([r['logprob'] for r in group]):.2f}\n")
            f.write("=" * 60 + "\n\n")
            for record in group:
                f.write(f"[{record['prompt']} #{record['run']}]\n")
                f.write(record["text"].strip() + "\n")
                f.write(f"(轍{record['rut']:.2f} 尤度{record['logprob']:.2f})\n\n")

    print(f"\n完了: {OUTPUT_FILE} と {VECTOR_FILE}")


main()

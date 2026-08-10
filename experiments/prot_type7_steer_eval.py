"""
センチュリオン: 表現への介入の盲検評価 (Phase 7)

用量反応が出た。強さと轍語彙率の相関は-0.419で、
4お題すべてで5%以降は単調に減る。10%なら尤度がほぼ無傷のまま轍が33%減る。

ただし方向ベクトルは轍語彙率で作っており、その指標で測れば下がって当然。
循環しているので、単独では「轍から出た」証拠にならない。
人間の判定でしか決着しない。

条件は4つ。type5固定 は「読める」で12戦12勝した既存の最良手で、
同一実行内の基準として置く。
"""

import random
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

from centurion_steer import Steering

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
VECTOR_FILE = Path("centurion_steer.pt")
OUTPUT_FILE = "centurion_eval.txt"
KEY_FILE = "centurion_eval_key.txt"

# 前回は4条件12件ずつで、介入20%の脱出率67%が制御なし42%を上回ったが
# p=0.207 で有意にならなかった。12件では原理的に届かない。
# 条件を2つに絞って1条件32件にする。差が本物なら p<0.05 に届く規模
RUNS = 8
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0
SHUFFLE_SEED = 20260811

# 介入10%は前回、制御なしより脱出率が低く崩壊も最多で、
# 用量反応として説明がつかなかった。20%だけを残す
STRENGTHS = [20.0]
INCLUDE_TYPE5 = False   # 既に「読める」で12戦12勝しており、今回の問いには不要

TYPE5_GATE = 3.5
TYPE5_TOP_K = 2
TYPE5_STRENGTH = 2.0

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


class Type5Suppressor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        if entropy.item() >= TYPE5_GATE:
            _, indices = scores[0].topk(TYPE5_TOP_K)
            scores[0, indices] -= TYPE5_STRENGTH
        return scores


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
    return tokenizer(prompt, return_tensors="pt").to("cuda")


def generate(model, tokenizer, inputs, processor=None):
    kwargs = dict(max_new_tokens=MAX_TOKENS, do_sample=True,
                  temperature=1.0, min_p=MIN_P, top_p=TOP_P)
    if processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    output = model.generate(**inputs, **kwargs)
    body = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    return PREFILL + body


def main():
    tokenizer, model = load_model()
    data = torch.load(VECTOR_FILE, map_location="cpu", weights_only=False)
    vector, layer, scale = data["vector"], data["layer"], data["scale"]
    print(f"方向ベクトル: 層{layer} / ノルム基準 {scale:.1f}")

    conditions = ["制御なし"] + [f"介入{int(s)}%" for s in STRENGTHS]
    if INCLUDE_TYPE5:
        conditions.append("type5固定")
    print(f"条件: {' / '.join(conditions)}"
          f" / 各条件 {len(USER_PROMPTS) * RUNS}件")

    samples = []
    for name in conditions:
        hook = None
        if name.startswith("介入"):
            percent = float(name[2:-1])
            hook = Steering(model, layer, vector, scale * percent / 100.0)
        try:
            for prompt in USER_PROMPTS:
                inputs = build_inputs(tokenizer, prompt)
                for index in range(RUNS):
                    print(f"{name} / {prompt} / {index + 1}")
                    processor = (Type5Suppressor()
                                 if name == "type5固定" else None)
                    samples.append((name, prompt,
                                    generate(model, tokenizer, inputs, processor)))
        finally:
            if hook is not None:
                hook.remove()

    random.Random(SHUFFLE_SEED).shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 盲検評価\n")
        out.write(f"条件(順不同): {' / '.join(conditions)}\n")
        out.write(f"試行数: {RUNS}\n")
        # 種も書いておく。答え合わせ側と手で同期させていると必ずずれる
        out.write(f"乱数種: {SHUFFLE_SEED}\n")
        out.write("どの出力がどの条件かは伏せてあります。\n")
        out.write("判定は2段階です。\n")
        out.write("  まず崩壊しているものを × にしてください。\n")
        out.write("  残ったものの中で、轍(宇宙・神秘・深淵・星など、\n")
        out.write("  このモデルが放っておくと必ず書くもの)から\n")
        out.write("  出ているものを ○、沈んでいるものを △ にしてください。\n\n")
        key.write("盲検の答え(判定を終えるまで開かないこと)\n\n")

        for index, (name, prompt, text) in enumerate(samples, 1):
            out.write("-" * 60 + "\n")
            out.write(f"標本{index:02d}\n")
            out.write(f"お題: {prompt}\n")
            out.write("判定: \n")
            out.write(text.strip() + "\n\n")
            key.write(f"標本{index:02d}: {name}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(samples)}件) と {KEY_FILE}")
    print("判定を終えるまで答えのファイルは開かないこと")


main()

"""
センチュリオン: 未来エントロピーの盲検評価 (Phase 6)

抑圧は轍から出られなかった(相関+0.862で逆効果)。機構を入れ替えて確かめる。

条件は3つ。交絡を避けるため、先読みの経路そのものを対照に入れている —
未来エントロピーは上位6候補に絞るので、その絞り込み自体が出力を変える。
α=0 で同じ経路を通せば、差は α の効果だけになる。

  先読みなし   future.generate(alpha=0)  上位6絞りのみ。α の効果を分離する対照
  未来α=1     future.generate(alpha=1)  記事の式そのもの
  type5固定   これまでの最良手。盲検で12戦12勝している

判定は盲検。答えは別ファイルに分ける。
"""

import random
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

import centurion_future as future

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_eval.txt"
KEY_FILE = "centurion_eval_key.txt"

RUNS = 3
MAX_TOKENS = 150
PREFILL = "そうですね、"
TOP_N = 6
ALPHA = 1.0
MIN_P = 0.05
TOP_P = 1.0
SHUFFLE_SEED = 20260809

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
    """エントロピーが門を越えた箇所だけ上位2個を押し下げる"""

    def __init__(self):
        self.fired = 0
        self.steps = 0

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        self.steps += 1
        if entropy.item() >= TYPE5_GATE:
            _, indices = scores[0].topk(TYPE5_TOP_K)
            scores[0, indices] -= TYPE5_STRENGTH
            self.fired += 1
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


def run_type5(model, tokenizer, inputs):
    processor = Type5Suppressor()
    output = model.generate(
        **inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
        temperature=1.0, min_p=MIN_P, top_p=TOP_P,
        logits_processor=LogitsProcessorList([processor]))
    body = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    rate = processor.fired / max(processor.steps, 1) * 100
    return PREFILL + body, f"門の発火 {rate:.0f}%"


def run_future(model, tokenizer, inputs, alpha):
    tokens = future.generate(
        model, tokenizer, inputs.input_ids, MAX_TOKENS,
        greedy=False, top_n=TOP_N, alpha=alpha, min_p=MIN_P)
    body = tokenizer.decode(tokens, skip_special_tokens=True)
    return PREFILL + body, f"α={alpha}"


def main():
    tokenizer, model = load_model()
    conditions = {
        "先読みなし": lambda i: run_future(model, tokenizer, i, 0.0),
        "未来エントロピー": lambda i: run_future(model, tokenizer, i, ALPHA),
        "type5固定": lambda i: run_type5(model, tokenizer, i),
    }
    print(f"条件: {' / '.join(conditions)}")

    samples = []
    for name, run in conditions.items():
        began = time.time()
        for prompt in USER_PROMPTS:
            inputs = build_inputs(tokenizer, prompt)
            for index in range(1, RUNS + 1):
                print(f"{name} / {prompt} / {index}")
                text, note = run(inputs)
                samples.append((name, prompt, text, note))
        print(f"  {name}: {time.time() - began:.0f}秒")

    random.Random(SHUFFLE_SEED).shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 盲検評価\n")
        out.write(f"条件(順不同): {' / '.join(conditions)}\n")
        out.write(f"試行数: {RUNS}\n")
        out.write("どの出力がどの条件かは伏せてあります。\n")
        out.write("判定は2段階です。\n")
        out.write("  まず崩壊しているものを × にしてください。\n")
        out.write("  残ったものの中で、轍(宇宙・神秘・深淵・星・山脈など、\n")
        out.write("  このモデルが放っておくと必ず書くもの)から\n")
        out.write("  出ているものを ○、沈んでいるものを △ にしてください。\n\n")
        key.write("盲検の答え(判定を終えるまで開かないこと)\n\n")

        for index, (name, prompt, text, note) in enumerate(samples, 1):
            out.write("-" * 60 + "\n")
            out.write(f"標本{index:02d}\n")
            out.write(f"お題: {prompt}\n")
            out.write("判定: \n")
            out.write(text.strip() + "\n\n")
            key.write(f"標本{index:02d}: {name} / {note}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(samples)}件) と {KEY_FILE}")
    print("判定を終えるまで答えのファイルは開かないこと")


main()

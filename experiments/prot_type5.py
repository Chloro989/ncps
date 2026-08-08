"""
センチュリオン: 分岐点での上位抑圧
意味を選んでいる瞬間だけ、最有力の候補を押し下げて別の道へ逸らす
"""

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_divert.txt"

RUNS = 5
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0

# 抑圧のパラメータ
ENTROPY_GATE = 3.5      # 2.5 → 3.5
SUPPRESS_TOP_K = 2      # 3 → 2
SUPPRESS_STRENGTH = 2.0 # 3.0 → 2.0

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)
USER_PROMPT = "青色にまつわる話を聞かせて"


class BranchDiverter(LogitsProcessor):
    """分岐点でのみ、上位候補のスコアを下げて別の選択を促す"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.diverted = []   # 抑圧した語の記録(観察用)

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # 分岐点でなければ何もしない
        if entropy.item() < ENTROPY_GATE:
            return scores

        # 上位K個を押し下げる
        _, top_indices = scores[0].topk(SUPPRESS_TOP_K)
        self.diverted.append(
            "".join(self.tokenizer.decode([i]) for i in top_indices)
        )
        scores[0, top_indices] -= SUPPRESS_STRENGTH
        return scores


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


def build_inputs(tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += PREFILL
    return tokenizer(prompt, return_tensors="pt").to("cuda")


def generate(model, tokenizer, inputs, temperature, processor=None):
    kwargs = dict(
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=temperature,
        min_p=MIN_P,
        top_p=TOP_P,
    )
    if processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])

    output = model.generate(**inputs, **kwargs)
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    body = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return PREFILL + body


def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"お題: {USER_PROMPT}\n")
        f.write(f"モデル: {MODEL_NAME}\n")
        f.write(f"抑圧: 上位{SUPPRESS_TOP_K}個を -{SUPPRESS_STRENGTH} "
                f"(エントロピー {ENTROPY_GATE} 以上の箇所)\n\n")

        f.write("=" * 60 + "\n抑圧あり(センチュリオン)\n" + "=" * 60 + "\n\n")
        for run in range(1, RUNS + 1):
            print(f"抑圧あり {run}/{RUNS}")
            processor = BranchDiverter(tokenizer)
            text = generate(model, tokenizer, inputs, 1.0, processor)

            f.write(f"--- 試行 {run} ---\n")
            f.write(text.strip() + "\n")
            f.write(f"[分岐点 {len(processor.diverted)}箇所で抑圧]\n")
            f.write(f"[避けた候補: {' / '.join(processor.diverted[:8])}]\n\n")

        f.write("=" * 60 + "\n比較: 抑圧なし 温度1.0\n" + "=" * 60 + "\n\n")
        for run in range(1, RUNS + 1):
            print(f"抑圧なし {run}/{RUNS}")
            text = generate(model, tokenizer, inputs, 1.0)
            f.write(f"--- 試行 {run} ---\n")
            f.write(text.strip() + "\n\n")

    print(f"完了: {OUTPUT_FILE} に保存しました")


main()
"""
センチュリオン: エントロピー連動の適応温度
分布が「開いている」瞬間だけ温度を上げ、文法が決まっている箇所には触らない
"""

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_adaptive.txt"

RUNS = 5              # 各条件での試行回数
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0

# 適応温度のパラメータ
TEMP_LOW  = 1.0       # 分岐点でない箇所(文法が決まっている)の温度
TEMP_HIGH = 1.8       # 分岐点(意味を選んでいる)の温度
ENTROPY_LOW  = 1.5    # これ以下なら「閉じている」とみなす
ENTROPY_HIGH = 3.5    # これ以上なら「開いている」とみなす

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)
USER_PROMPT = "青色にまつわる話を聞かせて"


class AdaptiveTemperature(LogitsProcessor):
    """分布の散らばり具合を測り、温度を動的に決める"""

    def __init__(self):
        self.log = []   # 後で観察するための記録

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # エントロピーを 0〜1 に正規化し、その割合で温度を決める
        ratio = (entropy - ENTROPY_LOW) / (ENTROPY_HIGH - ENTROPY_LOW)
        ratio = ratio.clamp(0.0, 1.0)
        temp = TEMP_LOW + (TEMP_HIGH - TEMP_LOW) * ratio

        self.log.append((entropy.item(), temp.item()))
        return scores / temp.unsqueeze(-1)


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


def generate(model, tokenizer, inputs, processor=None, temperature=1.0):
    """processorを渡せば適応温度、渡さなければ固定温度で生成する"""
    kwargs = dict(
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        min_p=MIN_P,
        top_p=TOP_P,
    )
    if processor is not None:
        # 温度はprocessor側で掛けるので、ここでは等倍にしておく
        kwargs["temperature"] = 1.0
        kwargs["logits_processor"] = LogitsProcessorList([processor])
    else:
        kwargs["temperature"] = temperature

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
        f.write(f"適応温度: {TEMP_LOW}〜{TEMP_HIGH}"
                f" (エントロピー {ENTROPY_LOW}〜{ENTROPY_HIGH})\n\n")

        # --- 適応温度版 ---
        f.write("=" * 60 + "\n適応温度(センチュリオン)\n" + "=" * 60 + "\n\n")
        for run in range(1, RUNS + 1):
            print(f"適応温度 {run}/{RUNS}")
            processor = AdaptiveTemperature()
            text = generate(model, tokenizer, inputs, processor=processor)

            temps = [t for _, t in processor.log]
            f.write(f"--- 試行 {run} ---\n")
            f.write(text.strip() + "\n")
            f.write(f"[温度: 最小{min(temps):.2f} 最大{max(temps):.2f} "
                    f"平均{sum(temps)/len(temps):.2f}]\n\n")

        # --- 比較用: 固定温度 ---
        for fixed in (1.0, 1.8):
            f.write("=" * 60 + f"\n固定温度 {fixed}\n" + "=" * 60 + "\n\n")
            for run in range(1, RUNS + 1):
                print(f"固定温度 {fixed} {run}/{RUNS}")
                text = generate(model, tokenizer, inputs, temperature=fixed)
                f.write(f"--- 試行 {run} ---\n")
                f.write(text.strip() + "\n\n")

    print(f"完了: {OUTPUT_FILE} に保存しました")


main()
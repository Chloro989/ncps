"""
センチュリオン: 禁止語テスト
「轍」の正体を確かめるための計測用コード(使い捨て)
温度1.2と1.8のみ、禁止語ありで生成して比較する
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_banned.txt"

TEMPERATURES = [1.2, 1.8]   # 検証する温度
RUNS_PER_TEMP = 5           # 各温度での試行回数
MAX_TOKENS = 150            # 1回あたりの生成長
PREFILL = "そうですね、"      # 書き出し
MIN_P = 0.05                # 低確率トークンの足切り
TOP_P = 1.0                 # min_pを効かせるため無効化

# 前回の結果で頻出した「轍」の語彙
BANNED_WORDS = ["宇宙", "神秘", "深淵", "無限", "静寂", "星"]

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "ただし、人格は自由で規定しない。"
    "ユーザーの口調に寄せて、親近感を出す場合もある。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "「お手伝いします」「情報を提供します」といった助力の申し出をしない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)
USER_PROMPT = "青色にまつわる話を聞かせて"


# ===== モデルの準備 =====
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


def build_inputs(tokenizer):
    """会話形式のプロンプトを組み立ててトークン化する"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += PREFILL
    return tokenizer(prompt, return_tensors="pt").to("cuda")


def build_bad_words_ids(tokenizer):
    """禁止語をトークンIDの列に変換する"""
    return [tokenizer(w, add_special_tokens=False).input_ids
            for w in BANNED_WORDS]


def generate(model, tokenizer, inputs, temperature, bad_words_ids):
    """指定した温度で1回生成し、本文だけを返す"""
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=temperature,
        min_p=MIN_P,
        top_p=TOP_P,
        bad_words_ids=bad_words_ids,
    )
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    body = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return PREFILL + body


# ===== 実行 =====
def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer)
    bad_words_ids = build_bad_words_ids(tokenizer)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"お題: {USER_PROMPT}\n")
        f.write(f"モデル: {MODEL_NAME}\n")
        f.write(f"禁止語: {', '.join(BANNED_WORDS)}\n\n")

        for temp in TEMPERATURES:
            print(f"生成中... temperature = {temp}")
            f.write("=" * 60 + "\n")
            f.write(f"temperature = {temp}\n")
            f.write("=" * 60 + "\n\n")

            for run in range(1, RUNS_PER_TEMP + 1):
                text = generate(model, tokenizer, inputs, temp, bad_words_ids)
                f.write(f"--- 試行 {run} ---\n")
                f.write(text.strip() + "\n\n")

    print(f"完了: {OUTPUT_FILE} に保存しました")


main()
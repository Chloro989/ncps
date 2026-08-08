"""
センチュリオン: 隠れ状態の観察
生成中の中間表現を取り出し、その形と層ごとの性質を調べる
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "説明口調を避け、断定せず、余韻を残して書く。"
)
USER_PROMPT = "青色にまつわる話を聞かせて"
PREFILL = "そうですね、青色はあ"


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


def inspect_hidden_states(model, tokenizer, inputs):
    """1トークン分だけ順伝播し、各層の中間表現を調べる"""
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states   # (層数+1) 個のテンソル
    print(f"層の数: {len(hidden)}  (埋め込み層 + 各Transformer層)")
    print(f"1層あたりの形: {hidden[0].shape}  (バッチ, トークン数, 次元)")
    print()

    # 最後のトークン位置(=次に何を言うかを決める場所)の表現を層ごとに見る
    print("層  ノルム(ベクトルの大きさ)")
    for i, h in enumerate(hidden):
        last = h[0, -1, :]           # 最終トークンの表現
        print(f"{i:3d}  {last.norm().item():8.2f}")


def show_next_token_candidates(model, tokenizer, inputs, top_k=10):
    """今の文脈で、次に来る候補語を確認する"""
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = probs.topk(top_k)

    print(f"\n『{PREFILL}』の次に来る候補:")
    for p, idx in zip(values, indices):
        token = tokenizer.decode([idx])
        print(f"  {p.item()*100:5.1f}%  {token!r}")


def main():
    tokenizer, model = load_model()
    inputs = build_inputs(tokenizer)
    inspect_hidden_states(model, tokenizer, inputs)
    show_next_token_candidates(model, tokenizer, inputs)


main()
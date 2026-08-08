"""
センチュリオン: ラベル付け用の標本づくり (Phase 3の検証データ)
V2回路をランダム初期値で複数走らせ、抑圧の効き方が異なる出力を集める。
未学習なので当たりも外れも混ざる。それが狙いで、
跳躍指標が人間の判定を再現できるかを確かめるための標本にする。

Colabで実行し、centurion_samples.txt を持ち帰って判定欄を埋める。
"""

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

from centurion_circuit import (
    CenturionCircuitV2, apply_suppression, load_stats,
)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_samples.txt"

CIRCUITS = 6          # 初期値の異なる回路を何個試すか

# 未学習の回路は抑圧が 0.3〜0.4 程度と弱く、放っておくと全部が
# ベースライン寄りの出力になってラベルを付ける意味がなくなる。
# 回路ごとに出力のバイアスをずらし、弱い抑圧から強い抑圧まで幅を持たせる
STRENGTH_BIAS = [-1.0, -0.3, 0.4, 1.1, 1.8, 2.5]
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

# Phase 1 と同じお題。無制御の対照が既に揃っている
USER_PROMPTS = [
    "青色にまつわる話を聞かせて",
    "朝の匂いについて書いて",
    "忘れられた道具の話をして",
    "冬の終わりをどう感じる",
]


class NcpsSuppressor(LogitsProcessor):
    """回路の出力に従って上位候補を抑圧する。round による量子化はしない"""

    def __init__(self, circuit):
        self.circuit = circuit
        self.history = []

    def extract_features(self, probs, device):
        """回路に渡す4つの数値。標準化は回路側が持っている"""
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        step = min(len(self.history) / MAX_TOKENS, 1.0)
        return torch.tensor(
            [entropy.item(), probs.max().item(),
             probs.topk(5).values.sum().item(), step],
            device=device, dtype=torch.float32,
        )

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        features = self.extract_features(probs, scores.device)

        with torch.no_grad():
            control = self.circuit(features)

        scores, strength, width = apply_suppression(scores, control)
        self.history.append((strength.item(), width.item()))
        return scores


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda")
    return tokenizer, model


def build_inputs(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt += PREFILL
    return tokenizer(prompt, return_tensors="pt").to("cuda")


def generate(model, tokenizer, inputs, processor):
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=1.0,
        min_p=MIN_P,
        top_p=TOP_P,
        logits_processor=LogitsProcessorList([processor]),
    )
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    body = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return PREFILL + body


def main():
    mean, std, measured = load_stats()
    if not measured:
        print("警告: centurion_trace.npz が無いので暫定統計で動いている")

    tokenizer, model = load_model()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("センチュリオン: ラベル付け用の標本\n")
        f.write(f"回路: V2 未学習 × {CIRCUITS}個 / お題 {len(USER_PROMPTS)}件\n")
        f.write("各出力の「判定:」に ○(跳躍が成功) か ×(失敗) を書いてください。\n")
        f.write("どちらとも言えないものは空欄のままで構いません。\n\n")

        for seed in range(CIRCUITS):
            torch.manual_seed(seed)
            circuit = CenturionCircuitV2((mean, std)).to("cuda")
            # ゲートと強度をずらして、この回路の抑圧の効き方を決める
            offset = STRENGTH_BIAS[seed]
            with torch.no_grad():
                circuit.bias[0] += offset * 0.5
                circuit.bias[1] += offset
            circuit.settle()

            for prompt_id, user_prompt in enumerate(USER_PROMPTS):
                print(f"回路{seed} / お題{prompt_id + 1}")
                inputs = build_inputs(tokenizer, user_prompt)
                circuit.reset()
                processor = NcpsSuppressor(circuit)
                text = generate(model, tokenizer, inputs, processor)

                strengths = [s for s, _ in processor.history]
                widths = [w for _, w in processor.history]
                f.write("-" * 60 + "\n")
                f.write(f"標本: 回路{seed}-お題{prompt_id}"
                        f" (バイアス {STRENGTH_BIAS[seed]:+.1f})\n")
                f.write(f"お題: {user_prompt}\n")
                f.write("判定: \n")
                f.write(text.strip() + "\n")
                f.write(f"[抑圧強度: 平均{sum(strengths) / len(strengths):.2f}"
                        f" 最大{max(strengths):.2f}]\n")
                f.write(f"[実効幅: 平均{sum(widths) / len(widths):.2f}]\n\n")

    print(f"完了: {OUTPUT_FILE}"
          f" ({CIRCUITS * len(USER_PROMPTS)}件)")


main()

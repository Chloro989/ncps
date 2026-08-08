"""
センチュリオン: ncps回路の組み込み(第一段階)
線虫由来のニューロン回路を抑圧制御に接続する。
この段階では未学習(ランダム初期値)。壊れずに動くことの確認が目的。
"""

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_ncps.txt"

RUNS = 5
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0

# 回路の構成
NEURON_COUNT = 16      # 回路全体のニューロン数
OUTPUT_COUNT = 2       # 出力の数(抑圧強度と抑圧幅)
FEATURE_COUNT = 4      # 回路に入力する特徴の数

# 抑圧の可動域(回路の出力をこの範囲に写像する)
STRENGTH_MAX = 4.0     # 抑圧の最大値
TOP_K_MAX = 4          # 抑圧する候補数の上限

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)
USER_PROMPT = "青色にまつわる話を聞かせて"


# ===== ニューロン回路 =====
class CenturionCircuit(nn.Module):
    """線虫の神経回路構造を模した、抑圧を制御する小さな回路"""

    def __init__(self):
        super().__init__()
        wiring = AutoNCP(NEURON_COUNT, OUTPUT_COUNT)
        self.rnn = CfC(FEATURE_COUNT, wiring, batch_first=True)
        self.state = None

    def reset(self):
        """1回の生成を始める前に、回路の状態を初期化する"""
        self.state = None

    def forward(self, features):
        """特徴を1ステップ入力し、制御値を0〜1で返す"""
        x = features.view(1, 1, -1)          # (バッチ, 時間, 特徴)
        out, self.state = self.rnn(x, self.state)
        return torch.sigmoid(out.view(-1))   # 0〜1に収める


# ===== 生成への接続 =====
class NcpsSuppressor(LogitsProcessor):
    """回路の出力に従って、上位候補を動的に抑圧する"""

    def __init__(self, circuit, tokenizer):
        self.circuit = circuit
        self.tokenizer = tokenizer
        self.history = []

    def extract_features(self, probs, scores):
        """回路に渡す、今の状況を表す4つの数値"""
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        top1 = probs.max()
        top5 = probs.topk(5).values.sum()
        step = min(len(self.history) / MAX_TOKENS, 1.0)   # 生成の進み具合
        return torch.tensor(
            [entropy.item() / 5.0, top1.item(), top5.item(), step],
            device=scores.device, dtype=torch.float32,
        )

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        features = self.extract_features(probs, scores)

        with torch.no_grad():
            control = self.circuit(features)

        strength = control[0].item() * STRENGTH_MAX
        top_k = max(1, int(round(control[1].item() * TOP_K_MAX)))

        _, top_indices = scores[0].topk(top_k)
        scores[0, top_indices] -= strength

        self.history.append((strength, top_k))
        return scores


# ===== モデルの準備 =====
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


def generate(model, tokenizer, inputs, processor=None):
    kwargs = dict(
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=1.0,
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

    circuit = CenturionCircuit().to("cuda")
    print(f"回路: {NEURON_COUNT}ニューロン, "
          f"パラメータ数 {sum(p.numel() for p in circuit.parameters())}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"お題: {USER_PROMPT}\n")
        f.write(f"回路: AutoNCP({NEURON_COUNT}, {OUTPUT_COUNT}) 未学習\n\n")

        f.write("=" * 60 + "\nncps制御(未学習)\n" + "=" * 60 + "\n\n")
        for run in range(1, RUNS + 1):
            print(f"ncps {run}/{RUNS}")
            circuit.reset()
            processor = NcpsSuppressor(circuit, tokenizer)
            text = generate(model, tokenizer, inputs, processor)

            strengths = [s for s, _ in processor.history]
            ks = [k for _, k in processor.history]
            f.write(f"--- 試行 {run} ---\n")
            f.write(text.strip() + "\n")
            f.write(f"[抑圧強度: 最小{min(strengths):.2f} "
                    f"最大{max(strengths):.2f} 平均{sum(strengths)/len(strengths):.2f}]\n")
            f.write(f"[抑圧幅: 平均{sum(ks)/len(ks):.1f}個]\n\n")

        f.write("=" * 60 + "\n比較: 制御なし\n" + "=" * 60 + "\n\n")
        for run in range(1, RUNS + 1):
            print(f"制御なし {run}/{RUNS}")
            text = generate(model, tokenizer, inputs)
            f.write(f"--- 試行 {run} ---\n")
            f.write(text.strip() + "\n\n")

    print(f"完了: {OUTPUT_FILE} に保存しました")
    
main()
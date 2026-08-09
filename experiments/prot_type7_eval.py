"""
センチュリオン: 学習済み回路の評価 (Phase 5)
学習は貪欲法で行った。報酬が回路の決定的な関数になり初めて勾配が読めたが、
貪欲法の文は壊れにくく、崩壊の罰が40世代を通じて一度も働かなかった。
つまり回路は「壊れようがない環境」で抑圧を2.4まで上げることを学んでいる。
実運用のサンプリングで壊れないという保証はない。ここで確かめる。

条件は3つ:
  学習済み — centurion_circuit.pt を読んだV2回路
  未学習   — 同じ構成のランダム初期値
  制御なし — 抑圧しない素のモデル

判定は盲検で行う。これまで、こちらが先にスコアを見せたり
条件を明かしたりしたことで判定が偏る危険を何度も踏んだ。
出力は条件を伏せてシャッフルして書き出し、答えは別ファイルに分ける。
"""

import random
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

from centurion_circuit import (
    CenturionCircuitV2, apply_suppression, load_stats, unpack,
)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT = Path("centurion_circuit.pt")
OUTPUT_FILE = "centurion_eval.txt"
KEY_FILE = "centurion_eval_key.txt"

RUNS = 3               # 条件・お題ごとの試行回数
MAX_TOKENS = 150
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0
SHUFFLE_SEED = 20260809

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


class NcpsSuppressor(LogitsProcessor):
    """回路の出力に従って上位候補を抑圧する"""

    def __init__(self, circuit):
        self.circuit = circuit
        self.history = []

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        features = torch.tensor(
            [(-(probs * torch.log(probs + 1e-9)).sum()).item(),
             probs.max().item(),
             probs.topk(5).values.sum().item(),
             min(len(self.history) / MAX_TOKENS, 1.0)],
            device=scores.device, dtype=torch.float32)

        with torch.no_grad():
            control = self.circuit(features)

        _, strength, width = apply_suppression(scores, control)
        self.history.append((strength.item(), width.item()))
        return scores

    def summary(self):
        if not self.history:
            return 0.0, 0.0
        strengths = [s for s, _ in self.history]
        widths = [w for _, w in self.history]
        return (sum(strengths) / len(strengths), sum(widths) / len(widths))


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
    """実運用と同じサンプリングで生成する。学習は貪欲法だったが評価はこちら"""
    kwargs = dict(max_new_tokens=MAX_TOKENS, do_sample=True,
                  temperature=1.0, min_p=MIN_P, top_p=TOP_P)
    if processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([processor])

    output = model.generate(**inputs, **kwargs)
    body = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    return PREFILL + body


def build_circuits(mean, std):
    """学習済みと未学習を用意する。制御なしは回路を使わない"""
    conditions = {}

    torch.manual_seed(0)
    untrained = CenturionCircuitV2((mean, std)).to("cuda")
    untrained.settle()
    conditions["未学習"] = untrained

    if CHECKPOINT.exists():
        saved = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
        trained = CenturionCircuitV2((saved["mean"], saved["std"])).to("cuda")
        unpack(trained, saved["theta"].to("cuda"))
        conditions["学習済み"] = trained
        print(f"学習済み回路: 世代{saved['generation']}")
    else:
        print(f"警告: {CHECKPOINT} が無いので学習済みを評価できない")

    conditions["制御なし"] = None
    return conditions


def main():
    mean, std, measured = load_stats()
    if not measured:
        print("警告: centurion_trace.npz が無いので暫定統計で動いている")

    tokenizer, model = load_model()
    conditions = build_circuits(mean, std)

    samples = []
    for name, circuit in conditions.items():
        for prompt in USER_PROMPTS:
            inputs = build_inputs(tokenizer, prompt)
            for run in range(1, RUNS + 1):
                print(f"{name} / {prompt} / {run}")
                if circuit is None:
                    text, note = generate(model, tokenizer, inputs), ""
                else:
                    circuit.reset()
                    processor = NcpsSuppressor(circuit)
                    text = generate(model, tokenizer, inputs, processor)
                    strength, width = processor.summary()
                    note = f"抑圧{strength:.2f} 幅{width:.2f}"
                samples.append((name, prompt, text, note))

    # 条件を伏せてシャッフルする。判定が条件に引きずられないように
    random.Random(SHUFFLE_SEED).shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 盲検評価\n")
        out.write("どの出力がどの条件かは伏せてあります。\n")
        out.write("各出力の「判定:」に ○(良い) か ×(悪い) を書いてください。\n")
        out.write("基準は前回と同じ — 崩壊しているものを弾き、"
                  "意味はわからなくても読めるものを残す。\n\n")
        key.write("盲検の答え(判定を終えるまで開かないこと)\n\n")

        for index, (name, prompt, text, note) in enumerate(samples, 1):
            out.write("-" * 60 + "\n")
            out.write(f"標本{index:02d}\n")
            out.write(f"お題: {prompt}\n")
            out.write("判定: \n")
            out.write(text.strip() + "\n\n")
            key.write(f"標本{index:02d}: {name}"
                      f"{' / ' + note if note else ''}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(samples)}件) と {KEY_FILE}")
    print("判定を終えるまで答えのファイルは開かないこと")


main()

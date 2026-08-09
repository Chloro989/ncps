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
    CenturionCircuitV2, CenturionCircuitV3, apply_suppression, apply_values,
    load_stats, unpack,
)

# type5 の設定。この一連で唯一「海賊の航路」を生んだもの。
# 効くという前提を盲検で確かめたことがないので、条件に入れる
TYPE5_GATE = 3.5
TYPE5_TOP_K = 2
TYPE5_STRENGTH = 2.0

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


class Type5Suppressor(LogitsProcessor):
    """type5 の固定規則。エントロピーが門を越えた箇所だけ上位2個を押し下げる"""

    def __init__(self):
        self.history = []

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores[0].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()

        if entropy.item() < TYPE5_GATE:
            self.history.append((0.0, 0.0))
            return scores

        _, indices = scores[0].topk(TYPE5_TOP_K)
        scores[0, indices] -= TYPE5_STRENGTH
        self.history.append((TYPE5_STRENGTH, float(TYPE5_TOP_K)))
        return scores

    def summary(self):
        strengths = [s for s, _ in self.history]
        fired = sum(1 for s in strengths if s > 0)
        return (sum(strengths) / len(strengths) if strengths else 0.0,
                fired / len(strengths) * 100 if strengths else 0.0)


class V3Suppressor(LogitsProcessor):
    """V3回路。門は構造として組み込まれており、回路は内側でのみ動く"""

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
            effective, width = self.circuit.control(features)

        apply_values(scores, effective, width)
        self.history.append((effective.item(), width.item()))
        return scores

    def summary(self):
        if not self.history:
            return 0.0, 0.0
        strengths = [s for s, _ in self.history]
        widths = [w for _, w in self.history]
        return sum(strengths) / len(strengths), sum(widths) / len(widths)


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


def make_processor_factories(mean, std):
    """条件ごとに、生成のたびに新しいプロセッサを作る関数を用意する"""
    factories = {}

    # type5 の固定規則。効くという前提の検証がこの条件の目的
    factories["type5固定"] = lambda: Type5Suppressor()

    # V3。門が構造として入っており、回路は内側でのみ動く
    torch.manual_seed(0)
    v3 = CenturionCircuitV3((mean, std)).to("cuda")
    v3.settle()

    def make_v3():
        v3.reset()
        return V3Suppressor(v3)

    factories["V3未学習"] = make_v3

    # 学習済みV3があれば入れる。無ければ飛ばす
    if CHECKPOINT.exists():
        saved = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
        if saved.get("version") == "V3":
            trained = CenturionCircuitV3(
                (saved["mean"], saved["std"])).to("cuda")
            unpack(trained, saved["theta"].to("cuda"))

            def make_trained():
                trained.reset()
                return V3Suppressor(trained)

            factories["V3学習済み"] = make_trained
            print(f"学習済みV3: 世代{saved['generation']}")
        else:
            print(f"{CHECKPOINT} はV3ではないので条件に入れない")

    factories["制御なし"] = lambda: None
    return factories


def main():
    mean, std, measured = load_stats()
    if not measured:
        print("警告: centurion_trace.npz が無いので暫定統計で動いている")

    tokenizer, model = load_model()
    factories = make_processor_factories(mean, std)
    print(f"条件: {' / '.join(factories)}")

    samples = []
    for name, factory in factories.items():
        for prompt in USER_PROMPTS:
            inputs = build_inputs(tokenizer, prompt)
            for run in range(1, RUNS + 1):
                print(f"{name} / {prompt} / {run}")
                processor = factory()
                text = generate(model, tokenizer, inputs, processor)
                if processor is None:
                    note = ""
                else:
                    first, second = processor.summary()
                    note = f"抑圧{first:.2f} 幅{second:.2f}"
                samples.append((name, prompt, text, note))

    # 条件を伏せてシャッフルする。判定が条件に引きずられないように
    random.Random(SHUFFLE_SEED).shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 盲検評価\n")
        # 条件名は書くが順序は伏せる。答え合わせの再現に必要で、
        # かつどれがどれかは分からないままにできる
        out.write(f"条件(順不同): {' / '.join(factories)}\n")
        out.write(f"試行数: {RUNS}\n")
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

"""
センチュリオン: 回路の学習 (Phase 4)
報酬が非微分(テキストの質)なので、勾配ではなく進化戦略で回路を動かす。
対称サンプリングで方向を推定する OpenAI-ES。回路は1211パラメータと小さい。

速度の要はバッチ化にある。母集団の各個体は別の回路を持つが、
LogitsProcessor が行ごとに別の回路を引けば、1回の generate で全個体を走らせられる。
個体ごとに generate を呼ぶより桁で速い。

Colabで実行する。centurion_trace.npz を同じ場所に置くこと。
"""

import time
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

import centurion_reward as reward
from centurion_circuit import (
    CenturionCircuitV2, apply_suppression, load_stats, pack, unpack,
)
from centurion_score import find_data
from centurion_embed import MODEL_NAME as EMBED_NAME, PREFIX, sentences

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT = "centurion_circuit.pt"
LOG_FILE = "centurion_train.txt"

# 1回目は20世代で17分だった。1世代50秒ほどなので、倍にしても40分程度
GENERATIONS = 40        # 世代数
POPULATION = 12         # 個体数。対称サンプリングなので偶数にすること
SIGMA = 0.05            # 探索の広がり
LEARNING_RATE = 0.03

MAX_TOKENS = 120        # 学習中は短めにして回数を稼ぐ
PREFILL = "そうですね、"
MIN_P = 0.05
TOP_P = 1.0

SAMPLE_EVERY = 5        # 何世代ごとに出力を書き出すか

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

# 轍の位置を測るための無制御出力。Phase 1 のトレースから読む。
# カレントに無ければリポジトリの results/ を見る
TRACE_TEXT = find_data("centurion_trace.txt")


# ===== 生成への接続 =====
class BatchedSuppressor(LogitsProcessor):
    """バッチの行ごとに別の回路を引いて抑圧する。母集団を一度に走らせるための要"""

    def __init__(self, circuits):
        self.circuits = circuits
        self.history = [[] for _ in circuits]
        self.step = 0

    def __call__(self, input_ids, scores):
        # 特徴はバッチ全体でまとめて出す。行ごとに .item() を呼ぶと
        # 1回の生成で数千回のGPU同期が入り、学習ループが目に見えて遅くなる
        probs = torch.softmax(scores.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        top1 = probs.max(dim=-1).values
        top5 = probs.topk(5, dim=-1).values.sum(dim=-1)
        step = torch.tensor(min(self.step / MAX_TOKENS, 1.0),
                            device=scores.device, dtype=torch.float32)

        for row, circuit in enumerate(self.circuits):
            features = torch.stack([entropy[row], top1[row], top5[row], step])
            with torch.no_grad():
                control = circuit(features)

            # apply_suppression は (1, 語彙) を前提にしている。
            # scores[row:row+1] はビューなので、書き戻しは要らない
            _, strength, _ = apply_suppression(scores[row:row + 1], control)
            self.history[row].append(strength.detach())

        self.step += 1
        return scores

    def mean_strength(self):
        """テンソルのまま溜めておき、ここで一度だけCPUに落とす"""
        return [torch.stack(h).mean().item() if h else 0.0
                for h in self.history]


# ===== モデル =====
def load_all():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"      # 生成をバッチでそろえるため
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16).to("cuda").eval()

    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_NAME).to("cuda").eval()
    return tokenizer, model, embed_tokenizer, embed_model


def make_embed_fn(embed_tokenizer, embed_model):
    """報酬から呼ぶ埋め込み関数。長さ1にそろえたベクトルを返す"""
    def embed(texts):
        batch = embed_tokenizer([PREFIX + t for t in texts], padding=True,
                                truncation=True, max_length=512,
                                return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = embed_model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)
    return embed


def build_prefix(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return prompt + PREFILL


def parse_trace_texts(path, prompt):
    """轍の重心を作るための無制御出力を読む"""
    from centurion_score import parse_trace
    return parse_trace(path, prompt)


def build_centers(embed_fn):
    """お題ごとに、無制御出力の重心を求める"""
    centers = {}
    for prompt in USER_PROMPTS:
        texts = parse_trace_texts(TRACE_TEXT, prompt)
        if not texts:
            raise SystemExit(f"対照が見つからない: {prompt}")
        vectors = embed_fn(texts)
        centers[prompt] = torch.nn.functional.normalize(
            vectors.mean(dim=0), dim=-1)
    return centers


# ===== 評価 =====
def generate_batch(model, tokenizer, prefix, circuits):
    """母集団の全個体を1回の generate で走らせる"""
    inputs = tokenizer([prefix] * len(circuits), return_tensors="pt").to("cuda")
    for circuit in circuits:
        circuit.reset()

    processor = BatchedSuppressor(circuits)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=1.0,
            min_p=MIN_P,
            top_p=TOP_P,
            logits_processor=LogitsProcessorList([processor]),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    start = inputs.input_ids.shape[1]
    texts = [PREFILL + tokenizer.decode(row[start:], skip_special_tokens=True)
             for row in output]
    return texts, processor.mean_strength()


def evaluate(circuits, models, centers, collect=None):
    """各個体の報酬を、全お題の平均で求める"""
    tokenizer, model, embed_fn = models
    totals = np.zeros(len(circuits))
    details = [[] for _ in circuits]

    for prompt in USER_PROMPTS:
        prefix = build_prefix(tokenizer, prompt)
        texts, strengths = generate_batch(model, tokenizer, prefix, circuits)

        for i, (text, strength) in enumerate(zip(texts, strengths)):
            value, parts = reward.compute(
                text, centers[prompt], strength, model, tokenizer, prefix,
                embed_fn, sentences, prefill=PREFILL)
            totals[i] += value / len(USER_PROMPTS)
            details[i].append(parts)
            if collect is not None and i == 0:
                collect.append((prompt, text, parts))

    return totals, details


# ===== 進化戦略 =====
def make_population(base, theta, noise, sigma):
    """theta ± sigma*noise の回路をつくる。対称サンプリング"""
    circuits = []
    for eps in noise:
        for sign in (1.0, -1.0):
            circuits.append(unpack(base(), theta + sign * sigma * eps))
    return circuits


def rank_normalize(values):
    """報酬の絶対値ではなく順位を使う。外れ値に引きずられないため"""
    order = values.argsort().argsort().astype(np.float64)
    return order / (len(values) - 1) - 0.5


def main():
    mean, std, measured = load_stats()
    if not measured:
        print("警告: centurion_trace.npz が無いので暫定統計で動いている")

    tokenizer, model, embed_tokenizer, embed_model = load_all()
    embed_fn = make_embed_fn(embed_tokenizer, embed_model)
    centers = build_centers(embed_fn)
    models = (tokenizer, model, embed_fn)

    def base():
        return CenturionCircuitV2((mean, std)).to("cuda")

    torch.manual_seed(0)
    start_circuit = base()
    start_circuit.settle()      # 平均入力の定常状態から始める
    theta = pack(start_circuit).clone()
    print(f"回路: 学習対象 {theta.numel()}パラメータ (配線マスクは除外)"
          f" / 母集団{POPULATION} / {GENERATIONS}世代")

    half = POPULATION // 2
    log = open(LOG_FILE, "w", encoding="utf-8")
    log.write(f"母集団{POPULATION} σ{SIGMA} 学習率{LEARNING_RATE}\n\n")

    for generation in range(1, GENERATIONS + 1):
        began = time.time()
        noise = [torch.randn_like(theta) for _ in range(half)]
        circuits = make_population(base, theta, noise, SIGMA)

        rewards, details = evaluate(circuits, models, centers)
        advantage = rank_normalize(rewards)

        # 対称な対ごとに差をとって方向を推定する
        step = torch.zeros_like(theta)
        for i, eps in enumerate(noise):
            step += (advantage[2 * i] - advantage[2 * i + 1]) * eps
        theta += LEARNING_RATE / (POPULATION * SIGMA) * step

        best = int(rewards.argmax())
        parts = reward.summarize([p for d in details for p in d])
        line = (f"世代{generation:3d}  報酬 平均{rewards.mean():+.3f}"
                f" 最良{rewards.max():+.3f}"
                f"  跳躍{parts['跳躍']:.2f} 崩壊{parts['崩壊']:.2f}"
                f" 局所{parts['局所']:.2f} 切断{parts['切断']:.2f}"
                f" 轍{parts['轍率']:.2f} 抑圧{parts['抑圧']:.2f}"
                f"  {time.time() - began:.0f}秒")
        print(line)
        log.write(line + "\n")
        log.flush()

        if generation % SAMPLE_EVERY == 0 or generation == GENERATIONS:
            circuit = unpack(base(), theta)
            collected = []
            evaluate([circuit], models, centers, collect=collected)
            log.write(f"\n--- 世代{generation} の出力 ---\n")
            for prompt, text, parts in collected:
                log.write(f"[{prompt}]\n{text.strip()}\n")
                log.write(f"(跳躍{parts['跳躍']:.2f} 尤度{parts['尤度']:.2f}"
                          f" 局所{parts['局所']:.2f} 抑圧{parts['抑圧']:.2f})\n\n")
            log.flush()
            torch.save({"theta": theta.cpu(), "mean": mean, "std": std,
                        "generation": generation}, CHECKPOINT)

    log.close()
    print(f"完了: {CHECKPOINT} と {LOG_FILE}")


main()

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
    CenturionCircuitV3, apply_values, load_stats, pack, unpack,
)
from centurion_score import find_data
from centurion_embed import MODEL_NAME as EMBED_NAME, PREFIX, sentences

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT = "centurion_circuit.pt"
LOG_FILE = "centurion_train.txt"

# ログがどの版で出たものかを、ログ自身に書かせる。
# Colabの復元で古いログが混ざったとき、中身だけでは見分けがつかなかった
VERSION = "V3 (エントロピーの門を構造として持つ) / 貪欲法での評価"

# 分岐点かどうかの基準。門が働いているかをログで見るためだけに使う
GATE_REFERENCE = 3.5

# 評価を貪欲法で行うか。
# サンプリングだと報酬が120トークンぶんのサイコロに支配され、
# 3回の学習でいずれも勾配が読み取れなかった。
# 同一乱数でもノイズは19%しか減らなかった —
# 抑圧で1トークン変われば、そこから先は別の文になるため。
# 貪欲法にすれば報酬は回路の決定的な関数になり、抑圧だけが変化の源になる。
# 実運用のサンプリングとはずれるので、学習後に必ず再評価すること
EVAL_GREEDY = True

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
        self.entropies = [[] for _ in circuits]
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
                effective, width = circuit.control(features)

            # scores[row:row+1] はビューなので、書き戻しは要らない
            apply_values(scores[row:row + 1], effective, width)
            self.history[row].append(effective.detach())
            self.entropies[row].append(entropy[row].detach())

        self.step += 1
        return scores

    def mean_strength(self):
        """テンソルのまま溜めておき、ここで一度だけCPUに落とす"""
        return [torch.stack(h).mean().item() if h else 0.0
                for h in self.history]

    def gate_ratio(self):
        """分岐点とそれ以外で、抑圧の強さがどれだけ違うか。
        1.0付近なら門が働いていない — V2がそうだった"""
        ratios = []
        for strengths, entropies in zip(self.history, self.entropies):
            if not strengths:
                ratios.append(1.0)
                continue
            value = torch.stack(strengths)
            level = torch.stack(entropies)
            branch = level >= GATE_REFERENCE
            if branch.any() and (~branch).any():
                high = value[branch].mean().item()
                low = value[~branch].mean().item()
                ratios.append(high / max(low, 1e-6))
            else:
                ratios.append(1.0)
        return ratios


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
def generate_batch(model, tokenizer, prefix, circuits, seed):
    """母集団の全個体を1回の generate で走らせる。
    seed を固定するのが要点 — 対称サンプリングの +ε 群と -ε 群を
    同じ種で走らせれば、対になる個体が同じ乱数列を引く。
    報酬の差から「サイコロの差」が消え、回路の効果だけが残る"""
    torch.manual_seed(seed)
    inputs = tokenizer([prefix] * len(circuits), return_tensors="pt").to("cuda")
    for circuit in circuits:
        circuit.reset()

    processor = BatchedSuppressor(circuits)
    kwargs = dict(
        max_new_tokens=MAX_TOKENS,
        logits_processor=LogitsProcessorList([processor]),
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if EVAL_GREEDY:
        # 貪欲法では抑圧だけが選択を変える。回路の効果が最も見えやすい
        kwargs["do_sample"] = False
    else:
        kwargs.update(do_sample=True, temperature=1.0,
                      min_p=MIN_P, top_p=TOP_P)

    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)

    start = inputs.input_ids.shape[1]
    texts = [PREFILL + tokenizer.decode(row[start:], skip_special_tokens=True)
             for row in output]
    return texts, processor.mean_strength(), processor.gate_ratio()


def evaluate(circuits, models, centers, seed_base, collect=None):
    """各個体の報酬を、全お題の平均で求める。
    seed_base が同じなら、別の呼び出しでも同じ乱数列で生成される"""
    tokenizer, model, embed_fn = models
    totals = np.zeros(len(circuits))
    details = [[] for _ in circuits]

    for index, prompt in enumerate(USER_PROMPTS):
        prefix = build_prefix(tokenizer, prompt)
        texts, strengths, ratios = generate_batch(
            model, tokenizer, prefix, circuits, seed_base * 100 + index)

        for i, (text, strength, ratio) in enumerate(
                zip(texts, strengths, ratios)):
            value, parts = reward.compute(
                text, centers[prompt], strength, model, tokenizer, prefix,
                embed_fn, sentences, prefill=PREFILL)
            parts["門比"] = ratio
            totals[i] += value / len(USER_PROMPTS)
            details[i].append(parts)
            if collect is not None and i == 0:
                collect.append((prompt, text, parts))

    return totals, details


# ===== 進化戦略 =====
def make_population(base, theta, noise, sigma, sign):
    """theta + sign*sigma*noise の回路をつくる。
    +ε 群と -ε 群を分けて作るのは、同じ乱数種で別々に走らせるため。
    対になる個体を同じ行位置に置くことで、乱数列がそろう"""
    return [unpack(base(), theta + sign * sigma * eps) for eps in noise]


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
        return CenturionCircuitV3((mean, std)).to("cuda")

    torch.manual_seed(0)
    start_circuit = base()
    start_circuit.settle()      # 平均入力の定常状態から始める
    theta = pack(start_circuit).clone()
    print(f"回路: 学習対象 {theta.numel()}パラメータ (配線マスクは除外)"
          f" / 母集団{POPULATION} / {GENERATIONS}世代")

    half = POPULATION // 2
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (f"版: {VERSION}\n"
              f"開始: {started}\n"
              f"評価: {'貪欲法' if EVAL_GREEDY else 'サンプリング'}\n"
              f"母集団{POPULATION} σ{SIGMA} 学習率{LEARNING_RATE}"
              f" 世代{GENERATIONS}\n"
              f"報酬: 天井{reward.JUMP_CEILING} 床{reward.FLUENCY_FLOOR}"
              f" 局所{reward.LOCAL_FLOOR} 帯の重み{reward.W_BAND}\n")
    print(header)
    log = open(LOG_FILE, "w", encoding="utf-8")
    log.write(header + "\n")

    for generation in range(1, GENERATIONS + 1):
        began = time.time()
        noise = [torch.randn_like(theta) for _ in range(half)]

        # 同じ種で2回走らせる。対になる個体が同じ乱数列を引く
        plus = make_population(base, theta, noise, SIGMA, +1.0)
        minus = make_population(base, theta, noise, SIGMA, -1.0)
        reward_plus, detail_plus = evaluate(plus, models, centers, generation)
        reward_minus, detail_minus = evaluate(minus, models, centers, generation)

        rewards = np.concatenate([reward_plus, reward_minus])
        advantage = rank_normalize(rewards)

        # 対ごとの差をとって方向を推定する
        step = torch.zeros_like(theta)
        for i, eps in enumerate(noise):
            step += (advantage[i] - advantage[half + i]) * eps
        theta += LEARNING_RATE / (POPULATION * SIGMA) * step

        details = detail_plus + detail_minus
        parts = reward.summarize([p for d in details for p in d])
        # 対ごとの報酬差。同じ乱数で比べているので、これが回路の効果そのもの
        paired = float(np.abs(reward_plus - reward_minus).mean())
        line = (f"世代{generation:3d}  報酬 平均{rewards.mean():+.3f}"
                f" 最良{rewards.max():+.3f} 対差{paired:.3f}"
                f"  跳躍{parts['跳躍']:.2f} 崩壊{parts['崩壊']:.2f}"
                f" 局所{parts['局所']:.2f} 切断{parts['切断']:.2f}"
                f" 轍{parts['轍率']:.2f} 抑圧{parts['抑圧']:.2f}"
                f" 門比{parts['門比']:.1f}"
                f"  {time.time() - began:.0f}秒")
        print(line)
        log.write(line + "\n")
        log.flush()

        if generation % SAMPLE_EVERY == 0 or generation == GENERATIONS:
            circuit = unpack(base(), theta)
            collected = []
            evaluate([circuit], models, centers, generation, collect=collected)
            log.write(f"\n--- 世代{generation} の出力 ---\n")
            for prompt, text, parts in collected:
                log.write(f"[{prompt}]\n{text.strip()}\n")
                log.write(f"(跳躍{parts['跳躍']:.2f} 尤度{parts['尤度']:.2f}"
                          f" 局所{parts['局所']:.2f} 抑圧{parts['抑圧']:.2f})\n\n")
            log.flush()
            # 版を書いておく。評価側がV2とV3を取り違えないため
            torch.save({"theta": theta.cpu(), "mean": mean, "std": std,
                        "generation": generation, "version": "V3"},
                       CHECKPOINT)

    log.close()
    print(f"完了: {CHECKPOINT} と {LOG_FILE}")


main()

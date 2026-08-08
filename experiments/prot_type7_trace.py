"""
センチュリオン: 実特徴トレースの記録 (Phase 1)
抑圧をかけずに生成し、回路に渡している4特徴が実際どう動いているかを記録する。
目的は3つ:
  1. Phase 2 で使う正規化統計(平均・標準偏差・分位点)を得る
  2. LLMを回さずに回路応答を検証できるオフライン再生データを作る
  3. カスタムprocessorが min_p の前後どちらで呼ばれているかを実測で確定させる
"""

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRACE_FILE = "centurion_trace.npz"
TEXT_FILE = "centurion_trace.txt"

RUNS = 5              # プロンプトごとの試行回数
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

# 統計を1つのお題に偏らせないため、複数のお題から集める
USER_PROMPTS = [
    "青色にまつわる話を聞かせて",
    "朝の匂いについて書いて",
    "忘れられた道具の話をして",
    "冬の終わりをどう感じる",
]


class TraceRecorder(LogitsProcessor):
    """スコアには一切触れず、回路に渡すはずの特徴だけを記録する"""

    def __init__(self):
        self.rows = []

    def __call__(self, input_ids, scores):
        row = scores[0]
        probs = torch.softmax(row.float(), dim=-1)

        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        top1 = probs.max()
        top5 = probs.topk(5).values.sum()
        step = min(len(self.rows) / MAX_TOKENS, 1.0)

        # -inf の数を数えれば、min_p が既に適用済みかどうかが分かる
        finite = torch.isfinite(row).sum()

        self.rows.append((
            entropy.item(), top1.item(), top5.item(), step,
            int(finite.item()),
        ))
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


def generate(model, tokenizer, inputs, recorder):
    """抑圧なしで生成しつつ、特徴を記録する"""
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_TOKENS,
        do_sample=True,
        temperature=1.0,
        min_p=MIN_P,
        top_p=TOP_P,
        logits_processor=LogitsProcessorList([recorder]),
    )
    generated_ids = output[0][inputs.input_ids.shape[1]:]
    body = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_ids.tolist(), PREFILL + body


def collect(model, tokenizer):
    """全お題・全試行の特徴とテキストを集める"""
    records = []   # (prompt_id, run, rows, token_ids, text)

    for prompt_id, user_prompt in enumerate(USER_PROMPTS):
        inputs = build_inputs(tokenizer, user_prompt)
        for run in range(1, RUNS + 1):
            print(f"[{prompt_id + 1}/{len(USER_PROMPTS)}] {user_prompt}"
                  f"  {run}/{RUNS}")
            recorder = TraceRecorder()
            token_ids, text = generate(model, tokenizer, inputs, recorder)
            records.append((prompt_id, run, recorder.rows, token_ids, text))

    return records


def flatten(records):
    """記録を1枚の配列に畳む。行の並びは生成順と一致する"""
    columns = {k: [] for k in
               ("entropy", "top1", "top5", "step", "n_finite",
                "token_id", "prompt_id", "run_id")}

    for prompt_id, run, rows, token_ids, _ in records:
        # processorの呼び出し回数と生成トークン数がずれる場合は短いほうに合わせる
        length = min(len(rows), len(token_ids))
        for i in range(length):
            entropy, top1, top5, step, finite = rows[i]
            columns["entropy"].append(entropy)
            columns["top1"].append(top1)
            columns["top5"].append(top5)
            columns["step"].append(step)
            columns["n_finite"].append(finite)
            columns["token_id"].append(token_ids[i])
            columns["prompt_id"].append(prompt_id)
            columns["run_id"].append(run)

    return {k: np.array(v) for k, v in columns.items()}


def report_processor_order(data, vocab_size):
    """min_p の前に呼ばれているのか後なのかを実測から判定する"""
    n_finite = data["n_finite"]
    print("\n" + "=" * 60)
    print("プロセッサ順序の確認")
    print("=" * 60)
    print(f"語彙数: {vocab_size}")
    print(f"生きている候補数: 最小{n_finite.min()} 中央{int(np.median(n_finite))}"
          f" 最大{n_finite.max()}")

    if n_finite.min() >= vocab_size:
        print("→ 全候補が生きている。processorは min_p より前に呼ばれている。")
        print("  エントロピーは素の分布に対する値。")
    else:
        print("→ 候補が既に絞られている。processorは min_p より後に呼ばれている。")
        print("  エントロピーは切り詰め後の分布に対する値であり、"
              "type5/type6 の ENTROPY_GATE の基準もこの分布に基づく。")


def report_stats(data):
    """Phase 2 の標準化に使う統計を出す"""
    print("\n" + "=" * 60)
    print("特徴の実分布")
    print("=" * 60)
    print(f"{'特徴':<10} {'平均':>8} {'標準偏差':>9} {'最小':>8}"
          f" {'5%':>8} {'50%':>8} {'95%':>8} {'最大':>8}")

    for name in ("entropy", "top1", "top5"):
        v = data[name]
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        print(f"{name:<10} {v.mean():8.3f} {v.std():9.3f} {v.min():8.3f}"
              f" {p5:8.3f} {p50:8.3f} {p95:8.3f} {v.max():8.3f}")

    # type6 は entropy/5.0 という固定のスケールを使っていた。その妥当性を見る
    scaled = data["entropy"] / 5.0
    print(f"\ntype6 の entropy/5.0: 平均{scaled.mean():.3f}"
          f" 標準偏差{scaled.std():.3f}"
          f" 範囲{scaled.min():.3f}〜{scaled.max():.3f}")
    print(f"top1 と top5 の相関: {np.corrcoef(data['top1'], data['top5'])[0, 1]:.3f}")


def save(data, records):
    np.savez(TRACE_FILE, **data)
    print(f"\nトレース: {TRACE_FILE} ({len(data['entropy'])}行)")

    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(f"モデル: {MODEL_NAME}\n")
        f.write("制御なし(トレース記録のみ)\n\n")
        for prompt_id, run, rows, _, text in records:
            f.write("-" * 60 + "\n")
            f.write(f"お題: {USER_PROMPTS[prompt_id]}  試行 {run}\n")
            f.write(text.strip() + "\n")
            entropies = [r[0] for r in rows]
            f.write(f"[エントロピー: 最小{min(entropies):.2f}"
                    f" 最大{max(entropies):.2f}"
                    f" 平均{sum(entropies) / len(entropies):.2f}]\n\n")
    print(f"本文: {TEXT_FILE}")


# ===== 実行 =====
def main():
    tokenizer, model = load_model()
    records = collect(model, tokenizer)
    data = flatten(records)

    report_processor_order(data, model.config.vocab_size)
    report_stats(data)
    save(data, records)


main()

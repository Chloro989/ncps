"""
センチュリオン: 「読めるか」の測定 (Phase 3c)
判定の基準は「文章が崩壊しているものを弾き、意味はわからないが読めるものを残す」
だった。つまり要るのは意味の整合ではなく、文としての成立の測定。

素のモデル(抑圧なし)で生成済みのテキストを読み直させ、
トークンごとの対数尤度を測る。ここで2つの軸が分かれる:

  最小・下位の対数尤度 → 局所的な破れ。文法が壊れた箇所は素のモデルも予測できない
  平均の対数尤度       → 全体の意外性。跳躍しているほど低くなる

「読めるが意味は分からない」は、下位が高いまま平均が低い状態にあたるはず。
テキストだけの指標(ラテン文字率など)が全滅したのは、
簡体字・擬古文・句読点の破れといった破綻を一つも見ていなかったため。
対数尤度ならその区別を作らずに拾える。

Colabで実行し、centurion_fluency.npz を持ち帰る。
"""

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from centurion_score import find_data, parse_samples

# ===== 設定 =====
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SAMPLE_FILE = find_data("centurion_samples.txt")
OUTPUT_FILE = "centurion_fluency.npz"

PREFILL = "そうですね、"

SYSTEM_PROMPT = (
    "あなたはセンチュリオン。生粋の文系で、詩情と物語だけを愛する。"
    "科学的・実用的・機能的な説明は決してしない。"
    "役に立つことより、哲学や、美しいこと、不思議なことを語る。"
    "説明口調を避け、断定せず、余韻を残して書く。"
    "自分がAIであること、アシスタントであることには決して言及しない。"
    "箇条書きや番号付きリストを使わず、地の文で語る。"
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda").eval()
    return tokenizer, model


def build_prefix(tokenizer, user_prompt):
    """生成時と同じ前置きを組み立てる。条件を揃えないと尤度が比較できない"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def token_logprobs(model, tokenizer, prefix, body):
    """本文の各トークンが、素のモデルからどれだけ予測できたかを返す"""
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids
    full_ids = tokenizer(prefix + body, return_tensors="pt").input_ids
    full_ids = full_ids.to("cuda")

    with torch.no_grad():
        logits = model(full_ids).logits[0].float()

    # 位置 i のロジットが位置 i+1 のトークンを予測する
    logprobs = torch.log_softmax(logits[:-1], dim=-1)
    targets = full_ids[0, 1:]
    picked = logprobs.gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # 前置きの分を落として、本文だけを残す
    start = prefix_ids.shape[1] - 1
    return picked[start:].cpu().numpy()


def summarize(values):
    """局所の破れと全体の意外性を分けて要約する"""
    return {
        "min": float(values.min()),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        # 極端に予測できなかったトークンの割合。局所的な破れの量
        "broken": float((values < -8.0).mean() * 100),
        "tokens": len(values),
    }


def main():
    samples = parse_samples(SAMPLE_FILE)
    labeled = [s for s in samples if s["label"]]
    print(f"標本 {len(samples)}件 (判定済み {len(labeled)}件)")

    tokenizer, model = load_model()

    keys = ("min", "p5", "p25", "mean", "median", "broken", "tokens")
    columns = {k: [] for k in keys}
    columns["id"], columns["label"], columns["strength"] = [], [], []

    print(f"\n{'標本':<14} {'判定':<4} {'最小':>8} {'下位5%':>8}"
          f" {'平均':>8} {'破れ%':>7}")
    for s in samples:
        prefix = build_prefix(tokenizer, s["prompt"])
        body = s["text"][len(PREFILL):] if s["text"].startswith(PREFILL) else s["text"]
        values = token_logprobs(model, tokenizer, prefix + PREFILL, body)
        stats = summarize(values)

        for k in keys:
            columns[k].append(stats[k])
        columns["id"].append(s["id"])
        columns["label"].append(s["label"])
        columns["strength"].append(s["strength"])

        print(f"{s['id']:<14} {s['label']:<4} {stats['min']:8.2f}"
              f" {stats['p5']:8.2f} {stats['mean']:8.2f} {stats['broken']:7.1f}")

    np.savez(OUTPUT_FILE, **{k: np.array(v) for k, v in columns.items()})
    print(f"\n完了: {OUTPUT_FILE}")


main()

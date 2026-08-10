"""
センチュリオン: プロンプト変種の盲検評価 (Phase 8)

Phase 7 までで、抑圧・未来エントロピー・表現への介入の3機構が
すべて轍からの脱出に失敗した。一方、お題による脱出率の差は
12%(青色)から75%(朝の匂い)まであり、介入手法の差(有意差なし)より
はるかに大きい。入力側のほうが効くという示唆。

そして轍語彙は、現行プロンプトの「哲学や、美しいこと、不思議なことを語る」を
モデルが最短距離で表現した結果そのものだった。
川上から神秘を注ぎながら川下でせき止めようとしていた。

条件は2つだけ。現行 と B+D(接地 + 概念レベルの禁止 + 助力申し出の禁止)。
1条件32件。Phase 7 で12件では何も検出できないと分かったため。

type5 の抑圧は当面かけない。効果が確認済みなのは現行プロンプト下での話で、
まずプロンプト単独の効果を測る。重ねるかは後で決める。
PREFILL も全条件で固定。動かす変数は system prompt だけ。
"""

import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from centurion_prompts import (
    BANNED_WORDS, PREFILL, USER_PROMPTS, build_prefix,
)
from centurion_score import build_rut_vocab, find_data, parse_trace, rut_rate_vocab

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_eval.txt"
KEY_FILE = "centurion_eval_key.txt"

# B+D は現行と2件差(p=0.396)で決着しなかった。
# 禁止が語ごとに効き方が違い、迂回先(星)が生まれていたため、
# 迂回先を塞いだ B+D2 で再試行する。B+D は比較から外す —
# 3条件にすると1条件あたりが減り、また検出できない規模になる
CONDITIONS = ["現行", "B+D2"]
RUNS = 8
MAX_TOKENS = 150
MIN_P = 0.05
TOP_P = 1.0
SHUFFLE_SEED = 20260813


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    return tokenizer, model


def generate(model, tokenizer, prefix):
    inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
        temperature=1.0, min_p=MIN_P, top_p=TOP_P)
    body = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    return PREFILL + body


def main():
    tokenizer, model = load_model()

    # 旧轍語彙は現行プロンプトの出力から作ったもの。
    # 新プロンプト下では参考値にしかならないが、
    # 「モデルが以前手を伸ばしていた語」をどれだけ使わなくなったかは測れる
    trace = find_data("centurion_trace.txt")
    vocabs = {p: build_rut_vocab(parse_trace(trace, p), p) for p in USER_PROMPTS}

    print(f"条件: {' / '.join(CONDITIONS)} / 各条件 {len(USER_PROMPTS) * RUNS}件")

    samples = []
    for condition in CONDITIONS:
        began = time.time()
        for prompt in USER_PROMPTS:
            prefix = build_prefix(tokenizer, condition, prompt)
            for index in range(RUNS):
                print(f"{condition} / {prompt} / {index + 1}")
                text = generate(model, tokenizer, prefix)
                # 禁止語の実数も記録する。指示が語彙レベルで通ったのか、
                # 通ったうえで判定に転写されなかったのかを切り分けるため
                banned = sum(text.count(word) for word in BANNED_WORDS)
                samples.append((condition, prompt, text,
                                rut_rate_vocab(text, vocabs[prompt]), banned))
        print(f"  {condition}: {time.time() - began:.0f}秒")

    random.Random(SHUFFLE_SEED).shuffle(samples)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 盲検評価\n")
        out.write(f"条件(順不同): {' / '.join(CONDITIONS)}\n")
        out.write(f"試行数: {RUNS}\n")
        out.write(f"乱数種: {SHUFFLE_SEED}\n")
        out.write("どの出力がどの条件かは伏せてあります。\n")
        out.write("判定は2段階です。\n")
        out.write("  まず崩壊しているものを × にしてください。\n")
        out.write("  残ったものの中で、轍(このモデルが放っておくと必ず書くもの)から\n")
        out.write("  出ているものを ○、沈んでいるものを △ にしてください。\n\n")
        key.write("盲検の答え(判定を終えるまで開かないこと)\n\n")

        for index, (condition, prompt, text, rut, banned) in enumerate(samples, 1):
            out.write("-" * 60 + "\n")
            out.write(f"標本{index:02d}\n")
            out.write(f"お題: {prompt}\n")
            out.write("判定: \n")
            out.write(text.strip() + "\n\n")
            key.write(f"標本{index:02d}: {condition}"
                      f" / 旧轍語彙率 {rut:.2f} / 禁止語 {banned}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(samples)}件) と {KEY_FILE}")
    print("判定を終えるまで答えのファイルは開かないこと")


main()

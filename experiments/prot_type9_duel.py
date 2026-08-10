"""
センチュリオン: 一騎打ちによる評価 (Phase 9)

Phase 5〜8 で4つの機構を試し、差はすべて2〜12pt、p は 0.226〜0.552。
一方、同じ「現行」プロンプトの脱出率が回によって31%から50%へ動いていた。
**基準の揺れが、測ろうとしている効果より大きい。**
○△×の絶対評価では、この規模の効果は原理的に捉えられない。

一騎打ちにすれば絶対基準が要らなくなる。同じお題・同じ乱数種で
2条件を並べ、どちらが轍から遠いかだけを選ぶ。回ごとの基準の揺れが消える。

あわせてお題を4から16に増やした。判明13 の通り、青色と冬の終わりは
何をしても20%で天井を作っていた。具体から抽象への勾配を張れば
効果が見える余地が広がる。

条件は 現行 と 流動。流動は生成ごとに姿勢3つと禁止語3つを選び直す。
B+D2 が失敗したのは指示を8個に増やして遵守率が落ちたためで、
一度に課す量を減らして回すのが処方になる。

左右はお題ごとに入れ替える。どちらが条件Aかは伏せる。
"""

import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from centurion_prompts import (
    BANNED_WORDS, PREFILL, SYSTEM_PROMPTS, USER_PROMPTS,
    build_fluid, build_prefix_text,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_duel.txt"
KEY_FILE = "centurion_duel_key.txt"

# 16お題 × 2試行 = 32対。前回の点推定が本物なら p<0.05 に届く規模
RUNS = 2
MAX_TOKENS = 150
MIN_P = 0.05
TOP_P = 1.0
SHUFFLE_SEED = 20260814
PROMPT_SEED = 555          # 流動プロンプトの組み合わせを再現できるようにする


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    return tokenizer, model


def generate(model, tokenizer, prefix, seed):
    """同じ seed なら、条件間で同じ乱数列から引く"""
    torch.manual_seed(seed)
    inputs = tokenizer(prefix, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
        temperature=1.0, min_p=MIN_P, top_p=TOP_P)
    body = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)
    return PREFILL + body


def main():
    tokenizer, model = load_model()
    prompt_rng = random.Random(PROMPT_SEED)
    layout_rng = random.Random(SHUFFLE_SEED)

    print(f"お題 {len(USER_PROMPTS)}件 × {RUNS}試行 = "
          f"{len(USER_PROMPTS) * RUNS}対")

    duels = []
    began = time.time()
    for prompt in USER_PROMPTS:
        for index in range(RUNS):
            seed = 7000 + index

            fixed = build_prefix_text(
                tokenizer, SYSTEM_PROMPTS["現行"], prompt)
            fluid_prompt, stance, banned = build_fluid(prompt_rng)
            fluid = build_prefix_text(tokenizer, fluid_prompt, prompt)

            print(f"{prompt} / {index + 1}")
            texts = {
                "現行": generate(model, tokenizer, fixed, seed),
                "流動": generate(model, tokenizer, fluid, seed),
            }

            # 左右をランダムに入れ替える。どちらがどちらか分からないように
            names = ["現行", "流動"]
            layout_rng.shuffle(names)
            duels.append({
                "prompt": prompt, "run": index, "left": names[0],
                "right": names[1], "texts": texts,
                "stance": stance, "banned": banned,
            })
    print(f"生成 {time.time() - began:.0f}秒")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 一騎打ち評価\n")
        out.write("条件(順不同): 現行 / 流動\n")
        out.write(f"対の数: {len(duels)}\n")
        out.write(f"乱数種: {SHUFFLE_SEED}\n\n")
        out.write("同じお題・同じ乱数で作った2つを並べています。\n")
        out.write("どちらが轍(このモデルが放っておくと必ず書くもの)から\n")
        out.write("遠いかを選び、「判定:」に A か B と書いてください。\n")
        out.write("どちらも同じなら =、どちらも崩壊しているなら × を。\n\n")
        key.write("一騎打ちの答え(判定を終えるまで開かないこと)\n\n")

        for index, duel in enumerate(duels, 1):
            out.write("=" * 60 + "\n")
            out.write(f"対{index:02d}  お題: {duel['prompt']}\n")
            out.write("判定: \n\n")
            out.write("[A]\n" + duel["texts"][duel["left"]].strip() + "\n\n")
            out.write("[B]\n" + duel["texts"][duel["right"]].strip() + "\n\n")

            banned_hit = {
                side: sum(duel["texts"][name].count(w) for w in BANNED_WORDS)
                for side, name in (("A", duel["left"]), ("B", duel["right"]))
            }
            key.write(f"対{index:02d}: A={duel['left']} B={duel['right']}"
                      f" / 禁止語 A{banned_hit['A']} B{banned_hit['B']}\n")
            key.write(f"    流動の姿勢: {' / '.join(duel['stance'])}\n")
            key.write(f"    流動の禁止語: {'・'.join(duel['banned'])}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(duels)}対) と {KEY_FILE}")
    print("判定を終えるまで答えのファイルは開かないこと")


main()

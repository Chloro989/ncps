"""
センチュリオン: 会話をまたぐ姿勢の漂い、一騎打ち (Phase 10)

Phase 9 で流動プロンプトが勝った。ただし句の選択は生成ごとに独立で、
そこには状態がない。回路の出番もない。

複数ターンにすれば状態が生まれる。回路が勝つべき相手は固定プロンプトでは
なく**ランダム流動**である。姿勢が変わるだけならランダムで足りる。
回路が加えられるのは変化の連続性だけ — ランダムは毎ターン独立に飛ぶが、
回路は記憶を持つので前のターンを引き継いで漂う。

仮説: 連続した漂いは一貫した語り手として読め、独立な飛びは不安定に読める。

学習は使わない。報酬は5回作ろうとして作れなかった。
連続性は回路の記憶による構造的性質として持たせ、
prot_type10_probe.py で事前に測ってある(漂い0.98、緩和4.67ターン)。

設計:
  - 1ターン目は両条件で共通。始まりを揃えて、差を2ターン目以降に絞る
  - 禁止語は毎ターン両条件で同じものを使う。動くのは姿勢の選び方だけ
  - 生成の乱数種も (会話, ターン) ごとに両条件で同じ
  - 回路の入力は自分の直前の応答から作る。閉じた輪になっている
"""

import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from centurion_prompts import (
    BANNED_WORDS, BAN_COUNT, PREFILL, RUT_WORDS, STANCE_CLAUSES, STANCE_COUNT,
)
from centurion_turn import (
    CONVERSATIONS, SHARED_TURNS, TURNS, StanceCircuit,
    build_conversation, compose, pick_by_circuit, response_features,
)
from prot_type10_probe import check, find_seed

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "centurion_turn_duel.txt"
KEY_FILE = "centurion_turn_duel_key.txt"

MAX_TOKENS = 150
MIN_P = 0.05
TOP_P = 1.0

PROMPT_SEED = 778          # 禁止語とランダム流動の姿勢
SHUFFLE_SEED = 20260813    # 左右の入れ替え
GENERATION_SEED = 8100

CONDITIONS = ["回路", "ランダム"]


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


def build_circuit():
    """回路の重みの種は受け入れ試験に選ばせる。
    種によって漂いが0.72〜1.41、凍結が0〜5本と大きく違うので、
    こちらで指定すると結果を見てから選んだことになる。
    find_seed() は条件を先に決めて種0から順に探す"""
    seed = find_seed()
    torch.manual_seed(seed)
    circuit = StanceCircuit(len(STANCE_CLAUSES))
    circuit.settle()
    drift, variety, frozen, share, relax = check(seed)
    print(f"回路 種{seed} (受け入れ試験が選んだ) / 漂い{drift:.2f} / "
          f"種類{variety} / 凍結{frozen}本 / 最頻句{share:.0%} / "
          f"緩和{relax}ターン")
    return circuit


def main():
    tokenizer, model = load_model()
    circuit = build_circuit()
    prompt_rng = random.Random(PROMPT_SEED)
    layout_rng = random.Random(SHUFFLE_SEED)

    total = len(CONVERSATIONS) * (SHARED_TURNS + (TURNS - SHARED_TURNS) * 2)
    print(f"会話 {len(CONVERSATIONS)}本 × {TURNS}ターン "
          f"(先頭{SHARED_TURNS}ターンは共通) = 生成{total}回")

    duels = []
    began = time.time()
    for index, topics in enumerate(CONVERSATIONS):
        circuit.reset()
        # 各条件のやり取りの記録。(お題, 応答) の並び
        history = {name: [] for name in CONDITIONS}
        stances = {name: [] for name in CONDITIONS}

        for turn, topic in enumerate(topics):
            seed = GENERATION_SEED + index * TURNS + turn
            # 禁止語は両条件で共通。動かすのは姿勢だけにする
            banned = prompt_rng.sample(RUT_WORDS, BAN_COUNT)
            random_stance = prompt_rng.sample(STANCE_CLAUSES, STANCE_COUNT)

            if turn < SHARED_TURNS:
                # 始まりを揃える。姿勢もランダムのものを両方で使う
                prefix = build_conversation(
                    tokenizer, compose(random_stance, banned),
                    history[CONDITIONS[0]], topic)
                text = generate(model, tokenizer, prefix, seed)
                for name in CONDITIONS:
                    history[name].append((topic, text))
                    stances[name].append(random_stance)
                print(f"[{index + 1:02d}/{turn + 1}] 共通 {topic}")
                continue

            previous = history["回路"][-1][1]
            features = response_features(previous, turn, TURNS)
            circuit_stance, _ = pick_by_circuit(circuit, features)
            chosen = {"回路": circuit_stance, "ランダム": random_stance}

            for name in CONDITIONS:
                prefix = build_conversation(
                    tokenizer, compose(chosen[name], banned),
                    history[name], topic)
                text = generate(model, tokenizer, prefix, seed)
                history[name].append((topic, text))
                stances[name].append(chosen[name])
            print(f"[{index + 1:02d}/{turn + 1}] {topic}")

        names = list(CONDITIONS)
        layout_rng.shuffle(names)
        duels.append({"topics": topics, "left": names[0], "right": names[1],
                      "history": history, "stances": stances})

    print(f"生成 {time.time() - began:.0f}秒")
    write(duels)


def write(duels):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(KEY_FILE, "w", encoding="utf-8") as key:
        out.write("センチュリオン: 会話をまたぐ姿勢の漂い、一騎打ち\n")
        out.write(f"条件(順不同): {' / '.join(CONDITIONS)}\n")
        out.write(f"対の数: {len(duels)}\n")
        out.write(f"乱数種: {SHUFFLE_SEED}\n\n")
        out.write(f"{TURNS}ターンの会話を2つ並べています。\n")
        out.write(f"先頭{SHARED_TURNS}ターンは両方まったく同じなので"
                  "一度だけ書いてあります。\n")
        out.write("お題の並びも乱数も揃えてあり、違うのは"
                  "各ターンの語り方の指示だけです。\n\n")
        out.write("2つのことを判定してください。どちらも A か B、\n")
        out.write("差がなければ =、どちらも崩壊していれば × を書きます。\n\n")
        out.write("  一貫性: 同じ一人の語り手が語っているように読めるのはどちらか\n")
        out.write("          (同じことを繰り返すのは一貫ではありません。\n")
        out.write("           語り口が移りながらも一人に見えるかどうか)\n")
        out.write("  面白さ: 轍(このモデルが放っておくと必ず書くもの)から\n")
        out.write("          遠く、読み物として面白いのはどちらか\n\n")
        key.write("会話の一騎打ちの答え(判定を終えるまで開かないこと)\n\n")

        for index, duel in enumerate(duels, 1):
            out.write("=" * 64 + "\n")
            out.write(f"対{index:02d}\n")
            out.write("一貫性: \n")
            out.write("面白さ: \n\n")

            shared = duel["history"][duel["left"]][:SHARED_TURNS]
            for turn, (topic, text) in enumerate(shared, 1):
                out.write(f"--- {turn}ターン目 (両方共通) お題: {topic}\n")
                out.write(text.strip() + "\n\n")

            for side, name in (("A", duel["left"]), ("B", duel["right"])):
                out.write(f"[{side}]\n")
                for turn, (topic, text) in enumerate(
                        duel["history"][name][SHARED_TURNS:],
                        SHARED_TURNS + 1):
                    out.write(f"  {turn}ターン目 お題: {topic}\n")
                    out.write("  " + text.strip() + "\n\n")

            key.write(f"対{index:02d}: A={duel['left']} B={duel['right']}\n")
            for side, name in (("A", duel["left"]), ("B", duel["right"])):
                hit = sum(text.count(word)
                          for _, text in duel["history"][name]
                          for word in BANNED_WORDS)
                key.write(f"    {side} ({name}) 轍語{hit}回\n")
                for turn, stance in enumerate(duel["stances"][name], 1):
                    mark = "共通" if turn <= SHARED_TURNS else "    "
                    key.write(f"      {turn} {mark} {' / '.join(stance)}\n")

    print(f"\n完了: {OUTPUT_FILE} ({len(duels)}対) と {KEY_FILE}")
    print(f"{len(duels)}対なら符号検定で "
          f"{needed(len(duels))}勝以上で p<0.05")
    print("判定を終えるまで答えのファイルは開かないこと")


def needed(pairs):
    """符号検定で片側 p<0.05 になる最小の勝ち数。
    Phase 9 の答え合わせと同じ統計にして、数字を並べて比べられるようにする"""
    from math import comb
    for wins in range(pairs // 2 + 1, pairs + 1):
        tail = sum(comb(pairs, k) for k in range(wins, pairs + 1))
        if tail / 2 ** pairs < 0.05:
            return wins
    return pairs


if __name__ == "__main__":
    main()

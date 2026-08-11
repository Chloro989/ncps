"""
原稿を読ませて、論評か発想を得る。

    python -m centurion.critique 原稿.txt                    # 発想の問いを出す
    python -m centurion.critique 原稿.txt --mode 査読
    python -m centurion.critique 原稿.txt --mode 接続
    python -m centurion.critique 原稿.txt --mode 連想
    python -m centurion.critique 原稿.txt --run               # その場でモデルに解かせる
    python -m centurion.critique 原稿.txt --check 答え.txt    # 段落番号を検査する

既定では**プロンプトを出すだけ**で、モデルは呼ばない。
手元にGPUが無くても使えるようにするためで、出したものを好きなチャットへ
貼れば、性能の高いモデルで読ませられる。
--run を付けたときだけモデルを読み込む(Colab を想定)。

段落番号の検査だけは、答えを受け取ってから別に走らせられる。
「実在しない箇所への指摘」は、モデルが強くなっても消えないため。
"""

import argparse
import random
import sys

from .connect import (DREAM_WORK, build_chain_prompt,
                      build_connection_prompt, distant_pairs, recurrences)
from .manuscript import Manuscript
from .review import build_prompt, check_citations, choose_lenses, resolve

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_TOKENS = 1200          # 論評は長い。生成の150では話にならない
CHUNK_SIZE = 6000
MODES = ["発想", "査読", "接続", "連想"]


def ask(model_name, head, body, max_tokens=MAX_TOKENS):
    """モデルに一度だけ解かせる。ここでは抑圧を入れない —
    type5 の抑圧は小説の読みやすさで検証したもので、
    分析の文章に効くかは確かめていない"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": head},
         {"role": "user", "content": body}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens,
                                do_sample=True, temperature=0.7,
                                min_p=0.05, top_p=1.0,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True)


def pick_chunk(manuscript, size, number):
    chunks = manuscript.chunks(size=size, overlap=1)
    if not chunks:
        raise SystemExit("切り出せる段落が無い")
    if number is None:
        return chunks, chunks[0]
    if not 1 <= number <= len(chunks):
        raise SystemExit(f"塊は1〜{len(chunks)}の範囲で指定する")
    return chunks, chunks[number - 1]


def compose(manuscript, args):
    """モードに応じて (指示, 本文, 添える説明) を作る"""
    rng = random.Random(args.seed)

    if args.mode in ("発想", "査読"):
        chunks, chunk = pick_chunk(manuscript, args.size, args.chunk)
        lenses = choose_lenses(rng, count=args.lenses)
        place = (f"{len(chunks)}つに分けたうちの"
                 f"{chunks.index(chunk) + 1}つ目、{chunk}")
        head, body = build_prompt(
            chunk, lenses, mode=args.mode, title=manuscript.title,
            author=manuscript.author, note=args.note, place=place)
        allowed = {p.index for p in chunk.paragraphs[chunk.carried:]}
        return head, body, allowed, f"観点: {'／'.join(l.key for l in lenses)}"

    if args.mode == "接続":
        # 反復があればそれを優先する。作者がすでに植えた種のほうが確度が高い
        found = recurrences(manuscript)
        motifs = [item for item in found if item.kind == "反復"]
        if motifs:
            item = motifs[rng.randrange(min(len(motifs), args.top))]
            pair = item.pair()
            label = f"反復 {item}"
        else:
            pairs = distant_pairs(manuscript, count=args.top, rng=rng)
            if not pairs:
                raise SystemExit("繋げる対が見つからない。原稿が短すぎる")
            pair = pairs[0]
            label = f"遠い対 {pair}"
        extra = [DREAM_WORK[rng.randrange(len(DREAM_WORK))]] if args.dream else ()
        head, body = build_connection_prompt(manuscript, pair,
                                             note=args.note, extra=extra)
        allowed = None
        return head, body, allowed, label

    # 連想
    usable = [p for p in manuscript.paragraphs if len(p.text) >= 40]
    if not usable:
        raise SystemExit("連想の起点にできる段落が無い")
    start = usable[rng.randrange(len(usable))]
    head, body = build_chain_prompt(manuscript, start, steps=args.steps,
                                    note=args.note)
    return head, body, None, f"起点 [{start.index}]"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="centurion.critique",
        description="原稿を読ませて、論評か発想を得る")
    parser.add_argument("path", help="原稿のファイル")
    parser.add_argument("--mode", default="発想", choices=MODES)
    parser.add_argument("--size", type=int, default=CHUNK_SIZE,
                        help=f"1塊の上限文字数 (既定 {CHUNK_SIZE})")
    parser.add_argument("--chunk", type=int,
                        help="何番目の塊を読ませるか (既定 1つ目)")
    parser.add_argument("--lenses", type=int, default=3,
                        help="一度に渡す観点の数 (既定 3)")
    parser.add_argument("--top", type=int, default=5,
                        help="接続モードで候補の上位いくつから選ぶか")
    parser.add_argument("--steps", type=int, default=4,
                        help="連想モードで何歩たどらせるか")
    parser.add_argument("--dream", action="store_true",
                        help="接続モードに夢の作業を一つ添える")
    parser.add_argument("--note", default="",
                        help="作者からの補足。狙いや訊きたいこと")
    parser.add_argument("--seed", type=int, help="観点や対の選び方を固定する")
    parser.add_argument("--list", action="store_true",
                        help="塊と反復の一覧だけを出す")
    parser.add_argument("--run", action="store_true",
                        help="その場でモデルに解かせる。GPUが要る")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--check", metavar="答え",
                        help="答えのファイルを読み、段落番号を検査する")
    return parser


def show_list(manuscript, args):
    print(manuscript.summary())
    chunks = manuscript.chunks(size=args.size, overlap=1)
    print(f"\n{args.size}文字ずつに切ると {len(chunks)}塊")
    for chunk in chunks:
        print("  " + str(chunk))
    found = recurrences(manuscript)
    for kind in ("反復", "主題"):
        rows = [item for item in found if item.kind == kind]
        print(f"\n{kind} {len(rows)}件" + (" (上位12)" if len(rows) > 12 else ""))
        for item in rows[:12]:
            print("  " + str(item))


def run_check(manuscript, args):
    """答えの段落番号を検査する。

    --chunk を渡すと、そのとき見せた範囲の外を指した指摘も見つける。
    番号が実在することと、モデルがその段落を読んでいたことは別で、
    見せていない段落への言及は中身を確かめずに書いたものになる。
    実際にこの検査を作った日に、範囲外の段落を2件引いて
    どちらも中身を取り違えた例が出た"""
    answer = open(args.check, encoding="utf-8").read()
    allowed = None
    if args.chunk is not None:
        _, chunk = pick_chunk(manuscript, args.size, args.chunk)
        allowed = {p.index for p in chunk.paragraphs}

    real, missing, outside = check_citations(answer, manuscript,
                                             allowed=allowed)
    total = len(real) + len(missing) + len(outside)
    print(f"示された段落番号 {total}件")
    print(f"  実在する {len(real)}件: {sorted(set(real))}")
    if missing:
        print(f"  × 存在しない {len(missing)}件: {sorted(set(missing))}")
        print("    実在しない箇所への指摘。捨てること")
    if outside:
        print(f"  △ 見せていない範囲 {len(outside)}件: {sorted(set(outside))}")
        print("    番号は実在するが、この読みでは渡していない段落。")
        print("    中身を確かめずに書いている可能性が高い")
    if not missing and not outside:
        print("  すべて渡した範囲の中にある"
              if allowed else "  存在しない番号は無し")
    if allowed is None:
        print("  ※ --chunk を渡すと、見せていない範囲への言及も検出できる")

    print("\n--- 番号を本文に戻したもの ---")
    print(resolve(answer, manuscript))
    return 1 if missing or outside else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    manuscript = Manuscript.load(args.path)

    if args.check:
        return run_check(manuscript, args)
    if args.list:
        show_list(manuscript, args)
        return 0

    head, body, allowed, label = compose(manuscript, args)
    print(f"# {args.mode}モード / {label}", file=sys.stderr)

    if not args.run:
        # 貼りやすい形で出す。指示と本文の境目を残す
        print(head)
        print()
        print("---")
        print()
        print(body)
        print(f"\n# 答えを得たら --check で段落番号を検査すること",
              file=sys.stderr)
        return 0

    print(f"# {args.model} を読み込んでいます…", file=sys.stderr)
    answer = ask(args.model, head, body, args.tokens)
    print(answer)

    real, missing, outside = check_citations(answer, manuscript,
                                             allowed=allowed)
    print(f"\n# 段落番号 実在{len(real)}件"
          + (f" / 存在しない{len(missing)}件 {sorted(set(missing))}"
             if missing else "")
          + (f" / 担当範囲の外{len(outside)}件 {sorted(set(outside))}"
             if outside else ""),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

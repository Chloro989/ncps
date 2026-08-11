"""
コマンドラインからセンチュリオンを使う。

    python -m centurion 青色にまつわる話を聞かせて
    python -m centurion --chat
    python -m centurion 朝の匂いについて書いて 沈黙について書いて --turns

お題を複数渡すと既定では別々の応答になる。--turns を付けると
前のやり取りを踏まえた一続きの会話になる。
"""

import argparse
import sys

from .generate import MAX_TOKENS, MODEL_NAME
from .prompts import PREFILL


def build_parser():
    parser = argparse.ArgumentParser(
        prog="centurion",
        description="文章を保ったまま、常套から少し外れた語りを返す")
    parser.add_argument("topics", nargs="*", help="お題")
    parser.add_argument("--chat", action="store_true",
                        help="対話する。空行か quit で終わる")
    parser.add_argument("--turns", action="store_true",
                        help="複数のお題を一続きの会話として扱う")
    parser.add_argument("--times", type=int, default=1,
                        help="同じお題を何回書かせるか (既定 1)")
    parser.add_argument("--no-suppress", action="store_true",
                        help="分岐点での抑圧を切る。読みやすさが落ちる")
    parser.add_argument("--fixed-prompt", action="store_true",
                        help="流動プロンプトをやめて固定文にする")
    parser.add_argument("--prefill", default=PREFILL,
                        help=f"書き出し (既定 {PREFILL!r})。"
                             "空にすると相槌なしで始まる")
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS,
                        help=f"生成する長さの上限 (既定 {MAX_TOKENS})")
    parser.add_argument("--raw", action="store_true",
                        help="途中で切れた文を残す")
    parser.add_argument("--seed", type=int, help="乱数種。同じなら同じ文章")
    parser.add_argument("--model", default=MODEL_NAME, help="使うモデル")
    parser.add_argument("--verbose", action="store_true",
                        help="そのとき選ばれた姿勢と禁止語も出す")
    return parser


def show(reply, verbose):
    print(reply.text)
    if verbose and reply.stance:
        print(f"\n  [姿勢] {' / '.join(reply.stance)}", file=sys.stderr)
        print(f"  [禁止] {'・'.join(reply.banned)}", file=sys.stderr)
        print(f"  [抑圧] {len(reply.diverted)}箇所", file=sys.stderr)


def chat(centurion, verbose):
    print("お題を入れてください。空行か quit で終わります。", file=sys.stderr)
    while True:
        try:
            topic = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if not topic or topic in ("quit", "exit"):
            return
        print()
        show(centurion.say(topic), verbose)
        print()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.topics and not args.chat:
        build_parser().print_help()
        return 1

    # モデルの読み込みは重いので、引数を確かめてから
    from .generate import Centurion
    print(f"{args.model} を読み込んでいます…", file=sys.stderr)
    centurion = Centurion(
        model_name=args.model, suppress=not args.no_suppress,
        fluid=not args.fixed_prompt, prefill=args.prefill,
        max_tokens=args.tokens, trim_tail=not args.raw, seed=args.seed)

    if args.chat:
        chat(centurion, args.verbose)
        return 0

    for index, topic in enumerate(args.topics):
        for run in range(args.times):
            if len(args.topics) > 1 or args.times > 1:
                label = f"■ {topic}"
                if args.times > 1:
                    label += f"  ({run + 1}/{args.times})"
                print(("\n" if index or run else "") + label)
            show(centurion.say(topic, remember=args.turns), args.verbose)
        if not args.turns:
            centurion.forget()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

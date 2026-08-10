"""
センチュリオン: 一騎打ちの答え合わせ (Phase 9)

対応表は centurion_duel_key.txt を読む。一騎打ちでは左右の入れ替えを
お題ごとに乱数で決めており、生成順から再現するのが煩雑なため
鍵ファイルを使う(盲検の順序を再現する方式は絶対評価のときだけ)。

符号検定で決着させる。引き分けは除外し、勝敗のついた対だけを数える。
"""

import re
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DUEL_FILE = RESULTS / "centurion_duel.txt"
KEY_FILE = RESULTS / "centurion_duel_key.txt"

BLOCK = re.compile(r"^={20,}$", re.MULTILINE)
HEAD = re.compile(r"^対(\d+)\s+お題: (.+)$", re.MULTILINE)
LABEL = re.compile(r"^判定:[^\S\n]*(.*)$", re.MULTILINE)
KEY_LINE = re.compile(r"^対(\d+): A=(\S+) B=(\S+)"
                      r"(?: / 禁止語 A(\d+) B(\d+))?", re.MULTILINE)

LEFT = {"A", "a", "Ａ", "左"}
RIGHT = {"B", "b", "Ｂ", "右"}
DRAW = {"=", "＝", "同じ", "-"}
BROKEN = {"×", "x", "X", "✕"}


def parse_duels(path):
    duels = []
    for block in BLOCK.split(path.read_text(encoding="utf-8")):
        head = HEAD.search(block)
        if not head:
            continue
        found = LABEL.search(block)
        mark = found.group(1).strip() if found else ""
        duels.append({
            "id": int(head.group(1)),
            "prompt": head.group(2).strip(),
            "choice": ("A" if mark in LEFT else "B" if mark in RIGHT
                       else "=" if mark in DRAW
                       else "×" if mark in BROKEN else ""),
            "mark": mark,
        })
    return sorted(duels, key=lambda d: d["id"])


def parse_key(path):
    key = {}
    for found in KEY_LINE.finditer(path.read_text(encoding="utf-8")):
        key[int(found.group(1))] = {
            "A": found.group(2), "B": found.group(3),
            "banned_a": int(found.group(4)) if found.group(4) else None,
            "banned_b": int(found.group(5)) if found.group(5) else None,
        }
    return key


def sign_test(wins, losses):
    """勝敗が偏っているかを、引き分けを除いた符号検定で見る"""
    total = wins + losses
    if total == 0:
        return float("nan")
    return sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total


def main():
    if not KEY_FILE.exists():
        raise SystemExit(f"{KEY_FILE.name} が無い。生成時の鍵ファイルが必要")

    duels = parse_duels(DUEL_FILE)
    key = parse_key(KEY_FILE)
    print(f"対 {len(duels)}件 / 鍵 {len(key)}件")

    missing = [d["id"] for d in duels if d["id"] not in key]
    if missing:
        raise SystemExit(f"鍵に無い対がある: {missing}")

    names = sorted({key[d["id"]]["A"] for d in duels}
                   | {key[d["id"]]["B"] for d in duels})
    tally = {name: 0 for name in names}
    draws = broken = unjudged = 0

    for duel in duels:
        entry = key[duel["id"]]
        if duel["choice"] in ("A", "B"):
            tally[entry[duel["choice"]]] += 1
        elif duel["choice"] == "=":
            draws += 1
        elif duel["choice"] == "×":
            broken += 1
        else:
            unjudged += 1
        duel["winner"] = (entry[duel["choice"]]
                          if duel["choice"] in ("A", "B") else duel["choice"])

    print("\n" + "=" * 56)
    print("勝敗")
    print("=" * 56)
    for name in names:
        print(f"  {name}: {tally[name]}勝")
    print(f"  引き分け {draws} / 両方崩壊 {broken} / 未判定 {unjudged}")

    if len(names) == 2:
        first, second = names
        wins, losses = tally[first], tally[second]
        winner, loser = ((first, second) if wins >= losses
                         else (second, first))
        high, low = max(wins, losses), min(wins, losses)
        print(f"\n符号検定 ({winner} が {loser} を {high}対{low}): "
              f"p={sign_test(high, low):.4f}")
        print(f"  勝敗のついた対 {high + low} / 全{len(duels)}対")

    print("\n" + "=" * 56)
    print("お題ごとの勝者")
    print("=" * 56)
    prompts = []
    for duel in duels:
        if duel["prompt"] not in prompts:
            prompts.append(duel["prompt"])
    for prompt in prompts:
        group = [d for d in duels if d["prompt"] == prompt]
        marks = " ".join(d.get("winner") or "?" for d in group)
        print(f"  {prompt[:22]:<24} {marks}")

    if all(key[d["id"]]["banned_a"] is not None for d in duels):
        print("\n" + "=" * 56)
        print("禁止語の総数")
        print("=" * 56)
        for name in names:
            total = sum(
                key[d["id"]]["banned_a" if key[d["id"]]["A"] == name
                             else "banned_b"]
                for d in duels)
            print(f"  {name}: {total}")


main()

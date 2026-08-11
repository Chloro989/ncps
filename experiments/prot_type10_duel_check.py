"""
センチュリオン: 会話の一騎打ちの答え合わせ (Phase 10)

Phase 9 と同じく対応表は鍵ファイルを読む。左右の入れ替えを乱数で
決めているので、生成順から再現するのは煩雑になる。

Phase 9 との違いは判定が2つあること。
  一貫性: 同じ一人の語り手として読めるのはどちらか(仮説の本体)
  面白さ: 轍から遠く、読み物として面白いのはどちらか(副作用の確認)

回路が一貫性で勝ち、面白さで負けていないことが狙い。
一貫性で勝って面白さで負けるなら、それは「安全になっただけ」で
発想の飛躍という目的からは後退になる。両方を見る理由がここにある。

あわせて、鍵に残した姿勢の並びから、回路が実際に漂ったかを数える。
受け入れ試験は代用の入力で測ったものなので、
本番の応答から作った特徴でも漂ったかを確かめる必要がある。
"""

import re
from math import comb
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
DUEL_FILE = RESULTS / "centurion_turn_duel.txt"
KEY_FILE = RESULTS / "centurion_turn_duel_key.txt"

BLOCK = re.compile(r"^={20,}$", re.MULTILINE)
HEAD = re.compile(r"^対(\d+)$", re.MULTILINE)
# [^\S\n] は改行を含まない空白。\s* だと空欄の判定が次の行を吸ってしまう
JUDGEMENTS = ["一貫性", "面白さ"]
KEY_HEAD = re.compile(r"^対(\d+): A=(\S+) B=(\S+)", re.MULTILINE)
KEY_SIDE = re.compile(r"^    ([AB]) \((\S+)\) 轍語(\d+)回$", re.MULTILINE)
KEY_STANCE = re.compile(r"^      (\d+) (共通|    ) (.+)$", re.MULTILINE)

LEFT = {"A", "a", "Ａ", "左"}
RIGHT = {"B", "b", "Ｂ", "右"}
DRAW = {"=", "＝", "同じ", "-"}
BROKEN = {"×", "x", "X", "✕"}


def classify(mark):
    return ("A" if mark in LEFT else "B" if mark in RIGHT
            else "=" if mark in DRAW else "×" if mark in BROKEN else "")


def parse_duels(path):
    duels = []
    for block in BLOCK.split(path.read_text(encoding="utf-8")):
        head = HEAD.search(block)
        if not head:
            continue
        entry = {"id": int(head.group(1))}
        for name in JUDGEMENTS:
            found = re.search(rf"^{name}:[^\S\n]*(.*)$", block, re.MULTILINE)
            mark = found.group(1).strip() if found else ""
            entry[name] = classify(mark)
            entry[name + "_生"] = mark
        duels.append(entry)
    return sorted(duels, key=lambda d: d["id"])


def parse_key(path):
    """対ごとに A/B の条件名、轍語の数、各ターンの姿勢を読む"""
    text = path.read_text(encoding="utf-8")
    starts = [(int(m.group(1)), m.start(), m.group(2), m.group(3))
              for m in KEY_HEAD.finditer(text)]
    key = {}
    for index, (number, start, left, right) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(text)
        body = text[start:end]
        entry = {"A": left, "B": right, "轍語": {}, "姿勢": {}}
        for side in re.split(r"(?=^    [AB] \()", body, flags=re.MULTILINE)[1:]:
            head = KEY_SIDE.search(side)
            if not head:
                continue
            entry["轍語"][head.group(2)] = int(head.group(3))
            entry["姿勢"][head.group(2)] = [
                (found.group(2).strip() == "共通", found.group(3).strip())
                for found in KEY_STANCE.finditer(side)]
        key[number] = entry
    return key


def sign_test(wins, losses):
    """片側の符号検定。Phase 9 と同じ統計にして比較できるようにする。
    仮説は方向付き(回路がランダムより一貫して読める)なので片側で見る"""
    total = wins + losses
    if total == 0:
        return float("nan")
    return sum(comb(total, k) for k in range(wins, total + 1)) / 2 ** total


def report(duels, key, name):
    names = sorted({key[d["id"]]["A"] for d in duels}
                   | {key[d["id"]]["B"] for d in duels})
    tally = {label: 0 for label in names}
    draws = broken = unjudged = 0
    winners = {}

    for duel in duels:
        entry = key[duel["id"]]
        choice = duel[name]
        if choice in ("A", "B"):
            tally[entry[choice]] += 1
            winners[duel["id"]] = entry[choice]
        else:
            winners[duel["id"]] = choice or "?"
            draws += choice == "="
            broken += choice == "×"
            unjudged += choice == ""

    print("\n" + "=" * 56)
    print(f"{name}")
    print("=" * 56)
    for label in names:
        print(f"  {label}: {tally[label]}勝")
    print(f"  引き分け {draws} / 両方崩壊 {broken} / 未判定 {unjudged}")

    if len(names) == 2:
        first, second = names
        high, low = max(tally[first], tally[second]), min(tally[first],
                                                          tally[second])
        winner = first if tally[first] >= tally[second] else second
        loser = second if winner == first else first
        print(f"  符号検定 ({winner} が {loser} を {high}対{low}): "
              f"片側 p={sign_test(high, low):.4f}")
        print(f"  勝敗のついた対 {high + low} / 全{len(duels)}対")

    marks = " ".join(f"{winners[d['id']][:2]}" for d in duels)
    print(f"  対ごとの勝者: {marks}")
    return tally


def stance_movement(duels, key):
    """鍵に残した姿勢の並びから、条件ごとの漂いを数える。
    本番の応答から作った特徴でも回路が漂ったかの確認"""
    print("\n" + "=" * 56)
    print("姿勢の動き (本番の応答から作った特徴で)")
    print("=" * 56)
    records = {}
    for duel in duels:
        for name, turns in key[duel["id"]]["姿勢"].items():
            moving = [text for shared, text in turns if not shared]
            record = records.setdefault(name, {"frozen": 0, "seen": set(),
                                               "shared": []})
            if moving and len(set(moving)) == 1:
                record["frozen"] += 1
            record["seen"].update(frozenset(t.split(" / ")) for t in moving)
            for before, after in zip(moving, moving[1:]):
                record["shared"].append(
                    len(set(before.split(" / ")) & set(after.split(" / "))))
    for name, record in sorted(records.items()):
        shared = (sum(record["shared"]) / len(record["shared"])
                  if record["shared"] else float("nan"))
        print(f"  {name}: 隣ターンの共有句 {shared:.2f} / "
              f"組み合わせ {len(record['seen'])}種 / "
              f"凍結した会話 {record['frozen']}本")
    print("  ランダム選択なら共有句は約0.45。回路がそれより明確に高ければ、"
          "設計どおり漂っている")


def rut_words(duels, key):
    print("\n" + "=" * 56)
    print("轍語の総数")
    print("=" * 56)
    totals = {}
    for duel in duels:
        for name, count in key[duel["id"]]["轍語"].items():
            totals[name] = totals.get(name, 0) + count
    for name, total in sorted(totals.items()):
        print(f"  {name}: {total}回")


def main():
    if not KEY_FILE.exists():
        raise SystemExit(f"{KEY_FILE.name} が無い。生成時の鍵ファイルが必要")

    duels = parse_duels(DUEL_FILE)
    key = parse_key(KEY_FILE)
    print(f"対 {len(duels)}件 / 鍵 {len(key)}件")

    missing = [d["id"] for d in duels if d["id"] not in key]
    if missing:
        raise SystemExit(f"鍵に無い対がある: {missing}")

    tallies = {name: report(duels, key, name) for name in JUDGEMENTS}
    stance_movement(duels, key)
    rut_words(duels, key)

    print("\n" + "=" * 56)
    print("まとめ")
    print("=" * 56)
    consistency, interest = (tallies[name] for name in JUDGEMENTS)
    if "回路" in consistency and "ランダム" in consistency:
        won = consistency["回路"] > consistency["ランダム"]
        kept = interest["回路"] >= interest["ランダム"]
        if won and kept:
            print("  回路が一貫性で勝ち、面白さでも負けていない。仮説どおり")
        elif won:
            print("  回路は一貫性で勝ったが面白さで負けた。"
                  "安定と引き換えに飛躍を失っている疑いがある")
        elif consistency["回路"] == consistency["ランダム"]:
            print("  一貫性に差がない。連続性は読み手に届いていない")
        else:
            print("  ランダムが一貫性で勝った。仮説は否定された")


if __name__ == "__main__":
    main()

"""
センチュリオン: 盲検の答え合わせ (Phase 5)
判定済みの centurion_eval.txt を読み、条件ごとの勝率を出す。

対応表は centurion_eval_key.txt があればそれを使い、無ければ
prot_type7_eval.py と同じ生成順とシャッフルの種から再現する。
再現が正しいかは、各標本のお題が一致するかで検証する —
36件すべてで一致すれば、順序の再現は間違いようがない。
"""

import random
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
EVAL_FILE = RESULTS / "centurion_eval.txt"

# prot_type7_eval.py と同じ設定。ずれると答え合わせが狂う
RUNS = 3
SHUFFLE_SEED = 20260809
CONDITIONS = ["未学習", "学習済み", "制御なし"]
USER_PROMPTS = [
    "青色にまつわる話を聞かせて",
    "朝の匂いについて書いて",
    "忘れられた道具の話をして",
    "冬の終わりをどう感じる",
]

BLOCK = re.compile(r"^-{20,}$", re.MULTILINE)
FIELD = {
    "id": re.compile(r"^標本(\d+)", re.MULTILINE),
    "prompt": re.compile(r"^お題: (.+)$", re.MULTILINE),
    "label": re.compile(r"^判定:\s*(.*)$", re.MULTILINE),
}
HIT = {"○", "〇", "◯", "o", "O"}
MISS = {"×", "✕", "✖", "✗", "x", "X"}


def parse_eval(path):
    rows = []
    for block in BLOCK.split(path.read_text(encoding="utf-8")):
        found = {k: p.search(block) for k, p in FIELD.items()}
        if not found["id"] or not found["prompt"]:
            continue
        mark = found["label"].group(1).strip() if found["label"] else ""
        rows.append({
            "id": int(found["id"].group(1)),
            "prompt": found["prompt"].group(1).strip(),
            # ○でも×でもない書き込みは未判定として扱う(誤入力を拾わない)
            "label": "○" if mark in HIT else "×" if mark in MISS else "",
            "mark": mark,
        })
    return sorted(rows, key=lambda r: r["id"])


def rebuild_key():
    """生成順とシャッフルの種から対応表を再現する"""
    order = [(name, prompt)
             for name in CONDITIONS
             for prompt in USER_PROMPTS
             for _ in range(RUNS)]
    random.Random(SHUFFLE_SEED).shuffle(order)
    return order


def main():
    rows = parse_eval(EVAL_FILE)
    key = rebuild_key()
    print(f"標本 {len(rows)}件 / 対応表 {len(key)}件")

    # お題が全件一致するかで、再現の正しさを確かめる
    mismatched = [r["id"] for r, (_, prompt) in zip(rows, key)
                  if r["prompt"] != prompt]
    if mismatched:
        raise SystemExit(f"再現に失敗。お題が一致しない標本: {mismatched}")
    print("検証: 36件すべてでお題が一致。対応表の再現は正しい\n")

    for row, (name, _) in zip(rows, key):
        row["condition"] = name

    odd = [(r["id"], r["mark"]) for r in rows
           if r["mark"] and not r["label"]]
    if odd:
        print(f"○×以外の書き込みは未判定として扱った: {odd}\n")

    print("=" * 56)
    print("条件ごとの結果")
    print("=" * 56)
    print(f"{'条件':<8}{'○':>5}{'×':>5}{'未判定':>7}{'○率':>9}")
    for name in CONDITIONS:
        group = [r for r in rows if r["condition"] == name]
        hit = sum(1 for r in group if r["label"] == "○")
        miss = sum(1 for r in group if r["label"] == "×")
        blank = len(group) - hit - miss
        rate = hit / (hit + miss) if hit + miss else float("nan")
        print(f"{name:<8}{hit:5d}{miss:5d}{blank:7d}{rate:8.0%}")

    print("\n" + "=" * 56)
    print("お題ごとの結果")
    print("=" * 56)
    print(f"{'お題':<16}{'○':>5}{'×':>5}{'未判定':>7}{'○率':>9}")
    for prompt in USER_PROMPTS:
        group = [r for r in rows if r["prompt"] == prompt]
        hit = sum(1 for r in group if r["label"] == "○")
        miss = sum(1 for r in group if r["label"] == "×")
        blank = len(group) - hit - miss
        rate = hit / (hit + miss) if hit + miss else float("nan")
        print(f"{prompt[:14]:<16}{hit:5d}{miss:5d}{blank:7d}{rate:8.0%}")

    print("\n" + "=" * 56)
    print("条件 × お題")
    print("=" * 56)
    header = "".join(f"{p[:6]:>9}" for p in USER_PROMPTS)
    print(f"{'条件':<8}{header}")
    for name in CONDITIONS:
        cells = []
        for prompt in USER_PROMPTS:
            group = [r for r in rows
                     if r["condition"] == name and r["prompt"] == prompt]
            hit = sum(1 for r in group if r["label"] == "○")
            miss = sum(1 for r in group if r["label"] == "×")
            cells.append(f"{hit}○{miss}×" if hit + miss else "-")
        print(f"{name:<8}" + "".join(f"{c:>9}" for c in cells))

    # 特に良いと言われた標本がどの条件だったか
    print("\n" + "=" * 56)
    print("特に良いとされた標本")
    print("=" * 56)
    for target in (8, 25, 29):
        row = next(r for r in rows if r["id"] == target)
        print(f"標本{target:02d} ({row['prompt']}): {row['condition']}")


main()

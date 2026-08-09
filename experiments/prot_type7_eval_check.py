"""
センチュリオン: 盲検の答え合わせ (Phase 5〜6)
判定済みの centurion_eval.txt を読み、条件ごとの成績を出す。

判定は3段階に対応する:
  ○ 読める、かつ轍から出ている
  △ 読めるが轍に沈んでいる
  × 崩壊している

対応表は prot_type7_eval.py / prot_type7_future.py と同じ生成順と
シャッフルの種から再現する。再現が正しいかは、各標本のお題が
一致するかで検証する — 全件で一致すれば、順序の再現は間違いようがない。
"""

import random
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
EVAL_FILE = RESULTS / "centurion_eval.txt"

# 条件と試行数は評価ファイルの見出しから読む。
# 手で同期させていると、条件を増やしたときに答え合わせが静かに狂う
SHUFFLE_SEED = 20260809
HEADER_CONDITIONS = re.compile(r"^条件\(順不同\): (.+)$", re.MULTILINE)
HEADER_RUNS = re.compile(r"^試行数: (\d+)$", re.MULTILINE)
FALLBACK_CONDITIONS = ["未学習", "学習済み", "制御なし"]
FALLBACK_RUNS = 3
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
# 〇(U+3007)と○(U+25CB)は見た目が同じで別の文字。両方受ける
HIT = {"○", "〇", "◯", "o", "O"}
MID = {"△", "▲", "~"}
MISS = {"×", "✕", "✖", "✗", "x", "X"}
LEVELS = ["○", "△", "×"]


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
            "label": ("○" if mark in HIT else "△" if mark in MID
                      else "×" if mark in MISS else ""),
            "mark": mark,
        })
    return sorted(rows, key=lambda r: r["id"])


def read_setup(path):
    """評価ファイルの見出しから条件と試行数を読む"""
    text = path.read_text(encoding="utf-8")
    found = HEADER_CONDITIONS.search(text)
    conditions = ([c.strip() for c in found.group(1).split("/")]
                  if found else FALLBACK_CONDITIONS)
    found = HEADER_RUNS.search(text)
    runs = int(found.group(1)) if found else FALLBACK_RUNS
    return conditions, runs


def rebuild_key(conditions, runs):
    """生成順とシャッフルの種から対応表を再現する"""
    order = [(name, prompt)
             for name in conditions
             for prompt in USER_PROMPTS
             for _ in range(runs)]
    random.Random(SHUFFLE_SEED).shuffle(order)
    return order


def tally(rows):
    counts = {level: sum(1 for r in rows if r["label"] == level)
              for level in LEVELS}
    counts["未"] = sum(1 for r in rows if not r["label"])
    judged = sum(counts[level] for level in LEVELS)
    counts["脱出率"] = counts["○"] / judged if judged else float("nan")
    counts["崩壊率"] = counts["×"] / judged if judged else float("nan")
    return counts


def report(title, groups):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'':<16}{'○':>4}{'△':>4}{'×':>4}{'未':>4}"
          f"{'脱出率':>9}{'崩壊率':>9}")
    for name, rows in groups.items():
        c = tally(rows)
        print(f"{name[:14]:<16}{c['○']:4d}{c['△']:4d}{c['×']:4d}{c['未']:4d}"
              f"{c['脱出率']:9.0%}{c['崩壊率']:9.0%}")


def main():
    rows = parse_eval(EVAL_FILE)
    conditions, runs = read_setup(EVAL_FILE)
    key = rebuild_key(conditions, runs)
    print(f"条件 {' / '.join(conditions)} / 試行数 {runs}")
    print(f"標本 {len(rows)}件 / 対応表 {len(key)}件")

    mismatched = [r["id"] for r, (_, prompt) in zip(rows, key)
                  if r["prompt"] != prompt]
    if mismatched:
        raise SystemExit(f"再現に失敗。お題が一致しない標本: {mismatched}")
    print(f"検証: {len(rows)}件すべてでお題が一致。対応表の再現は正しい")

    for row, (name, _) in zip(rows, key):
        row["condition"] = name

    odd = [(r["id"], r["mark"][:20]) for r in rows if r["mark"] and not r["label"]]
    if odd:
        print(f"○△×以外の書き込みは未判定として扱った: {odd}")

    report("条件ごとの結果",
           {name: [r for r in rows if r["condition"] == name]
            for name in conditions})
    report("お題ごとの結果",
           {prompt: [r for r in rows if r["prompt"] == prompt]
            for prompt in USER_PROMPTS})

    print("\n" + "=" * 60)
    print("条件 × お題 (○△×)")
    print("=" * 60)
    header = "".join(f"{p[:6]:>10}" for p in USER_PROMPTS)
    print(f"{'':<16}{header}")
    for name in conditions:
        cells = []
        for prompt in USER_PROMPTS:
            group = [r for r in rows
                     if r["condition"] == name and r["prompt"] == prompt]
            c = tally(group)
            cells.append(f"{c['○']}/{c['△']}/{c['×']}")
        print(f"{name[:14]:<16}" + "".join(f"{c:>10}" for c in cells))


main()

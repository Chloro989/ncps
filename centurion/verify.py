"""
出てきた指摘を、別のモデルに検証させる。

## なぜ要るか

機械の照合で捕まるのは「引用が本文と食い違う」までで、
**引用は正しいが判断が的外れ**な指摘は素通りする。
Qwen2.5-3B の実測では、引用14件のうち8件が取り違えだったが、
残る6件が良い指摘だという保証はどこにも無かった。

## どう組むか

Phase 9 で分かったのは、**絶対評価(○△×)は基準が回ごとに19pt動いて
使い物にならない**が、**一騎打ち(二つ並べてどちらかを選ぶ)なら
迷いのない判定が出る**ということだった。

ここでは二択にする。一件ずつ「残す」か「捨てる」かだけを選ばせ、
点数を付けさせない。

そして**疑う側に立たせる**。「良いところを探して」と頼めば全部が残る。
「反証を試み、できなければ残す」と頼めば、通ったものだけが残る。
迷ったら捨てるのが既定であることも明示する。

## 何を捨てさせるか

- 本文に無いことを言っている
- 本文にはあるが、読み違えている
- 何をどう直すのか分からない (「もっと丁寧に」の類)
- 代償を書いていない
- 他の指摘と同じことを言っている
"""

import re
from dataclasses import dataclass

from .answer import CITATION, RANGE, attach

# 「3: 捨てる 理由」の形。番号のあとの語だけを見る
VERDICT = re.compile(r"^\s*\[?(\d+)\]?\s*[:：.、]\s*(.+)$", re.MULTILINE)
KEEP_WORDS = ("残す", "支持", "妥当", "採用", "有効", "正しい")
DROP_WORDS = ("捨てる", "却下", "的外れ", "不採用", "無効", "誤り", "曖昧")


@dataclass
class Finding:
    """答えから切り出した一つの指摘"""
    number: int             # 検証で使う通し番号 (1から)
    text: str
    targets: list           # 指している段落

    @property
    def where(self):
        return ("段落 " + "・".join(f"[{n}]" for n in self.targets)
                if self.targets else "段落の指定なし")


def split_findings(answer, manuscript, min_length=12):
    """答えを一件ずつの指摘に切り分ける。

    行を単位にする。見出しだけの行や、短すぎて中身のない行は落とす —
    「### 熱量」や「**提案:**」を検証させても意味がない"""
    total = len(manuscript.paragraphs)
    found = []
    for line in answer.splitlines():
        stripped = line.strip()
        if len(stripped) < min_length:
            continue
        if stripped.startswith("#") or set(stripped) <= set("-=*_ "):
            continue
        spans = RANGE.findall(stripped)
        if spans:
            targets = [int(first) for first, _ in spans]
        else:
            targets = [int(n) for n in CITATION.findall(stripped)]
        targets = sorted({n for n in targets if 0 <= n < total})
        found.append(Finding(len(found) + 1, stripped, targets))
    return found


def build_prompt(manuscript, findings, body_text, title=""):
    """検証させるための指示と本文を組む。(指示, 本文) を返す"""
    head = [
        "あなたは編集会議で、他人が出した指摘を検分する側にいる。",
        "書き手の味方ではなく、指摘の粗を探す側に立つこと。",
        "",
        "一件ずつ「残す」か「捨てる」かだけを選ぶ。点数は付けない。",
        "**迷ったら捨てる。** 通らなかった指摘が混ざるより、"
        "確かなものだけが残るほうが役に立つ。",
        "",
        "次のどれかに当たれば捨てる。",
        "- 本文に書かれていないことを、書かれているかのように言っている",
        "- 本文にはあるが、読み違えている",
        "- 何をどう直すのか分からない (「もっと丁寧に」の類)",
        "- その変更で何を失うかを書いていない",
        "- 他の指摘と同じことを言っている",
        "",
        "答え方: 一行に一件、次の形だけを書く。前置きも総括も書かない。",
        "  番号: 残す  そう判断した理由を一文で",
        "  番号: 捨てる  そう判断した理由を一文で",
    ]
    body = []
    if title:
        body.append(f"作品: 「{title}」")
    body.append("")
    body.append("=== 本文 ===")
    body.append(body_text)
    body.append("")
    body.append("=== 検分する指摘 ===")
    for item in findings:
        body.append(f"{item.number}: ({item.where}) {item.text}")
    return "\n".join(head), "\n".join(body)


def parse_verdicts(text, findings):
    """判定を読む。読めなかった番号は「判定なし」として残す —
    検証が働かなかったことを、捨てたことにすり替えない"""
    verdicts = {}
    for found in VERDICT.finditer(text or ""):
        number = int(found.group(1))
        rest = found.group(2).strip()
        head = rest[:24]
        if any(word in head for word in DROP_WORDS):
            verdicts[number] = (False, rest)
        elif any(word in head for word in KEEP_WORDS):
            verdicts[number] = (True, rest)
    return {item.number: verdicts.get(item.number) for item in findings}


def sift(findings, verdicts):
    """(残った, 捨てた, 判定されなかった) に分ける"""
    kept, dropped, unjudged = [], [], []
    for item in findings:
        verdict = verdicts.get(item.number)
        if verdict is None:
            unjudged.append((item, ""))
        elif verdict[0]:
            kept.append((item, verdict[1]))
        else:
            dropped.append((item, verdict[1]))
    return kept, dropped, unjudged


def rebuild(kept, unjudged):
    """残った指摘だけで答えを組み直す。
    判定されなかったものは、印を付けて残す"""
    lines = [item.text for item, _ in kept]
    lines += [f"{item.text}  (検証で判定されなかった)"
              for item, _ in unjudged]
    return "\n".join(lines)


def report(kept, dropped, unjudged):
    lines = [f"検証: {len(kept)}件を残し、{len(dropped)}件を捨てた"
             + (f"、{len(unjudged)}件は判定されなかった" if unjudged else "")]
    for item, why in dropped:
        lines.append(f"  捨: {item.text[:38]}… — {why[:40]}")
    return "\n".join(lines)

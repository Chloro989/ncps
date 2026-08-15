"""
指摘を検分にかける仕組みの試験。

    python tests/test_verify.py

機械の照合で捕まるのは「引用が本文と食い違う」までで、
引用は正しいが判断が的外れな指摘は素通りする。そこを別のモデルに見せる。

Phase 9 の教訓に従い、点数ではなく一件ずつの二択にしてある。
そして疑う側に立たせる — 迷ったら捨てるのが既定。
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion.manuscript import Manuscript
from centurion.verify import (build_prompt, parse_verdicts, rebuild, report,
                              sift, split_findings)

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


work = Manuscript("\n\n".join([
    "　改札を抜けると、売店の灯りがまだついていた。",
    "　雨は上がっていたが、路面には水たまりが残っていた。",
    "　彼女は「行こう」と言って、先に歩き出した。",
    "　橋の上で立ち止まり、水路の暗がりを覗きこんだ。",
]))

ANSWER = """### 熱量

- [0] の売店の灯りは、あとで効く伏線になっていない。二度目を置くべきだ。
- [1] と [3] の水は同じものを指しているのに、繋がりが書かれていない。
**提案:**
- [2] の会話はもっと丁寧に書くとよい。
"""

print("== 指摘の切り分け ==")
findings = split_findings(ANSWER, work)
check("見出しは落とす", all("###" not in f.text for f in findings))
check("短すぎる行は落とす", all(len(f.text) >= 12 for f in findings))
check("中身のある指摘だけ残る", len(findings) == 3, str(len(findings)))
check("通し番号が1から付く", [f.number for f in findings] == [1, 2, 3])
check("指している段落を持つ", findings[0].targets == [0],
      str(findings[0].targets))
check("複数の段落も拾う", findings[1].targets == [1, 3],
      str(findings[1].targets))
check("段落の場所を言葉にできる", "[0]" in findings[0].where)
check("段落を指さない指摘も扱える",
      split_findings("全体として、語り口が一定していない。", work)[0].targets
      == [])
check("範囲の指定は先頭を取る",
      split_findings("[0] ~ [3]: 前半の流れについて述べる。", work)[0].targets
      == [0])

print("\n== 検分の指示 ==")
head, body = build_prompt(work, findings, "本文のところ", title="試作")
check("疑う側に立たせる", "粗を探す側" in head)
check("二択にする", "「残す」か「捨てる」" in head and "点数は付けない" in head)
check("迷ったら捨てると明示する", "迷ったら捨てる" in head)
check("捨てる基準を並べる", head.count("- ") >= 5)
check("代償を書かない指摘を捨てる基準がある", "何を失うか" in head)
check("答え方を指定する", "番号: 残す" in head)
check("本文が入る", "本文のところ" in body)
check("題が入る", "「試作」" in body)
check("指摘が番号付きで入る", "1: (段落 [0])" in body, body[-200:])

print("\n== 判定を読む ==")
verdicts = parse_verdicts(
    "1: 残す  本文の通りで、直し方も具体的\n"
    "2: 捨てる  そんな繋がりは本文に無い\n"
    "3: 捨てる  何をどう直すのか分からない", findings)
check("残すを読む", verdicts[1][0] is True)
check("捨てるを読む", verdicts[2][0] is False)
check("理由も持つ", "本文に無い" in verdicts[2][1])
check("全部の番号に答えが付く", len(verdicts) == 3)

check("角括弧つきでも読む",
      parse_verdicts("[1]: 残す 妥当", findings)[1][0] is True)
check("全角のコロンでも読む",
      parse_verdicts("1：捨てる 的外れ", findings)[1][0] is False)
check("言い換えも読む",
      parse_verdicts("1: 却下 読み違えている", findings)[1][0] is False)

# 検証が働かなかったことを、捨てたことにすり替えない
verdicts = parse_verdicts("1: 残す よい", findings)
check("答えの無い番号は判定なしにする",
      verdicts[2] is None and verdicts[3] is None)
check("答えが空でも落ちない",
      all(v is None for v in parse_verdicts("", findings).values()))
check("形が崩れていても落ちない",
      all(v is None for v in parse_verdicts("よく分かりません", findings).values()))

print("\n== ふるい分け ==")
kept, dropped, unjudged = sift(findings, parse_verdicts(
    "1: 残す よい\n2: 捨てる 本文に無い", findings))
check("残ったものを分ける", [f.number for f, _ in kept] == [1])
check("捨てたものを分ける", [f.number for f, _ in dropped] == [2])
check("判定されなかったものを分ける", [f.number for f, _ in unjudged] == [3])

built = rebuild(kept, unjudged)
check("残った指摘が入る", findings[0].text in built)
check("捨てた指摘は入らない", findings[1].text not in built)
check("判定されなかったものは印を付けて残す",
      "検証で判定されなかった" in built and findings[2].text in built)

summary = report(kept, dropped, unjudged)
check("件数を数える", "1件を残し、1件を捨てた" in summary, summary[:60])
check("判定されなかった数も出す", "1件は判定されなかった" in summary)
check("捨てた理由を出す", "本文に無い" in summary)

print("\n== 端の場合 ==")
check("指摘が無ければ空", split_findings("", work) == [])
check("見出しだけなら空", split_findings("### 熱量\n**提案:**", work) == [])
check("全部捨てられても組み直せる",
      rebuild([], []) == "")

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

"""
モデルの答えを受け取って、確かめて、読める形に直す。

## 引用の照合が要る理由

段落番号の実在だけを見る検査は弱い。Qwen2.5-3B に「あっちゃぐり」
(104段落)を読ませた答えでは、番号はすべて実在し検査は素通りしたが、
引用14件のうち8件が別の段落の中身を指していた。
うち3件は本文に存在しない文だった —
「帰る途中、エーマートの冷蔵庫の扉を開けると…」のような、
それらしいが書かれていない一文をモデルが作っている。

番号と中身の両方を突き合わせれば、これは機械で捕まる。
「引用が実在するか確認せよ」とモデルに自己申告させても、
申告する当人が取り違えているので効かない。

## 添削ファイルを作る理由

指摘が並んだ文章だけを渡されても、原稿のどこの話か分からない。
本文の各段落の下に、その段落あての指摘を貼り付ければ、
上から読むだけで直せる。照合に落ちた指摘には印を付けて、
信じてよいものと捨てるべきものを分けて見せる。
"""

import re
from dataclasses import dataclass

CITATION = re.compile(r"\[(\d+)\]")
# 「[1] ~ [56]」のような範囲指定。両端に同じ指摘を貼ると二重になるので、
# 先頭にだけ貼って範囲であることを添える
RANGE = re.compile(r"\[(\d+)\]\s*[~〜～\-–—]\s*\[(\d+)\]")
HEAD = 14              # 引用の頭から何文字を突き合わせるか

# これより短い鉤括弧は本文の引用とみなさない。
# 日本語では語をそのまま「あっちゃぐり」のように括るので、
# 短いものまで引用として突き合わせると、正しい指摘に×が付く。
# 実際にそれが起きた — 作中語を括っただけの指摘4件のうち3件に
# 誤って×を付けていた。見落とすほうが、良い指摘を捨てるより安全
QUOTE_MIN = 10

MARK_OK = "▸"
MARK_BAD = "×"


@dataclass
class Quote:
    """答えの中の一つの引用"""
    number: int             # 示された段落番号
    text: str               # 引用された本文
    ok: bool                # その段落の中身と一致したか
    home: list              # 一致しなかったとき、本当の出どころ
    line: str               # その引用があった行

    @property
    def invented(self):
        """本文のどこにも無い。モデルが作った文"""
        return not self.ok and not self.home

    def __str__(self):
        if self.ok:
            return f"○ [{self.number}] {self.text[:HEAD]}…"
        if self.invented:
            return (f"× [{self.number}] 本文に無い文を引用している: "
                    f"{self.text[:HEAD]}…")
        return (f"× [{self.number}] 引用は [{self.home[0]}] の中身: "
                f"{self.text[:HEAD]}…")


def find_quotes(answer, manuscript):
    """[番号] と「引用」が同じ行にある箇所を拾って、突き合わせる。

    括弧の入れ子(「…「あっちゃぐり」…」)があるので、
    最初の「から行末の最後の」までを一つの引用として取る"""
    total = len(manuscript.paragraphs)
    found = []
    for line in answer.splitlines():
        numbers = [int(n) for n in CITATION.findall(line)]
        if not numbers:
            continue
        start = line.find("「")
        end = line.rfind("」")
        if start < 0 or end <= start:
            continue
        quoted = line[start + 1:end].strip()
        if len(quoted) < QUOTE_MIN:
            continue          # 語を括っただけ。引用ではない
        # 観点の名前(【熱量】など)を括弧で括っただけの行は本文引用ではない
        if "【" in quoted or "】" in quoted:
            continue
        head = quoted[:HEAD]
        number = numbers[0]
        actual = (manuscript.paragraphs[number].text
                  if 0 <= number < total else "")
        home = [p.index for p in manuscript.paragraphs if head in p.text]
        found.append(Quote(number, quoted, head in actual, home,
                           line.strip()))
    return found


def attach(answer, manuscript):
    """答えの各行を、宛先の段落に振り分ける。
    番号を含まない行は前置きとしてまとめる。

    「[1] ~ [56]」のような範囲は先頭にだけ貼る。
    両端に貼ると、離れた二箇所に同じ指摘が現れて紛らわしい"""
    total = len(manuscript.paragraphs)
    preamble, notes = [], {}
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        spans = RANGE.findall(stripped)
        if spans:
            targets = {int(first) for first, _ in spans}
        else:
            targets = {int(n) for n in CITATION.findall(stripped)}
        inside = sorted(n for n in targets if 0 <= n < total)
        if not inside:
            preamble.append(stripped)
            continue
        for number in inside:
            notes.setdefault(number, []).append(stripped)
    return preamble, notes


def annotate(answer, manuscript, label="", lenses="", width=68):
    """本文の各段落の下に、その段落あての指摘を貼った文章を作る。

    照合に落ちた指摘には印を付ける。信じてよいものと捨てるものを
    分けて見せないと、添削として使えない"""
    quotes = find_quotes(answer, manuscript)
    bad_lines = {q.line for q in quotes if not q.ok}
    preamble, notes = attach(answer, manuscript)

    out = []
    head = f"# {manuscript.title or '原稿'} の添削"
    if manuscript.author:
        head += f" ({manuscript.author})"
    out.append(head)
    if label:
        out.append(f"# {label}")
    if lenses:
        out.append(f"# 観点: {lenses}")

    ok = sum(1 for q in quotes if q.ok)
    if quotes:
        out.append(f"# 引用の照合: {len(quotes)}件中 一致{ok}件 / "
                   f"不一致{len(quotes) - ok}件")
        invented = [q for q in quotes if q.invented]
        if invented:
            out.append(f"# うち{len(invented)}件は本文に無い文の引用。"
                       f"{MARK_BAD} の付いた指摘は捨てること")
    out.append("")

    if preamble:
        out.append("## 段落を指していない指摘")
        out.extend("  " + line for line in preamble)
        out.append("")

    out.append("## 本文と指摘")
    out.append("")
    for paragraph in manuscript.paragraphs:
        out.append(f"[{paragraph.index}] {paragraph.text}")
        for line in notes.get(paragraph.index, []):
            mark = MARK_BAD if line in bad_lines else MARK_OK
            out.append(f"    {mark} {line}")
        out.append("")
    return "\n".join(out)


def report_quotes(quotes):
    """照合の結果を、人が読む形にまとめる"""
    if not quotes:
        return "引用が見当たらない。段落番号だけで指摘している"
    ok = [q for q in quotes if q.ok]
    bad = [q for q in quotes if not q.ok]
    lines = [f"引用の照合 {len(quotes)}件中 一致{len(ok)}件 / "
             f"不一致{len(bad)}件"]
    lines.extend("  " + str(q) for q in bad)
    invented = [q for q in bad if q.invented]
    if invented:
        lines.append(f"  うち{len(invented)}件は本文に存在しない文。"
                     "モデルが作っている")
    return "\n".join(lines)

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


def annotate(answer, manuscript, records=(), axes=()):
    """本文の各段落の下に、その段落あての指摘を貼った文章を作る。

    照合に落ちた指摘には印を付ける。信じてよいものと捨てるものを
    分けて見せないと、添削として使えない。

    records は (見出し, 中身) の並び。何のモードで、どのモデルに
    解かせた添削なのかが残っていないと、溜まった添削を並べて比べられない"""
    quotes = find_quotes(answer, manuscript)
    bad_lines = {q.line for q in quotes if not q.ok}
    preamble, notes = attach(answer, manuscript)

    out = []
    head = f"# {manuscript.title or '原稿'} の添削"
    if manuscript.author:
        head += f" ({manuscript.author})"
    out.append(head)
    for name, value in records:
        if value:
            out.append(f"# {name}: {value}")

    anchored, total = anchoring(answer, manuscript)
    if total:
        for line in report_anchoring(anchored, total).splitlines():
            out.append("# " + line.strip())

    ok = sum(1 for q in quotes if q.ok)
    if quotes:
        out.append(f"# 引用の照合: {len(quotes)}件中 一致{ok}件 / "
                   f"不一致{len(quotes) - ok}件")
        invented = [q for q in quotes if q.invented]
        if invented:
            out.append(f"# うち{len(invented)}件は本文に無い文の引用。"
                       f"{MARK_BAD} の付いた指摘は捨てること")
    elif total:
        out.append("# 引用が一つも無い。照合できるものが無いということでもある")

    # 採点なら点数を検算する。合計の書き間違いは実際に起きている
    if axes:
        for line in check_scores(answer, axes):
            out.append("# " + line.strip())
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


def anchoring(answer, manuscript, min_length=12):
    """指摘のうち、原稿のどこかを指しているものの割合。

    引用の照合は「引用があるもの」しか見られない。
    どこも指さない指摘ばかりの答えは、照合を素通りしたまま
    「不一致0件」と表示されて、良い答えのように見えてしまう。

    実際にそうなった — 4件の指摘がどれも段落を指さず、
    「静寂と微細な血流の揺らぎ」のような本文に無い言葉で埋まっていたのに、
    照合は「1件中 一致1件」とだけ出ていた。

    (錨のある数, 指摘の総数) を返す"""
    total = anchored = 0
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or set(stripped) <= set("-=*_ "):
            continue
        numbers = [int(n) for n in CITATION.findall(stripped)]
        points = any(0 <= n < len(manuscript.paragraphs) for n in numbers)
        # 番号を示している行は、短くても指摘として数える。
        # 「[12] 冗長。」は言葉少なだが、行き先のある指摘である
        if not points and len(stripped) < min_length:
            continue
        total += 1
        anchored += points
    return anchored, total


def report_anchoring(anchored, total, floor=0.5):
    """錨の少なさを伝える。少なければ、その答えは使えない"""
    if not total:
        return ""
    share = anchored / total
    line = f"原稿を指している指摘 {anchored}/{total}件 ({share:.0%})"
    if share < floor:
        line += ("\n  半分も原稿を指していない。"
                 "本文から離れた一般論になっている疑いが強い。"
                 "\n  観点を減らすか、より大きいモデルに解かせること")
    return line


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


# ===== 採点の検算 =====
# 実測「運命の九十分」を LFM2.5-1.2B に採点させたところ、
# 4+3+5+5+4+5+4 = 30 を「合計点 16/35」と書いた。
# 引用の照合と同じで、これは機械が確かめられる。
# モデルが強くなっても算術は外すので、検査は残す価値がある

# モデルは見出しの書き方を揃えてくれない。実際に出た形:
#   ## 1. 構成・プロット: 4/5      ### 3. 文章・文体：5/5
#   **1. 構成・プロット**: 4/5     1．構成・プロット: 4 / 5
# 取りこぼすと検算が黙って素通りするので、書き方の揺れは広く許す。
# 行頭の空白に \s を使わないのは、re.M でも改行をまたがせないため
SCORE_LINE = re.compile(
    r"^[ \t　]*#{0,4}[ \t　]*[*_]{0,2}[ \t　]*(\d+)[ \t　]*[.．][ \t　]*"
    r"([^:：\n]+?)[ \t　]*[:：][ \t　]*(\d+)[ \t　]*[/／][ \t　]*5",
    re.M)
TOTAL_LINE = re.compile(r"合計点[^0-9\n]{0,6}(\d+)[ \t　]*[/／][ \t　]*35")


def scores(answer):
    """採点の答えから (番号, 観点, 点数) を取り出す"""
    return [(int(number), axis.strip(" 　*_"), int(point))
            for number, axis, point in SCORE_LINE.findall(answer)]


def claimed_total(answer):
    """モデルが書いた合計点。書いていなければ None"""
    found = TOTAL_LINE.search(answer)
    return int(found.group(1)) if found else None


def check_scores(answer, axes):
    """採点の答えを検算する。人が読む行の並びを返す。

    見るのは三つ。観点が抜けていないか、点数が範囲内か、合計が合うか。
    どれも本文を読まずに確かめられるものだけにしてある"""
    found = scores(answer)
    if not found:
        return ["採点の形になっていない。"
                "「## 1. 構成・プロット: 4/5」の形で出させること"]

    lines = []
    given = [point for _, _, point in found]
    lines.append(f"採点 {len(found)}観点 / 合計 {sum(given)}点")

    named = [axis for _, axis, _ in found]
    missing = [axis for axis in axes
               if not any(axis in name or name in axis for name in named)]
    if missing:
        lines.append(f"  抜けている観点 {len(missing)}件: "
                     f"{'、'.join(missing)}")
    if len(found) > len(axes):
        lines.append(f"  観点が多い。{len(axes)}個のはずが{len(found)}個ある")

    outside = [(axis, point) for _, axis, point in found
               if not 1 <= point <= 5]
    for axis, point in outside:
        lines.append(f"  {axis} が範囲外の {point}点")

    total = claimed_total(answer)
    if total is None:
        lines.append("  合計点が書かれていない")
    elif total != sum(given):
        lines.append(f"  合計が合わない。書かれた合計は{total}点だが、"
                     f"各観点の和は{sum(given)}点")
    return lines

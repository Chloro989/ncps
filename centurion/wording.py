"""
プロンプトの文面を外部ファイルから読む。

## なぜ

観点や規則の文面は、使ってみないと良し悪しが分からない。
「本文に無い要素について述べない」の一行を足すか外すかで出力は変わる。
その調整のたびにソースを書き換えるのでは、試すのに手間がかかりすぎるし、
更新のたびに手元の変更が消える。

そこで、文面だけを prompts/ に外へ出す。

    python main.py prompts        既定の文面を prompts/ に書き出す
    (prompts/査読-厳格.txt を編集)
    python main.py ask 原稿.txt --mode 査読 --severity 厳格

prompts/ に該当ファイルがあればそれが使われ、無ければ組み込みの既定値が
使われる。壊したら消せば元に戻る。

## ファイル

    発想.txt          役割と規則 (発想モード)
    査読-育成.txt     役割と規則 (査読モード・厳しさごと)
    査読-標準.txt
    査読-厳格.txt
    採点-育成.txt     役割と規則とルーブリック (採点モード)
    採点-標準.txt
    採点-厳格.txt
    観点.txt          観点の一覧

観点.txt だけは形式がある。1行1観点で、縦棒区切り。

    キー | 群 | 使えるモード | 問い

使えるモードは 発想 と 査読 をカンマで並べる。問いは複数行にできない。
"""
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent / "prompts"

LENS_FILE = "観点.txt"
SEPARATOR = "|"

# 観点.txt の先頭に置く説明。読み飛ばされる
LENS_HEADER = """\
# 観点の一覧。1行1観点。
#
#   キー | 群 | 使えるモード | 問い
#
# キー   出力に【削除】のように現れる短い名前
# 群     --spread で角度を散らすときの分類。同じ群からは1つしか選ばれない
# モード 発想,査読 のように並べる。片方だけでもよい
# 問い   モデルに渡す一文
#
# 提案を求める観点 (「〜を挙げよ」で本文に無いものを求めるもの) に
# 査読 を付けると、査読の規則「本文に無い要素について述べない」と
# 矛盾する。main.py test で検査している。
#
# # で始まる行と空行は読み飛ばす。
"""


def home(directory=None):
    return Path(directory) if directory else HOME


def path(name, directory=None):
    return home(directory) / f"{name}.txt"


def read(name, directory=None):
    """prompts/ の文面を読む。無ければ None"""
    target = path(name, directory)
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8-sig").strip()
    return text or None


def heading(name, fallback, directory=None):
    """外部ファイルがあればそれを、無ければ組み込みの既定値を返す"""
    return read(name, directory) or fallback


def parse_lenses(text):
    """観点.txt を (キー, 群, モードの組, 問い) の並びにする"""
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(SEPARATOR)]
        if len(parts) != 4:
            raise ValueError(
                f"{LENS_FILE} の {number} 行目: "
                f"縦棒で4つに区切る必要がある (今は{len(parts)}つ)\n  {line}")
        key, group, modes, question = parts
        if not key or not group or not question:
            raise ValueError(
                f"{LENS_FILE} の {number} 行目: 空の欄がある\n  {line}")
        kinds = tuple(m.strip() for m in modes.split(",") if m.strip())
        if not kinds:
            raise ValueError(
                f"{LENS_FILE} の {number} 行目: 使えるモードが空\n  {line}")
        rows.append((key, group, kinds, question))
    if not rows:
        raise ValueError(f"{LENS_FILE} に観点が1つも無い")
    return rows


def load_lenses(directory=None):
    """観点.txt があれば読む。無ければ None"""
    text = read(LENS_FILE[:-4], directory)
    return parse_lenses(text) if text else None


def format_lenses(lenses):
    """観点を観点.txt の形式にする。書き出し用"""
    lines = [LENS_HEADER]
    group = None
    for lens in lenses:
        if lens.group != group:
            group = lens.group
            lines.append("")
        lines.append(f"{lens.key} | {lens.group} | "
                     f"{','.join(lens.modes)} | {lens.question}")
    return "\n".join(lines) + "\n"


def export(defaults, directory=None, force=False):
    """既定の文面を prompts/ に書き出す。
    既にあるファイルは force=True でなければ触らない。
    (書いた, 飛ばした) を返す"""
    folder = home(directory)
    folder.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for name, text in defaults.items():
        target = folder / f"{name}.txt"
        if target.exists() and not force:
            skipped.append(target)
            continue
        if not text.endswith("\n"):
            text += "\n"
        target.write_text(text, encoding="utf-8")
        written.append(target)
    return written, skipped

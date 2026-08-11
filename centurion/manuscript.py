"""
原稿を読んで、章・段落・文に分け、モデルに渡せる大きさへ切り分ける。

ここにモデルは出てこない。どのモデルを使うかと独立に必要な土台なので、
torch も transformers も読み込まない。手元のCPUだけで全部確かめられる。

校正AIでは**指摘が原稿のどこを指すかが本体**になる。
「三章の二段落目、132文字目から」と言えなければ直しようがない。
そのため切り出した塊は、章・段落・原稿全体での文字位置を持ち歩く。
"""

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# 日本語の原稿でよくある文字コード。上から順に試す
ENCODINGS = ["utf-8-sig", "utf-8", "cp932", "euc-jp"]

# 章の見出しらしい行。短い行にしか当てない —
# 本文中の「第一章を書き終えた日のことだ」を見出しと誤認しないため
HEADING_MAX = 40
HEADING = re.compile(
    r"^\s*(?:"
    r"#{1,6}\s+"                                   # マークダウンの見出し
    r"|第\s*[0-9０-９一二三四五六七八九十百千]+\s*[章話部節篇編巻]"
    r"|[0-9０-９]{1,3}\s*[.．、]?\s*$"
    r"|序章?|終章|最終章|プロローグ|エピローグ|序文|あとがき|まえがき"
    r")")
MARKDOWN_HASH = re.compile(r"^\s*#{1,6}\s+")

# 場面の切れ目に使われる記号だけの行
SEPARATOR = re.compile(r"^\s*[＊*＄$※#＃◆◇■□●○▲△・…‥\-—―ー~〜]{1,}\s*$")

# 青空文庫の注記。ルビ・傍点・入力者注
RUBY = re.compile(r"[｜|]?([^｜|《》]+)《[^》]*》")
ANNOTATION = re.compile(r"［＃[^］]*］|\[#[^\]]*\]")

# 青空文庫の付属物。本文の前後に付く、作品ではない部分
AOZORA_MARKS = ("【テキスト中に現れる記号について】", "青空文庫",
                "底本：", "底本:")
AOZORA_RULE = re.compile(r"^-{10,}\s*$")
AOZORA_FOOTER = re.compile(r"^(?:底本[：:]|入力[：:]|校正[：:]|"
                           r"青空文庫作成ファイル|"
                           r"※この作品|このファイルは、インターネットの図書館)")

# 文の終わり。閉じ括弧が続くならそこまでを一文にする
SENTENCE_END = re.compile(r"[。！？!?]+[」』）\)】〉》”\"]*")


def read_text(path, strip_ruby=True):
    """原稿を読む。文字コードは上から順に試す"""
    raw = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"文字コードが読めない: {path}")
    return clean(text, strip_ruby)


def looks_like_aozora(text):
    head = text[:2000]
    return any(mark in head or mark in text[-2000:] for mark in AOZORA_MARKS)


def strip_aozora(text):
    """青空文庫の付属物を落とし、(本文, 題, 著者) を返す。

    この形式は本文の前後に作品でないものが付く。
    凡例と底本の記録をそのまま校正に回すと、指摘がそこに向かってしまう"""
    lines = text.split("\n")

    # 前置き: 罫線で挟まれた凡例。無いこともある
    rules = [i for i, line in enumerate(lines) if AOZORA_RULE.match(line)]
    head_end = rules[1] + 1 if len(rules) >= 2 else 0

    # 題と著者は罫線より前の、中身のある行
    heading = [line.strip() for line in lines[:rules[0] if rules else 4]
               if line.strip()]
    title = heading[0] if heading else ""
    author = heading[1] if len(heading) > 1 else ""

    # 後書き: 底本の記録から後ろを全部落とす
    body_end = len(lines)
    for index in range(head_end, len(lines)):
        if AOZORA_FOOTER.match(lines[index].strip()):
            body_end = index
            break

    return "\n".join(lines[head_end:body_end]).strip("\n"), title, author


def clean(text, strip_ruby=True):
    """改行を揃え、必要ならルビと注記を落とす。
    文字位置がずれるので、切り分ける前に一度だけ通すこと"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_ruby:
        text = ANNOTATION.sub("", text)
        text = RUBY.sub(r"\1", text)
    # 行末の空白だけ落とす。行頭の全角空白は段落の印なので残す
    return "\n".join(line.rstrip() for line in text.split("\n"))


def split_sentences(text):
    """文に分ける。「〜。」のように閉じ括弧が続く場合はそこまでを一文にする"""
    sentences, start = [], 0
    for found in SENTENCE_END.finditer(text):
        sentences.append(text[start:found.end()])
        start = found.end()
    if text[start:].strip():
        sentences.append(text[start:])
    return [s for s in sentences if s.strip()]


def width(text):
    """全角を2、半角を1として数えた見た目の長さ。
    日本語の原稿では文字数より実感に近い"""
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1
               for c in text)


@dataclass
class Paragraph:
    """段落。原稿全体での位置を持つ"""
    index: int                  # 原稿全体での通し番号
    chapter: int                # 属する章の番号
    start: int                  # 原稿全体での文字位置
    text: str

    @property
    def end(self):
        return self.start + len(self.text)

    def sentences(self):
        return split_sentences(self.text)

    def locate(self, offset):
        """段落内の位置を、原稿全体での位置に直す"""
        return self.start + offset


@dataclass
class Chapter:
    index: int
    title: str                  # 見出しが無ければ空
    paragraphs: list = field(default_factory=list)

    @property
    def text(self):
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def start(self):
        return self.paragraphs[0].start if self.paragraphs else 0

    def __len__(self):
        return sum(len(p.text) for p in self.paragraphs)


@dataclass
class Chunk:
    """モデルに渡す一塊。どこから来たかを必ず持つ"""
    index: int
    paragraphs: list
    carried: int = 0            # 先頭のうち、前の塊と重なっている段落数

    @property
    def text(self):
        return "\n\n".join(p.text for p in self.paragraphs)

    @property
    def body(self):
        """重なりを除いた、この塊が担当する部分"""
        return "\n\n".join(p.text for p in self.paragraphs[self.carried:])

    @property
    def chapter(self):
        return self.paragraphs[0].chapter if self.paragraphs else 0

    @property
    def start(self):
        return self.paragraphs[0].start if self.paragraphs else 0

    @property
    def end(self):
        return self.paragraphs[-1].end if self.paragraphs else 0

    @property
    def span(self):
        """担当する段落の通し番号の範囲"""
        owned = self.paragraphs[self.carried:] or self.paragraphs
        return owned[0].index, owned[-1].index

    def __len__(self):
        return len(self.text)

    def __str__(self):
        first, last = self.span
        return (f"塊{self.index}: {self.chapter}章 "
                f"段落{first}〜{last} ({len(self)}文字"
                + (f"、うち重なり{self.carried}段落)" if self.carried else ")"))


class Manuscript:
    """原稿。章と段落に分かれていて、好きな大きさに切り出せる"""

    def __init__(self, text, title="", author=""):
        if looks_like_aozora(text):
            text, aozora_title, aozora_author = strip_aozora(text)
            title = aozora_title or title
            author = aozora_author or author
        self.title = title
        self.author = author
        self.text = text
        self.chapters = []
        self.paragraphs = []
        self._split()

    @classmethod
    def load(cls, path, strip_ruby=True):
        path = Path(path)
        return cls(read_text(path, strip_ruby), title=path.stem)

    def _paragraph_lines(self):
        """段落の区切り方を原稿から決める。

        空行で区切る書き方と、1行1段落で書く書き方(web小説に多い)がある。
        空行が少なければ後者とみなす"""
        lines = self.text.split("\n")
        if not lines:
            return []
        blank = sum(1 for line in lines if not line.strip())
        by_blank = blank / len(lines) > 0.15

        blocks, buffer = [], []
        for line in lines:
            if not line.strip():
                if buffer:
                    blocks.append("\n".join(buffer))
                    buffer = []
                continue
            if by_blank:
                buffer.append(line)
            else:
                blocks.append(line)
        if buffer:
            blocks.append("\n".join(buffer))
        return blocks

    def _split(self):
        chapter = Chapter(index=0, title="")
        self.chapters = [chapter]
        position = 0

        for block in self._paragraph_lines():
            # 本文での位置を取り直す。clean() が行末を揃えているので
            # 塊はそのままの形で本文に現れる
            found = self.text.find(block, position)
            start = found if found >= 0 else position
            position = start + len(block)

            # 段落の本文からは行頭の全角空白を落とすが、位置はその分ずらす。
            # ここを合わせないと「何文字目」がすべて字下げ分だけ狂う
            stripped = block.strip()
            start += len(block) - len(block.lstrip())
            is_heading = (len(stripped) <= HEADING_MAX
                          and (HEADING.match(stripped)
                               or SEPARATOR.match(stripped)))
            if is_heading:
                title = MARKDOWN_HASH.sub("", stripped).strip() or stripped
                # 本文より前に来たマークダウンの見出しは、章ではなく原稿の題。
                # ファイル名より、書いてある題のほうが確かなので上書きする。
                # 「第一章」の類はこれに当てはめない — それは章の見出し
                if (not self.paragraphs and not chapter.title
                        and MARKDOWN_HASH.match(stripped)):
                    self.title = title
                    continue
                # 中身のある章のあとでだけ章を切る。
                # 見出しが連続しても空の章を作らない
                if chapter.paragraphs:
                    chapter = Chapter(index=len(self.chapters), title=title)
                    self.chapters.append(chapter)
                else:
                    chapter.title = title
                continue

            paragraph = Paragraph(index=len(self.paragraphs),
                                  chapter=chapter.index,
                                  start=start, text=stripped)
            chapter.paragraphs.append(paragraph)
            self.paragraphs.append(paragraph)

        # 中身のない章は落とす
        self.chapters = [c for c in self.chapters if c.paragraphs]
        for number, chapter in enumerate(self.chapters):
            for paragraph in chapter.paragraphs:
                paragraph.chapter = number
            chapter.index = number

    def sentences(self):
        return [s for p in self.paragraphs for s in p.sentences()]

    def chunks(self, size=6000, overlap=1, measure=len,
               respect_chapters=True):
        """モデルに渡せる大きさに切り分ける。

        size      1塊の上限。measure で測った値
        overlap   前の塊の末尾から何段落を持ち越すか。
                  境界で文脈が切れると、そこの指摘が的外れになる
        measure   大きさの測り方。トークンで測るなら
                  measure=lambda t: len(tokenizer(t).input_ids) を渡す
        respect_chapters
                  章をまたがない。章の切れ目は本当の断絶なので既定は True
        """
        chunks, current, carried = [], [], 0

        def flush():
            nonlocal current, carried
            if current:
                chunks.append(Chunk(len(chunks), list(current), carried))
                current, carried = [], 0

        for paragraph in self.paragraphs:
            crossing = (respect_chapters and current
                        and paragraph.chapter != current[-1].chapter)
            too_big = (current
                       and measure("\n\n".join(p.text for p in current)
                                   + "\n\n" + paragraph.text) > size)
            if crossing or too_big:
                tail = current[-overlap:] if overlap and not crossing else []
                flush()
                current = list(tail)
                carried = len(tail)
            current.append(paragraph)

            # 1段落だけで上限を超える場合は、そのまま1塊にして次へ
            if len(current) == 1 and measure(paragraph.text) > size:
                flush()
        flush()
        return chunks

    def summary(self):
        head = f"「{self.title}」" if self.title else "原稿"
        if self.author:
            head += f"  {self.author}"
        lines = [head,
                 f"  {len(self.text)}文字 / 見た目 {width(self.text)} / "
                 f"{len(self.chapters)}章 / {len(self.paragraphs)}段落 / "
                 f"{len(self.sentences())}文"]
        for chapter in self.chapters:
            head = chapter.title or "(見出しなし)"
            lines.append(f"  {chapter.index:>3}. {head[:24]:<26}"
                         f"{len(chapter):>7}文字 "
                         f"{len(chapter.paragraphs):>4}段落")
        return "\n".join(lines)


def main(argv=None):
    """python -m centurion.manuscript 原稿.txt [--size 6000]"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="centurion.manuscript",
        description="原稿を読んで、章・段落・切り出しの様子を見る")
    parser.add_argument("path", help="原稿のファイル")
    parser.add_argument("--size", type=int, default=6000,
                        help="1塊の上限文字数 (既定 6000)")
    parser.add_argument("--overlap", type=int, default=1,
                        help="持ち越す段落数 (既定 1)")
    parser.add_argument("--keep-ruby", action="store_true",
                        help="ルビと注記を落とさない")
    parser.add_argument("--show", type=int, default=0,
                        help="指定した塊の中身を表示する")
    args = parser.parse_args(argv)

    manuscript = Manuscript.load(args.path, strip_ruby=not args.keep_ruby)
    print(manuscript.summary())

    chunks = manuscript.chunks(size=args.size, overlap=args.overlap)
    print(f"\n{args.size}文字ずつに切ると {len(chunks)}塊")
    for chunk in chunks:
        print("  " + str(chunk))

    if args.show:
        chunk = chunks[args.show - 1]
        print(f"\n--- {chunk} ---")
        print(chunk.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

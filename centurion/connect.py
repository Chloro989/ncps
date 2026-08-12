"""
遠い二つを繋ぐ。発想の中身をここに置く。

## なぜこれが要るのか

Phase 5〜10 で5つの機構を作って、どれも跳躍を生まなかった。
共通していたのは、**一本の生成を横へずらそうとしていた**ことである。
抑圧も、未来エントロピーも、表現への介入も、姿勢の漂いも、
一点をどちらへ動かすかの話だった。

跳躍は一点の摂動ではなく、二点の関係である。
ずらしても、繋ぐ相手がいなければ跳ばない。
夢が繋ぐのは二つのものであり、連想も二つ以上の項を持つ。

だから相手を用意する。しかも**原稿の中から**取る。
外から奇抜な要素を持ち込めば、提案は奇を衒ったものになる。
[12]の止まった時計と[87]の折れた傘の骨を繋げと言えば、
提案は作者自身の世界の内側に留まる。

プロンプトには対を選べない。原稿を全部持っているこちらには選べる。
ここが、道具である意味になる。

## 距離の測り方について

判明8 で「埋め込み距離では跳躍を測れない」と結論している。
ただしあれは**生成された文章が跳んだかを測ろうとして**失敗したもので、
ここでの用途は違う — **入力として渡す対を選ぶ**ための距離である。
選んだ結果が良いかは人が判定するので、指標が跳躍を測れる必要はない。

測り方も埋め込みではなく、位置の隔たりと語彙の重なりで足りる。
説明できるほうが、外れたときに直せる。
"""

import random
import re
from dataclasses import dataclass

# 内容語らしきもの。漢字2文字以上、またはカタカナ2文字以上。
# 助詞や活用語尾を拾わないための粗い近似で、形態素解析は使わない —
# 依存を増やさずに済み、外れ方も読める
WORD = re.compile(r"[一-龥々]{2,}|[ァ-ヴー]{2,}")

# 対を選ぶときの既定。
# 隔たりは原稿全体に対する割合。近すぎる二つは繋げても当たり前になる。
#
# 語彙の重なりは、実測すると絞り込みとしてほとんど働かない —
# 日本語の短い段落は隣り合っていても重なりが平均0.026しかなく、
# 遠い対はたいてい0になる。効いているのは位置の隔たりのほうである。
# 重なりの上限は「すでに本文で繋がっている二つ」を外すための保険として残す
MIN_GAP = 0.2
MAX_OVERLAP = 0.12
MIN_LENGTH = 30        # 短い段落は繋ぐ手がかりが足りない

# 隔たりの実距離の下限。割合だけで測ると短い作品で破綻する。
# 太宰治「I can speak」(1949文字・15段落)では、「酔漢」が[8]と[12]に出て
# 隔たり27%と表示されたが、実際には4段落しか離れておらず、
# しかも同じ場面の同じ人物だった。繋がりはすでに本文にあり、架ける橋がない。
#
# 遠いと言えるためには、読者がいったん忘れる程度の間が要る。
# 段落数ではなく文字数で測る — 会話文と描写で段落の長さが桁違いに違うため
MIN_CHARS = 1500

# 一度きりの語を拾うのに必要な原稿の長さ。
# 4500文字の抜粋では221語が該当し、証明・歴史・二日目まで拾ってしまう。
# 短い原稿では「一度きり」がほとんどの語に当てはまり、意味を成さない
ONCE_MIN_TEXT = 20000

# 反復と主題の境目。参考作品では「虫干」が2回で伏線、
# 「レース」が工場・夢・遺品にまたがって10回以上出る主題だった。
# どちらも要るが、読み方が違うので分けて示す
MOTIF_TIMES = 4
MAX_TIMES = 14

# 遠くに散っていても繋がりの手がかりにならない一般語。
# 実測で上位に来てしまったもの(正面・最初・何度・一回・立派…)から作った。
# 形態素解析を入れずに済ませるための、割り切った小さな一覧
GENERIC = {
    "自分", "相手", "最初", "最後", "今度", "今日", "昨日", "明日",
    "本当", "普通", "一番", "一回", "一度", "何度", "何時", "時間",
    "場所", "部分", "全部", "以上", "以下", "程度", "様子", "感じ",
    "気持", "言葉", "意味", "理由", "必要", "問題", "結局", "実際",
    "正面", "反対", "立派", "大丈夫", "無理", "邪魔", "普段",
    "人間", "誰か", "二人", "三人", "自身", "彼女", "彼等",
}


def content_words(text):
    return set(WORD.findall(text))


def overlap(left, right):
    """語彙の重なり。0なら共通の語が無い"""
    a, b = content_words(left), content_words(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Pair:
    """繋ぐ相手として選ばれた二つの段落"""
    left: object
    right: object
    gap: float          # 位置の隔たり(原稿全体に対する割合)
    overlap: float      # 語彙の重なり

    @property
    def crosses_chapter(self):
        return self.left.chapter != self.right.chapter

    @property
    def chars(self):
        """二つの間に挟まっている文字数"""
        return max(self.right.start - self.left.end, 0)

    def __str__(self):
        return (f"[{self.left.index}] × [{self.right.index}] "
                f"(隔たり{self.gap:.0%}・{self.chars}文字 "
                f"/ 重なり{self.overlap:.0%}"
                + ("、章をまたぐ)" if self.crosses_chapter else ")"))


def distant_pairs(manuscript, count=5, rng=None, min_gap=MIN_GAP,
                  max_overlap=MAX_OVERLAP, min_length=MIN_LENGTH,
                  min_chars=MIN_CHARS):
    """遠く、かつ語彙の重ならない段落の対を選ぶ。

    遠いだけでは足りない。語彙が重なっていれば、繋がりはもう本文にある。
    重なっていない二つを繋ぐとき、橋は書かれていないものになる"""
    rng = rng or random
    usable = [p for p in manuscript.paragraphs if len(p.text) >= min_length]
    total = len(manuscript.paragraphs)
    if len(usable) < 2 or total < 2:
        return []

    candidates = []
    for index, left in enumerate(usable):
        for right in usable[index + 1:]:
            gap = abs(right.index - left.index) / total
            if gap < min_gap:
                continue
            if right.start - left.end < min_chars:
                continue          # 割合では遠いが、実距離が近い
            shared = overlap(left.text, right.text)
            if shared > max_overlap:
                continue
            candidates.append(Pair(left, right, gap, shared))

    if not candidates:
        return []
    rng.shuffle(candidates)
    # 同じ段落ばかり出ないよう、一度使った段落は避ける
    chosen, used = [], set()
    for pair in candidates:
        if pair.left.index in used or pair.right.index in used:
            continue
        chosen.append(pair)
        used.update((pair.left.index, pair.right.index))
        if len(chosen) == count:
            break
    # それでも足りなければ、重複を許して埋める
    for pair in candidates:
        if len(chosen) == count:
            break
        if pair not in chosen:
            chosen.append(pair)
    return chosen


@dataclass
class Recurrence:
    """稀な語が、離れた場所で二度以上使われている箇所。
    作者がすでに植えた種であり、繋がりの候補として最も確度が高い"""
    word: str
    paragraphs: list        # その語が現れる段落
    gap: float              # 最初と最後の隔たり

    @property
    def times(self):
        return len(self.paragraphs)

    @property
    def chars(self):
        """最初と最後の間に挟まっている文字数"""
        return max(self.paragraphs[-1].start - self.paragraphs[0].end, 0)

    @property
    def kind(self):
        """回数で性質が変わる。少なければ仕掛け、多ければ作品の背骨。
        参考作品では「虫干」が2回で伏線、「レース」が10回で主題だった"""
        return "反復" if self.times <= MOTIF_TIMES else "主題"

    def pair(self):
        """最も離れた二つを対にする"""
        return Pair(self.paragraphs[0], self.paragraphs[-1], self.gap,
                    overlap(self.paragraphs[0].text,
                            self.paragraphs[-1].text))

    def __str__(self):
        places = "・".join(f"[{p.index}]" for p in self.paragraphs[:6])
        if self.times > 6:
            places += f"…他{self.times - 6}箇所"
        return (f"《{self.kind}》「{self.word}」{places} "
                f"(隔たり{self.gap:.0%}・{self.chars}文字)")


def recurrences(manuscript, min_gap=MIN_GAP, max_times=MAX_TIMES, min_word=2,
                min_chars=MIN_CHARS):
    """遠く離れて繰り返される稀な語を探す。

    正解の分かっている作品で確かめたところ、作者が架けた橋のうち
    語彙で繋がるものは、まさにこの形をしていた —
    「コスモス」が夢[17]と幼年時代[24]に、
    「レース」が機械[27]と遺品[62]に。

    無作為に対を選ぶ方式ではこれを引けなかった(候補1165対から6個)。
    稀な語の反復は、候補を桁違いに絞る強い手がかりになる。

    max_times   これより多く出る語は、繋がりというより地の文の常用語になる。
                ただし絞りすぎない — 参考作品では「レース」が
                工場・夢・遺品にまたがって何度も出ており、
                回数の多さこそが主題の背骨を示していた
    """
    total = len(manuscript.paragraphs)
    if total < 2:
        return []

    where = {}
    for paragraph in manuscript.paragraphs:
        for word in content_words(paragraph.text):
            if len(word) >= min_word:
                where.setdefault(word, []).append(paragraph)

    found = []
    for word, paragraphs in where.items():
        if not 2 <= len(paragraphs) <= max_times:
            continue
        if word in GENERIC:
            continue
        gap = (paragraphs[-1].index - paragraphs[0].index) / total
        if gap < min_gap:
            continue
        if paragraphs[-1].start - paragraphs[0].end < min_chars:
            continue              # 割合では遠いが、実距離が近い
        found.append(Recurrence(word, paragraphs, gap))

    # 語の長さを先に見る。隔たりだけで並べると、
    # たまたま遠くに散った「正面」「最初」のような一般語が上に来た。
    # 長い語のほうが具体的で、繋がりの手がかりになりやすい
    found.sort(key=lambda r: (-len(r.word), -r.gap, r.times))
    return found


def once_only(manuscript, min_length=2):
    """原稿全体で一度しか出てこない内容語と、その段落。

    一度きりの事物は、二度目を置ける場所である。
    繋ぐ相手として最も効きやすい素材でもある"""
    where, counts = {}, {}
    for paragraph in manuscript.paragraphs:
        for word in content_words(paragraph.text):
            counts[word] = counts.get(word, 0) + paragraph.text.count(word)
            where.setdefault(word, []).append(paragraph.index)
    return sorted(
        ((word, where[word][0]) for word, count in counts.items()
         if count == 1 and len(word) >= min_length),
        key=lambda item: item[1])


# 夢の作業を、原稿に対する操作に置き換えたもの。
# 「独創的に」と頼まず、手を動かせる形にする
DREAM_WORK = [
    ("圧縮",
     "二つを一つにまとめる。人物・場所・場面・物のうち二つを選び、"
     "同一のものだったとしたら何が起きるかを述べよ。"
     "まとめることで浮くもの、失われるものを両方書け。"),
    ("移動",
     "感情の重心を、本来それを担うべき対象から、"
     "その場にある些末な物へ移す。どの感情をどの物へ移すか、"
     "移した結果その場面がどう読めるようになるかを述べよ。"),
    ("視覚化",
     "この範囲にある抽象的な観念を一つ選び、"
     "手で触れられる物・見える出来事に置き換えよ。"
     "置き換えたものが作中の他の何と響くかまで述べよ。"),
    ("後付け",
     "二つの出来事の間に、本文には無い因果を通してみよ。"
     "その因果が成り立つために、どこに何を足す必要があるかを述べよ。"),
]

# 連想の鎖。一歩ずつは近いのに、着く先は遠い
CHAIN_STEPS = 4


def build_connection_prompt(manuscript, pair, note="", extra=()):
    """遠い二つの段落を並べて、橋を架けさせる。(指示, 本文) を返す"""
    head = [
        "あなたは書き手の伴走者である。"
        "作品を評価せず、この原稿がまだ繋いでいないものを繋ぐ。",
        "",
        "遠く離れた二つの箇所を渡す。この二つの間に、"
        "本文にはまだ無い関係を作れないかを考える。",
        "",
        "守ること:",
        "- 提案は、この二つの箇所にすでにある要素から導く。"
        "外から新しい設定を持ち込まない。",
        "- 繋がりを作るために本文へ何を足し、何を削るのかを、"
        "段落番号 [12] を示して書く。",
        "- その繋がりによって何を失うかを併記する。",
        "- 繋がらないと判断したなら、そう書いてよい。"
        "無理に繋げた案は使えない。",
        "- 作品を褒めない。総括や励ましを書かない。",
        "",
        f"[{pair.left.index}] と [{pair.right.index}] は"
        f"原稿の{pair.gap:.0%}ぶん離れており、"
        + ("章も異なる。" if pair.crosses_chapter else "同じ章にある。")
        + ("共通する語はほとんど無い。"
           if pair.overlap < 0.05 else ""),
    ]
    if extra:
        head.append("")
        head.append("次の観点も併せて使う。")
        for key, question in extra:
            head.append(f"- 【{key}】{question}")

    body = []
    if manuscript.title:
        body.append(f"作品: 「{manuscript.title}」")
    body.append("")
    body.append(f"[{pair.left.index}] {pair.left.text}")
    body.append("")
    body.append(f"[{pair.right.index}] {pair.right.text}")
    if note:
        body.append("")
        body.append(f"作者からの補足: {note}")
    return "\n".join(head), "\n".join(body)


def build_chain_prompt(manuscript, paragraph, steps=CHAIN_STEPS, note=""):
    """連想の鎖。一歩ずつ近いものを辿らせ、着いた先を作品へ戻させる。

    直接「遠いものを出せ」と言うと、遠さのための遠さが返る。
    近い一歩を重ねさせると、着く先は遠いのに道が通っている"""
    head = [
        "あなたは書き手の伴走者である。",
        "",
        f"渡した箇所にある事物から始めて、連想を{steps}歩たどる。",
        "一歩ごとの繋がりは近くてよい。ありふれた連想でかまわない。",
        "ただし同じ場所へ戻らず、毎回そこから離れること。",
        "",
        "守ること:",
        f"- {steps}歩をすべて書き出し、"
        "各歩でなぜそこへ移ったのかを一行で添える。",
        "- 最後に、たどり着いたものをこの作品へ戻す。"
        "どの段落 [12] のどこに、どう置くかを書く。",
        "- 戻せないなら戻せないと書く。無理に戻した案は使えない。",
        "- 作品を褒めない。",
    ]
    body = []
    if manuscript.title:
        body.append(f"作品: 「{manuscript.title}」")
    body.append("")
    body.append(f"[{paragraph.index}] {paragraph.text}")
    if note:
        body.append("")
        body.append(f"作者からの補足: {note}")
    return "\n".join(head), "\n".join(body)

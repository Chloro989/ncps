"""
原稿を読ませるための問いと、その組み立て。

ここにもモデルは出てこない。どのモデルを使うかと独立に決まる部分で、
手元のCPUだけで確かめられる。

## なぜモードを分けるか

査読と発想は、規則が逆を向く。

  査読 … 作品に無いものを言ってはいけない。幻覚を防ぐための正しい規則
  発想 … 作品に無いものを言うのが仕事。書かれなかった道を示すのだから

一つのプロンプトに両方を入れると、防御側の規則が発想側を黙らせる。
だから分ける。査読で足場を作り、発想でそこから離れる。

## なぜ「独創的に」と頼まないか

Phase 8〜9 の実測では、抽象的な指示は効かず具体的な操作が効いた
(「感覚を書け」50% 対 「名前のある場所か人を置く」78%)。
「独創的な意見を」と頼めば平板な答えが返る。
だから観点は願望ではなく**操作**として書く —
「消したら何が壊れるか」「読者の予想を三つ挙げて全部外せ」。

## なぜ観点を回すか

指示を増やすと一つひとつが薄まる(3Bで実測、判明16)。
一度に2〜3の観点だけを渡し、読み直すたびに入れ替える。
一回ごとの答えは深くなり、何回か通せば全体を覆える。
ただしこの効果は3Bで測ったもので、強いモデルでは薄まる可能性がある。
"""

import random
import re
from dataclasses import dataclass

# 段落の指し方。モデルにはこの番号で答えさせ、実在するかを機械が検査する。
# 「引用が実在するか確認せよ」と自己申告させるより確実
CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Lens:
    """一つの観点。願望ではなく、実行できる操作として書く"""
    key: str
    group: str
    question: str


# 観点の一覧。群ごとに性質が違うので、選ぶときは群を散らす
LENSES = [
    # --- 構造を疑う。あるものを取り去って、何が支えていたかを見る
    Lens("削除", "構造",
         "この範囲から一つの場面か人物か段落を消したとき、"
         "何が壊れるかを述べよ。壊れないものがあれば、"
         "それは何のためにそこにあるのかを問え。"),
    Lens("順序", "構造",
         "この範囲の出来事を別の順に並べ替えたとき、"
         "読者が受け取るものがどう変わるかを述べよ。"
         "現在の順序が最善でない可能性を具体的に検討せよ。"),
    Lens("一度きり", "構造",
         "この範囲に一度しか出てこない事物・仕草・言葉を挙げよ。"
         "そのうち二度目があれば効くものはどれか、"
         "二度目をどこに置くかを述べよ。"),
    Lens("重心", "構造",
         "この範囲で最も紙幅を割かれている対象と、"
         "作品にとって最も重要と思われる対象を、それぞれ挙げよ。"
         "その二つがずれているなら、ずれの意味を述べよ。"),

    # --- 予想を外す。安全な選択を見つけて、そこを揺らす
    Lens("予想", "逸脱",
         "この範囲を読んだ読者が次に来ると考える展開を三つ挙げよ。"
         "その三つすべてを外し、かつ理屈の通る道を一つ示せ。"),
    Lens("安全", "逸脱",
         "この範囲で作者が選んだ最も安全な選択はどれか。"
         "それを裏切った場合に何が起きるかを述べよ。"),
    Lens("約束事", "逸脱",
         "この作品が属すると思われる型の約束事を挙げよ。"
         "そのうちまだ破られていないものを指摘し、"
         "破るとしたらどこが要点かを述べよ。"),
    Lens("既視", "逸脱",
         "この範囲で既視感のある型を名指しせよ。"
         "その型が使われている理由を推し量ったうえで、"
         "同じ役割を果たす別の形を示せ。"),

    # --- 視点を移す。同じ出来事を別の位置から見る
    Lens("視点", "視点",
         "この範囲の場面を別の人物の目で語った場合、"
         "何が見えるようになり、何が見えなくなるかを述べよ。"),
    Lens("時制", "視点",
         "この範囲を回想として、あるいはずっと後から振り返る形で"
         "語った場合に何が変わるかを述べよ。"),
    Lens("信頼", "視点",
         "語り手が意図的にか無自覚にか、"
         "事実と違うことを語っている可能性のある箇所を探せ。"
         "無ければ、そう読ませる余地を作れる箇所を挙げよ。"),

    # --- 書かれなかったもの。ここが発想モードの本体
    Lens("分岐", "不在",
         "この範囲で作者が選ばなかったと思われる道を三つ挙げよ。"
         "それぞれが物語をどこへ運ぶかを、一つずつ書け。"),
    Lens("沈黙", "不在",
         "この範囲で人物が言わなかったこと、"
         "語り手が触れなかったことを挙げよ。"
         "その沈黙が働いているか、単なる不足かを判じよ。"),
    Lens("欠落", "不在",
         "この作品に現れていない要素で、"
         "入れれば効くと考えられるものを挙げよ。"
         "なぜ効くのか、どこに入れるのかまで述べよ。"),

    # --- 書き手の熱量。文章の温度差を読む
    Lens("熱量", "熱量",
         "作者が明らかに書きたくて書いた箇所と、"
         "必要だから書いた箇所を分けよ。"
         "後者を減らすか、前者に変える道を示せ。"),
    Lens("密度", "熱量",
         "描写の細かい箇所と粗い箇所を挙げ、"
         "その配分が場面の重要さと合っているかを述べよ。"),

    # --- 具体の検査。抽象に逃げていないかを見る
    Lens("固有", "具体",
         "名前を与えられている事物と、"
         "無名のままの事物を分けよ。"
         "無名のうち、名前を持つべきものを挙げよ。"),
    Lens("感覚", "具体",
         "この範囲で使われている感覚と、使われていない感覚を挙げよ。"
         "欠けている感覚を入れるとしたらどこかを示せ。"),
]

LENS_BY_KEY = {lens.key: lens for lens in LENSES}
GROUPS = sorted({lens.group for lens in LENSES})

# 査読モード。note などで共有されている審査プロンプトと同じ思想。
# 幻覚と忖度を防ぐための規則で固める
REVIEW_RULES = [
    "日本語の文芸作品として読み、他の言語の基準を持ち込まない。",
    "作者への配慮や忖度をしない。好悪の感情で評価しない。",
    "指摘は必ず段落番号 [12] の形で示す。番号は渡された本文のものだけを使う。",
    "本文に無い要素について述べない。",
    "作品の「言葉にならない魅力」や読者の内面については述べない。"
    "確かめられないことを書けば、それは作り話になる。",
]

# 発想モード。査読と逆を向く規則。
# 「本文に無い要素を述べない」を外さないと、書かれなかった道を示せない
IDEA_RULES = [
    "本文に無いものを述べてよい。それがこの作業の目的である。",
    "ただし提案は必ず本文の一箇所に錨を下ろす。"
    "段落番号 [12] を示し、そこから何をどう変えるのかを書く。",
    "提案には必ず、それによって何を失うかを併記する。"
    "代償を書かない提案は採用の判断ができない。",
    "「もっと丁寧に」「深みを出す」のような、"
    "実行できない助言を書かない。手を動かせる操作として書く。",
    "作品を褒めない。褒める作業はここではしない。",
    "奇抜さのための奇抜さを出さない。"
    "提案は、本文にすでにある要素から導けるものにする。",
]


def choose_lenses(rng=None, count=3, spread=True, groups=None):
    """観点を選ぶ。読み直すたびに入れ替えるためのもの。

    spread=True なら別々の群から選ぶ。同じ群の観点は似た答えを生むので、
    一度の読みで角度を散らしたい"""
    rng = rng or random
    pool = [l for l in LENSES if groups is None or l.group in groups]
    if not spread:
        return rng.sample(pool, min(count, len(pool)))

    by_group = {}
    for lens in pool:
        by_group.setdefault(lens.group, []).append(lens)
    order = list(by_group)
    rng.shuffle(order)

    chosen = []
    while len(chosen) < min(count, len(pool)):
        progressed = False
        for group in order:
            if not by_group[group]:
                continue
            chosen.append(by_group[group].pop(
                rng.randrange(len(by_group[group]))))
            progressed = True
            if len(chosen) == min(count, len(pool)):
                break
        if not progressed:
            break
    return chosen


def number_paragraphs(paragraphs, mark=None):
    """段落に番号を振って本文にする。
    mark に段落番号の集合を渡すと、その段落に印を付ける
    (持ち越した部分と、今回見てほしい部分を区別するため)"""
    lines = []
    for paragraph in paragraphs:
        prefix = "＞" if mark is not None and paragraph.index not in mark else ""
        lines.append(f"[{paragraph.index}] {prefix}{paragraph.text}")
    return "\n".join(lines)


def build_prompt(chunk, lenses, mode="発想", title="", author="",
                 note="", place=""):
    """一回ぶんの読みを組み立てる。(指示, 本文) を返す。

    chunk   manuscript.chunks() が返す塊
    lenses  この回で使う観点
    mode    査読 か 発想
    note    作者からの補足。狙いや訊きたいこと
    place   原稿全体の中でこの塊がどこかの説明
    """
    rules = REVIEW_RULES if mode == "査読" else IDEA_RULES
    role = ("あなたは日本の文芸作品を専門とする分析者である。"
            "小説家・批評家・編集者の視点を併せ持つ。"
            if mode == "査読" else
            "あなたは書き手の伴走者である。"
            "作品を評価するのではなく、この原稿がまだ行っていない先を探す。")

    head = [role, ""]
    head.append("守ること:")
    head.extend(f"- {rule}" for rule in rules)
    head.append("")
    head.append(f"今回の観点は次の{len(lenses)}つだけである。"
                "これ以外のことは書かない。")
    for index, lens in enumerate(lenses, 1):
        head.append(f"{index}. 【{lens.key}】{lens.question}")
    head.append("")
    head.append("観点ごとに見出しを立てて答える。"
                "全体のまとめや励ましは書かない。")

    body = []
    if title:
        body.append(f"作品: 「{title}」" + (f"  {author}" if author else ""))
    if place:
        body.append(f"範囲: {place}")
    if chunk.carried:
        body.append(f"先頭の{chunk.carried}段落は前の範囲との重なりで、"
                    "文脈のために付けてある。指摘は ＞ の付いていない段落に絞る。")
    body.append("")
    mark = {p.index for p in chunk.paragraphs[chunk.carried:]}
    body.append(number_paragraphs(chunk.paragraphs, mark))
    if note and mode != "査読":
        body.append("")
        body.append(f"作者からの補足: {note}")
    return "\n".join(head), "\n".join(body)


def citations(text):
    """答えの中で示された段落番号"""
    return [int(number) for number in CITATION.findall(text)]


def check_citations(text, manuscript, allowed=None):
    """示された段落番号が実在するかを検査する。

    査読プロンプトの多くは「引用が実在するか確認せよ」とモデルに
    自己申告させるが、番号で示させれば機械が確かめられる。
    (実在した番号, 存在しない番号, 範囲外の番号) を返す"""
    total = len(manuscript.paragraphs)
    real, missing, outside = [], [], []
    for number in citations(text):
        if not 0 <= number < total:
            missing.append(number)
        elif allowed is not None and number not in allowed:
            outside.append(number)
        else:
            real.append(number)
    return real, missing, outside


def resolve(text, manuscript, length=28):
    """答えの中の [12] を、その段落の冒頭に置き換えて読めるようにする"""
    def replace(found):
        number = int(found.group(1))
        if not 0 <= number < len(manuscript.paragraphs):
            return f"[{number}?]"
        body = manuscript.paragraphs[number].text[:length]
        return f"[{number}「{body}…」]"
    return CITATION.sub(replace, text)

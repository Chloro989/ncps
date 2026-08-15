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

import math
import random
import re
import statistics
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

    # --- 調子。実測「運命の九十分」で、18の観点のどれも
    #     喜劇であることに触れなかった。可笑しみを扱う問いが無かったため
    Lens("調子", "調子",
         "地の文の調子が一定しているかを見よ。"
         "ですます体とである体、体言止め、話しかけの混ざり方を挙げ、"
         "混ざりが効いている箇所と、ただ揺れている箇所を分けよ。"),
    Lens("笑い", "調子",
         "可笑しみが生まれている箇所を挙げ、"
         "何と何のずれで生じているのかを述べよ。"
         "笑わせようとして滑っている箇所があれば、それも挙げよ。"),
    Lens("落差", "調子",
         "軽く書かれている箇所と、重く書かれている箇所を分けよ。"
         "その落差が働いているか、それとも調子が定まっていないだけかを判じよ。"),

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


# ===== 原稿を測って観点を選ぶ =====
# 観点をくじ引きで選ぶと、その原稿に要らない問いが当たる。
# 感覚描写で埋まった原稿に「感覚を足せ」と言っても意味がない。
# 測れるものだけを測って、足りていないところへ問いを向ける。
#
# 測り方は実測112件の分析で使ったものと同じ。
# 形態素解析は使わない — 依存を増やさずに済み、外れ方も読めるため

# 名前らしきもの。敬称・地名の接尾辞と、長めのカタカナ語を拾う。
# 形態素解析を使わないので、これは固有名詞の判定ではなく
# 「名前を与えられた具体物」の粗い近似である。
# カタカナは4文字以上に限る — エーマート・サイバーキューブは拾い、
# コート・スマホのような短い一般語は落とすため。
# それでもエンジニアやアパートは混じる。精度ではなく、
# 原稿どうしを並べたときの順序が保てればよい
NAME = re.compile(r"[一-龥]{1,3}(?:さん|くん|ちゃん|氏|先生|様)"
                  r"|[一-龥]{2,4}(?:駅|町|村|市|川|山|橋|通り|坂|寺)"
                  r"|[ァ-ヴ][ァ-ヴー]{3,}")
FIRST_PERSON = re.compile(r"私|僕|俺|わたし|あたし")

# 出来事を語っている文の終わり方。
# 「た。/だ。」だけを見ていたときは、である体の地の文を丸ごと落としていた。
# 「運命の九十分」(である体)で出来事12%と出て、
# 実際には語り通しているのに【分岐】(出来事を語っていない原稿に効く観点)が
# 上位に来ていた。同じ7段落で測り直すと29%が71%になる
PAST_END = re.compile(r"(?:た|だ|である|でした|ている|ていた|のだ|のである)"
                      r"[。！？」]")

# 地の文の調子。ですます体とである体が混ざっていないかを見る。
# 「ました。」「でした。」は「た。」で終わるので、
# 素直に書くとですます体まで である体として数えてしまう。先に除く
POLITE_END = re.compile(r"(?:です|ます|ました|ません|でした|でしょう)[。！？]")
PLAIN_END = re.compile(r"(?<!まし)(?<!でし)(?:である|だ|た|ない|る)[。！？]")
SENSES = {
    "音": "音|響|鳴|静か|囁|声が|轟|きしむ|ざわ",
    "におい": "匂|臭|香",
    "手触り": "触|手ざわり|ざらざら|つるつる|滑ら|硬|柔らか",
    "温度": "冷た|温か|熱|寒|暖|ぬる",
    "光": "光|明る|暗|眩|影|翳",
}
RUT = ("宇宙", "神秘", "深淵", "無限", "静寂", "星", "幻想", "生命",
       "永遠", "彼方", "象徴", "囁く")


# 測る項目。ここだけを直せば、空の原稿を返す側も一緒に揃う
MEASURES = ("名前", "会話", "出来事", "感覚", "一人称", "偏り", "混在", "轍")


def survey(paragraphs):
    """観点を選ぶための実測。すべて0〜1に収める"""
    texts = [p.text for p in paragraphs]
    if not texts:
        return {key: 0.5 for key in MEASURES}
    whole = "".join(texts)
    lengths = [len(t) for t in texts]
    # 段落の長さのばらつき(変動係数)。手持ちの4作で 0.81〜1.10 だったので、
    # そのまま0〜1に切ると全部が上限に張り付いて観点を選べない。
    # 観測した幅を0〜1に伸ばす。較正は4作ぶんしかない粗いもの
    raw = (statistics.pstdev(lengths) / statistics.mean(lengths)
           if len(lengths) > 1 and statistics.mean(lengths) else 0.0)
    spread = min(max((raw - 0.7) / 0.6, 0.0), 1.0)
    # 地の文の調子の混ざり具合。少ないほうが全体の何割かを見る。
    # ですます体とである体が半々なら1.0、片方だけなら0.0
    polite = len(POLITE_END.findall(whole))
    plain = len(PLAIN_END.findall(whole))
    mixed = (min(polite, plain) / max(polite, plain, 1) if polite + plain
             else 0.0)

    return {
        "名前": sum(1 for t in texts if NAME.search(t)) / len(texts),
        "会話": sum(1 for t in texts if t.startswith("「")) / len(texts),
        "出来事": sum(1 for t in texts
                    if len(PAST_END.findall(t)) >= 2) / len(texts),
        "感覚": sum(1 for pattern in SENSES.values()
                  if re.search(pattern, whole)) / len(SENSES),
        "一人称": sum(1 for t in texts if FIRST_PERSON.search(t)) / len(texts),
        "偏り": spread,
        "混在": min(mixed, 1.0),
        "轍": min(sum(whole.count(word) for word in RUT)
                 / max(len(whole), 1) * 200, 1.0),
    }


# 各観点が、どの実測のどちら側で効くか。
# 値が無い観点は 0.5 を返して、くじ引きの土俵には残す
NEED = {
    "固有": lambda s: 1 - s["名前"],          # 名前が無い原稿ほど効く
    "感覚": lambda s: 1 - s["感覚"],
    "分岐": lambda s: 1 - s["出来事"],        # 出来事を語っていないほど効く
    "沈黙": lambda s: s["会話"],              # 会話が多いほど、言わなかったことが効く
    "視点": lambda s: s["一人称"],
    "信頼": lambda s: s["一人称"],
    "時制": lambda s: s["出来事"],
    "密度": lambda s: s["偏り"],
    "熱量": lambda s: s["偏り"],
    "既視": lambda s: s["轍"],
    "安全": lambda s: s["轍"],
    "約束事": lambda s: s["轍"],
    "欠落": lambda s: 1 - s["感覚"],
    "順序": lambda s: s["出来事"],
    "一度きり": lambda s: 1 - s["名前"],
    "調子": lambda s: s["混在"],          # 体が混ざっているほど効く
    "落差": lambda s: s["混在"],
}


def needs(paragraphs):
    """観点ごとの必要度。高いほどこの原稿に効くと見込まれる"""
    measured = survey(paragraphs)
    return ({lens.key: NEED.get(lens.key, lambda s: 0.5)(measured)
             for lens in LENSES}, measured)


def suggest_lenses(paragraphs, count=3, rng=None, jitter=0.15):
    """原稿を測って観点を選ぶ。群は散らす。

    jitter は同じ原稿でも回すたびに少し変わるようにするための揺らぎ。
    必要度が拮抗している観点を毎回同じ順で出すと、
    読み直しても同じ角度の指摘しか出てこない"""
    rng = rng or random
    score, measured = needs(paragraphs)
    ranked = sorted(LENSES,
                    key=lambda l: -(score[l.key] + rng.uniform(0, jitter)))

    chosen, used = [], set()
    for lens in ranked:                     # まず群ごとに一つずつ
        if lens.group in used:
            continue
        chosen.append(lens)
        used.add(lens.group)
        if len(chosen) == count:
            return chosen, measured
    for lens in ranked:                     # 足りなければ順に詰める
        if lens not in chosen:
            chosen.append(lens)
            if len(chosen) == count:
                break
    return chosen, measured


def describe(measured):
    """実測を一行で。なぜその観点が選ばれたかを見せるため"""
    return " / ".join(f"{key}{value:.0%}" for key, value in measured.items())

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

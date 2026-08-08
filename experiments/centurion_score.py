"""
センチュリオン: 生成テキストの評価 (Phase 3a)
「跳躍」と「崩壊」を分離して測る。外部モデルを使わないテキストだけの指標。
学習の報酬に使うため、まず既存のresultsで人間の判定を再現できるか較正する。
較正は prot_type7_score.py 側で行う。

跳躍の測り方は、判明2「轍は語彙でなく概念レベル」を踏まえて2本立てにする:
  轍率   — 前回の結果で頻出した語がどれだけ残っているか
  新語率 — 無制御のベースラインには一度も現れなかった語をどれだけ持ち込めたか
崩壊は判明5で実際に観測された副作用を直接数える:
  ラテン文字率 — 「mysteries」「buds」「distant future」の混入
  文字反復率   — 「遠遠と」のような破格
  未完結       — 句点で終わっていない(打ち切りか、文が崩れた跡)
"""

import re

# 前回の結果で頻出した「轍」の語彙 (type2 の BANNED_WORDS と同じ)
RUT_WORDS = ["宇宙", "神秘", "深淵", "無限", "静寂", "星"]

# 語として扱う文字種の連なり。形態素解析器を使わずに内容語を近似する
WORD_PATTERN = re.compile(r"[一-鿿]{2,}|[゠-ヿ]{2,}")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
SENTENCE_END = "。！？」』…"

# 文字の連続は漢字とカタカナに限って数える。
# ひらがなを含めると「お話しした」の「しし」のような正常な語形を拾ってしまう
REPEAT_TARGET = re.compile(r"[一-鿿゠-ヿ]")

# システムプロンプトが禁じている振る舞い。人格の逸脱であって崩壊ではないが、
# 学習が「無難な助手」へ寄るのを防ぐために別枠で数える
PERSONA_BREAKS = [
    "お手伝い", "お役に立", "情報を提供", "ご提供", "ご案内",
    "アシスタント", "AI", "人工知能", "言語モデル",
    "してもいいかも", "しましょうか", "ご質問", "お気軽に",
]


def words(text):
    """内容語らしい文字列を取り出す。漢字2文字以上、またはカタカナ2文字以上"""
    return WORD_PATTERN.findall(text)


def rut_rate(text):
    """轍語が100文字あたり何回出るか"""
    if not text:
        return 0.0
    hits = sum(text.count(w) for w in RUT_WORDS)
    return hits / len(text) * 100


def novelty(text, baseline_vocab):
    """ベースラインに無かった語が、語の種類のうちどれだけを占めるか"""
    vocab = set(words(text))
    if not vocab:
        return 0.0
    return len(vocab - baseline_vocab) / len(vocab)


def latin_rate(text):
    """ラテン文字が全体のどれだけを占めるか。英語混入の直接の指標"""
    if not text:
        return 0.0
    return len(LATIN_PATTERN.findall(text)) / len(text) * 100


def char_repeat(text):
    """同じ漢字・カタカナが連続する率。「遠遠」のような破格を拾う"""
    if len(text) < 2:
        return 0.0
    pairs = sum(1 for a, b in zip(text, text[1:])
                if a == b and REPEAT_TARGET.match(a))
    return pairs / len(text) * 100


def persona_break(text):
    """助力の申し出やAIへの言及。禁じられている振る舞いの数"""
    return float(sum(text.count(p) for p in PERSONA_BREAKS))


def ngram_dup(text, n=5):
    """同じn文字の並びが繰り返される率。言い直しや堂々巡りを拾う"""
    if len(text) < n * 2:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams) * 100


def unterminated(text):
    """文末が句点で終わっていなければ1。打ち切りや崩れた文の跡"""
    stripped = text.rstrip()
    if not stripped:
        return 1.0
    return 0.0 if stripped[-1] in SENTENCE_END else 1.0


def build_baseline_vocab(texts):
    """無制御の出力から、モデルが放っておくと使う語の集合を作る"""
    vocab = set()
    for text in texts:
        vocab.update(words(text))
    return vocab


def measure(text, baseline_vocab):
    """1つの出力について全指標を測る"""
    return {
        "轍率": rut_rate(text),
        "新語率": novelty(text, baseline_vocab),
        "ラテン": latin_rate(text),
        "反復": char_repeat(text),
        "重複": ngram_dup(text),
        "未完結": unterminated(text),
        "逸脱": persona_break(text),
    }


def score(metrics, weights):
    """跳躍から崩壊を引いて1つの数にまとめる"""
    jump = (weights["新語率"] * metrics["新語率"]
            - weights["轍率"] * metrics["轍率"])
    breakdown = (weights["ラテン"] * metrics["ラテン"]
                 + weights["反復"] * metrics["反復"]
                 + weights["重複"] * metrics["重複"]
                 + weights["未完結"] * metrics["未完結"]
                 + weights["逸脱"] * metrics["逸脱"])
    return jump - weights["崩壊の重み"] * breakdown, jump, breakdown

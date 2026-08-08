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

# 崩壊を厳しく罰する重み付け。較正でも学習の報酬でも同じものを使う
DEFAULT_WEIGHTS = {
    "新語率": 10.0,     # 0〜1 の割合なので大きめに掛ける
    "轍率": 1.0,        # 100文字あたりの回数
    "ラテン": 1.0,      # 同上
    "反復": 1.0,
    "重複": 0.5,
    "未完結": 1.0,
    "逸脱": 2.0,        # 助力の申し出は設定への正面からの違反なので重く
    "崩壊の重み": 3.0,   # 崩壊側をまとめて重くする
}

# トレースの書式。「お題: ... 試行 N」で1件が始まる
TRACE_ENTRY = re.compile(r"^お題: (.+?)\s+試行 (\d+)$", re.MULTILINE)

# 標本ファイルの書式
BLOCK_SPLIT = re.compile(r"^-{20,}$", re.MULTILINE)
SAMPLE_FIELD = {
    "id": re.compile(r"^標本: (\S+)", re.MULTILINE),
    "prompt": re.compile(r"^お題: (.+)$", re.MULTILINE),
    "label": re.compile(r"^判定:\s*(.*)$", re.MULTILINE),
    "strength": re.compile(r"^\[抑圧強度: 平均([\d.]+)", re.MULTILINE),
    "width": re.compile(r"^\[実効幅: 平均([\d.]+)", re.MULTILINE),
}

# 〇(U+3007 漢数字のゼロ)と ○(U+25CB 白丸)は見た目が同じで別の文字。
# 日本語入力の変換ではどちらも出てくるので両方受ける
HIT_MARKS = {"○", "〇", "◯", "o", "O", "1"}
MISS_MARKS = {"×", "✕", "✖", "✗", "x", "X", "0"}

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


def find_data(name):
    """データファイルの置き場所を探す。
    Colabではカレントに置くことが多く、リポジトリでは results/ にある。
    どちらでも動くように両方を見る"""
    from pathlib import Path

    here = Path(name)
    if here.exists():
        return here
    in_results = Path(__file__).resolve().parent.parent / "results" / name
    if in_results.exists():
        return in_results
    return here      # 見つからないときは呼び出し側でエラーにさせる


def clean(body):
    """[抑圧強度: ...] のような計測メモと区切り線を落として本文だけにする"""
    lines = [line.strip() for line in body.splitlines()]
    return "".join(line for line in lines
                   if line and not line.startswith("[")
                   and not set(line) <= set("-="))


def parse_trace(path, prompt):
    """トレースの本文から、指定したお題の無制御出力だけを取り出す"""
    parts = TRACE_ENTRY.split(path.read_text(encoding="utf-8"))
    # split の結果は [前置き, お題, 試行, 本文, お題, 試行, 本文, ...]
    return [clean(body) for topic, body in zip(parts[1::3], parts[3::3])
            if topic.strip() == prompt]


def parse_samples(path):
    """ラベル付けした標本ファイルを、1件ずつの辞書に分解する"""
    samples = []
    for block in BLOCK_SPLIT.split(path.read_text(encoding="utf-8")):
        found = {k: p.search(block) for k, p in SAMPLE_FIELD.items()}
        if not found["id"] or not found["prompt"] or not found["label"]:
            continue

        # 判定行より後、計測メモより前が本文
        lines = block.splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("判定:"))
        mark = found["label"].group(1).strip()

        samples.append({
            "id": found["id"].group(1),
            "prompt": found["prompt"].group(1).strip(),
            "label": ("○" if mark in HIT_MARKS
                      else "×" if mark in MISS_MARKS else ""),
            "strength": (float(found["strength"].group(1))
                         if found["strength"] else 0.0),
            "width": float(found["width"].group(1)) if found["width"] else 0.0,
            "text": clean("\n".join(lines[start + 1:])),
        })
    return samples


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

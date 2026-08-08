"""
センチュリオン: 評価関数の較正 (Phase 3a)
既存のresultsに評価関数を当て、人間の判定を再現できるか確かめる。
再現できない指標は学習信号として使えないので、ここを通さずにPhase 4へ進まない。

人間の判定 (centurion_divert.txt の「抑圧あり」5試行):
  当たり: 試行2(深海と朝靄) 試行3(海賊の航路) 試行5(詩人達の引用)
  外れ  : 試行1(轍に戻る)   試行4(助力の申し出で終わる)
重みの方針: 崩壊を厳しく罰する
"""

import re
from pathlib import Path

from centurion_score import build_baseline_vocab, measure, score

# ===== 設定 =====
RESULTS = Path(__file__).resolve().parent.parent / "results"
DIVERT_FILE = RESULTS / "centurion_divert.txt"
NCPS_FILE = RESULTS / "centurion_ncps.txt"
TRACE_FILE = RESULTS / "centurion_trace.txt"

# トレースは他のファイルと書式が違うので別に読む
TRACE_ENTRY = re.compile(r"^お題: (.+?)\s+試行 (\d+)$", re.MULTILINE)
BLUE = "青色にまつわる話を聞かせて"

# 人間がつけた正解ラベル
# 試行2 は概念としては轍の圏内にとどまったまま語彙と映像だけが濃い例で、
# 跳躍とは別の軸だと判断したため較正から外している
HITS = {3, 5}
MISSES = {1, 4}
EXCLUDED = {2}

# 崩壊を厳しく罰する重み付け
WEIGHTS = {
    "新語率": 10.0,     # 0〜1 の割合なので大きめに掛ける
    "轍率": 1.0,        # 100文字あたりの回数
    "ラテン": 1.0,      # 同上
    "反復": 1.0,
    "重複": 0.5,
    "未完結": 1.0,
    "逸脱": 2.0,        # 助力の申し出は設定への正面からの違反なので重く
    "崩壊の重み": 3.0,   # 崩壊側をまとめて重くする
}

# 区切り線に挟まれた行が節の名前になっている
SECTION_HEADER = re.compile(r"^=+\n(.+?)\n=+$", re.MULTILINE)
TRIAL_HEADER = re.compile(r"^--- 試行 (\d+) ---$", re.MULTILINE)


# ===== resultsの読み取り =====
def clean(body):
    """[抑圧強度: ...] のような計測メモと区切り線を落として本文だけにする"""
    lines = [line.strip() for line in body.splitlines()]
    return "".join(line for line in lines
                   if line and not line.startswith("[")
                   and not set(line) <= set("-="))


def parse_trace(path, prompt):
    """トレースの本文から、指定したお題の出力だけを取り出す"""
    parts = TRACE_ENTRY.split(path.read_text(encoding="utf-8"))
    # split の結果は [前置き, お題, 試行, 本文, お題, 試行, 本文, ...]
    return [clean(body) for topic, body in zip(parts[1::3], parts[3::3])
            if topic.strip() == prompt]


def parse(path):
    """結果ファイルを {節の名前: {試行番号: 本文}} に分解する"""
    text = path.read_text(encoding="utf-8")

    # split の結果は [前置き, 節名, 中身, 節名, 中身, ...] と並ぶ
    parts = SECTION_HEADER.split(text)
    sections = {}

    for name, body in zip(parts[1::2], parts[2::2]):
        chunks = TRIAL_HEADER.split(body)
        trials = {int(n): clean(t)
                  for n, t in zip(chunks[1::2], chunks[2::2])}
        if trials:
            sections[name.strip()] = trials

    return sections


def pick(sections, keyword):
    """節の名前に語を含むものを1つ選ぶ"""
    for name, trials in sections.items():
        if keyword in name:
            return trials
    raise KeyError(f"節が見つからない: {keyword} / 候補 {list(sections)}")


# ===== 較正 =====
def report_table(title, trials, baseline_vocab, labels=None):
    print(f"\n--- {title} ---")
    print(f"{'試行':<6} {'判定':<6} {'轍率':>7} {'新語率':>7} {'ラテン':>7}"
          f" {'反復':>6} {'重複':>6} {'未完結':>6} {'逸脱':>5}"
          f" {'跳躍':>7} {'崩壊':>7} {'総合':>8}")

    rows = {}
    for n in sorted(trials):
        m = measure(trials[n], baseline_vocab)
        total, jump, breakdown = score(m, WEIGHTS)
        rows[n] = (total, jump, breakdown)
        mark = ({"○": "当たり", "×": "外れ", "-": "除外"}.get(mark_of(n), "")
                if labels else "")
        print(f"{n:<6} {mark:<6} {m['轍率']:7.2f} {m['新語率']:7.3f}"
              f" {m['ラテン']:7.2f} {m['反復']:6.2f} {m['重複']:6.2f}"
              f" {m['未完結']:6.0f} {m['逸脱']:5.0f}"
              f" {jump:7.2f} {breakdown:7.2f} {total:8.2f}")
    return rows


def mark_of(n):
    if n in HITS:
        return "○"
    if n in MISSES:
        return "×"
    return "-" if n in EXCLUDED else ""


def separation(rows, index):
    """指定した成分だけで当たりと外れがどれだけ離れたか。除外分は数えない"""
    hit = {n: v[index] for n, v in rows.items() if n in HITS}
    miss = {n: v[index] for n, v in rows.items() if n in MISSES}
    # 崩壊は小さいほど良いので符号を反転して揃える
    sign = -1 if index == 2 else 1
    margin = min(sign * v for v in hit.values()) - max(sign * v for v in miss.values())
    order = sorted(rows, key=lambda n: sign * rows[n][index], reverse=True)
    return margin, order


def check_separation(rows):
    """当たりが外れより上に並んだか、そしてそれがどの成分のおかげかを見る"""
    print("\n" + "=" * 60)
    print("較正の判定")
    print("=" * 60)

    # 総合だけでなく、跳躍だけ・崩壊だけでも並べて、分離の出どころを確かめる
    for name, index in (("総合", 0), ("跳躍のみ", 1), ("崩壊のみ", 2)):
        margin, order = separation(rows, index)
        ranking = " > ".join(f"試行{n}{mark_of(n)}" for n in order)
        print(f"{name:<6} 余裕 {margin:+6.2f}   {ranking}")

    total_margin, _ = separation(rows, 0)
    jump_margin, _ = separation(rows, 1)

    print()
    if total_margin <= 0:
        print("総合で分離できていない。テキストだけの指標では足りない。")
        print("Phase 3b (素のモデルによる teacher-forcing 再スコア) が要る。")
    elif jump_margin <= 0:
        print("総合では分離できているが、跳躍だけでは分離できていない。")
        print("つまり効いているのは崩壊の検出だけで、跳躍は測れていない。")
        print("このまま学習に使うと「崩壊しないこと」しか報酬にならず、")
        print("肝心の跳躍が伸びない。跳躍側の指標を作り直す必要がある。")
    else:
        print("跳躍と崩壊の両方で分離できている。学習信号として使える。")

    print(f"\n注意: 標本は当たり{len(HITS)}件・外れ{len(MISSES)}件しかない。")
    print("この規模では、どんな指標でも偶然分離しうる。"
          "重みを調整した時点で過適合している。")
    return total_margin, jump_margin


# ===== 実行 =====
def main():
    divert = parse(DIVERT_FILE)
    baseline = pick(divert, "抑圧なし")
    suppressed = pick(divert, "抑圧あり")

    # Phase 1 のトレースにも同じお題の無制御出力があるので、対照に足す
    baseline_texts = list(baseline.values())
    if TRACE_FILE.exists():
        baseline_texts += parse_trace(TRACE_FILE, BLUE)

    vocab = build_baseline_vocab(baseline_texts)
    print(f"ベースライン: {len(baseline_texts)}件 / 語彙 {len(vocab)}語")

    report_table("抑圧なし (対照)", baseline, vocab)
    rows = report_table("抑圧あり (較正の対象)", suppressed, vocab,
                        labels=(HITS, MISSES))
    margins = check_separation(rows)

    # ncps(未学習)にも当てて、定常状態の出力がどう評価されるか見る
    ncps = parse(NCPS_FILE)
    report_table("ncps 未学習 (参考)", pick(ncps, "ncps制御"), vocab)

    report_embedding(suppressed, baseline_texts)
    return margins


def report_embedding(suppressed, baseline_texts):
    """埋め込みで測った跳躍が、人間の判定を再現できるか確かめる"""
    from centurion_embed import measure_jump

    print("\n" + "=" * 60)
    print("埋め込みによる跳躍 (Phase 3b)")
    print("=" * 60)
    print("初回はモデルの取得に時間がかかる")

    result = measure_jump(suppressed, baseline_texts)

    print(f"\n{'試行':<6} {'判定':<6} {'全体距離':>9} {'文の最大':>9} {'文の平均':>9}")
    for n in sorted(result):
        mark = {"○": "当たり", "×": "外れ", "-": "除外"}.get(mark_of(n), "")
        whole, top, mean = result[n]
        print(f"{n:<6} {mark:<6} {whole:9.4f} {top:9.4f} {mean:9.4f}")

    # 3つの測り方それぞれで、当たりが外れより上に来たかを見る
    print()
    for index, name in enumerate(("全体距離", "文の最大", "文の平均")):
        rows = {n: (v[index],) for n, v in result.items()}
        margin, order = separation(rows, 0)
        ranking = " > ".join(f"試行{n}{mark_of(n)}" for n in order)
        print(f"{name:<8} 余裕 {margin:+7.4f}   {ranking}")

    print("\n注意: 3つの測り方から分離するものを選べば、それ自体が過適合になる。")
    print("標本5件では、どれが正しいかを決められない。ラベルを増やすこと。")


main()

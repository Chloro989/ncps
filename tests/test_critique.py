"""
実行系の試験。モデルは呼ばない。

    python tests/test_critique.py

この試験の主眼は、渡していない段落への言及を捕まえられるかにある。
実際に、番号は実在するが読ませていない段落を2件引き、
どちらも中身を取り違えた答えが出た。
番号の実在だけを見ていたときは、その2件を素通りさせていた。
"""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion import critique
from centurion.manuscript import Manuscript

FIXTURES = HERE / "fixtures"
NOVEL = FIXTURES / "sample_novel.txt"
AOZORA = FIXTURES / "aozora_akuma.txt"
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


def run(*argv):
    """標準出力を捕まえて (終了値, 出力) を返す"""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = critique.main(list(argv))
    return code, buffer.getvalue()


print("== 引数 ==")
args = critique.build_parser().parse_args([str(NOVEL)])
check("既定は発想モード", args.mode == "発想")
check("既定ではモデルを呼ばない", not args.run)
check("モードは四つ", set(critique.MODES) == {"発想", "査読", "接続", "連想"})

print("\n== プロンプトを出す ==")
code, out = run(str(NOVEL), "--mode", "発想", "--seed", "3")
check("正常に終わる", code == 0)
check("指示と本文を区切る", "\n---\n" in out)
check("段落番号つきの本文が入る", "[0]" in out)
check("発想モードの規則が入る", "本文に無いものを述べてよい" in out)

code, out = run(str(NOVEL), "--mode", "査読", "--seed", "3")
check("査読モードに切り替わる", "本文に無い要素について述べない" in out)
check("査読では発想の規則が入らない",
      "本文に無いものを述べてよい" not in out)

code, out = run(str(AOZORA), "--mode", "連想", "--seed", "1", "--steps", "6")
check("連想モードが動く", code == 0 and "6歩" in out)

# 接続は遠い二点を要るので、長い原稿でしか成り立たない。
# 芥川「悪魔」は1558文字しかなく、断られるのが正しい
try:
    run(str(AOZORA), "--mode", "接続", "--seed", "1")
    refusal = ""
except SystemExit as stop:
    refusal = str(stop)
check("短い原稿では接続を断る", "1558文字" in refusal, refusal[:50])
check("断る理由を述べる", "繋ぐ先が無い" in refusal)
check("代わりに使うモードを示す", "査読と発想" in refusal)

with TemporaryDirectory() as folder:
    # 遠い二点を持つ原稿を組む。端と端に「万華鏡」を置き、間を埋め草で伸ばす
    filler = ("　その日も同じ坂を下りて改札を抜けた。売店の灯りがついていて、"
              "並んだ雑誌の表紙だけが明るく見えた。橋の上で立ち止まると、"
              "水路の暗がりには何も映っていなかった。遠くで踏切が鳴って、"
              "それから急に静かになった。")
    long_path = Path(folder) / "長い原稿.txt"
    long_path.write_text(
        "\n\n".join(["　祖母の簞笥の奥に万華鏡が仕舞われていた。"
                     "覗くと硝子の欠片が鳴って、模様がゆっくりと崩れた。"]
                    # 遠い二点と呼べる間隔を空けるため、埋め草を厚く積む。
                    # 同じ文を繰り返すので、埋め草の語は出現回数が多すぎて
                    # 反復の候補から外れる。残るのは「万華鏡」だけになる
                    + [filler] * 30
                    + ["　病室の窓辺に万華鏡が置いてあった。"
                       "誰が持ってきたのか、看護師に訊いても分からなかった。"]),
        encoding="utf-8")
    code, out = run(str(long_path), "--mode", "接続", "--seed", "1")
    check("長い原稿では接続モードが動く", code == 0 and "繋いでいない" in out)
    check("両端の段落を渡す", "簞笥" in out and "病室" in out)
    code, out = run(str(long_path), "--mode", "接続", "--dream", "--seed", "1")
    check("夢の作業を添えられる", "圧縮" in out or "移動" in out
          or "視覚化" in out or "後付け" in out)

print("\n== 塊の指定 ==")
manuscript = Manuscript.load(NOVEL)
chunks = manuscript.chunks(size=600, overlap=1)
code, out = run(str(NOVEL), "--size", "600", "--chunk", "2", "--seed", "3")
check("指定した塊を出す",
      f"[{chunks[1].paragraphs[0].index}]" in out)
try:
    run(str(NOVEL), "--size", "600", "--chunk", "99")
    ok = False
except SystemExit:
    ok = True
check("存在しない塊を指定したら止まる", ok)

print("\n== 観点の決め方 ==")
code, out = run(str(NOVEL), "--survey")
check("実測を出す", code == 0 and "実測" in out)
check("必要度を並べる", "観点の必要度" in out)
check("観点を名指しできると伝える", "--lens" in out)

code, out = run(str(NOVEL), "--lens", "視点,熱量", "--seed", "3")
check("名指しした観点を使う",
      "【視点】" in out and "【熱量】" in out)
check("名指しなら数の指定より優先する", out.count("\n1. 【") == 1)
code, out = run(str(NOVEL), "--lens", "視点／熱量")
check("全角の区切りでも通る", "【視点】" in out and "【熱量】" in out)
try:
    run(str(NOVEL), "--lens", "無い観点")
    rejected = False
except SystemExit as stop:
    rejected = "知らない観点" in str(stop)
check("知らない観点は断る", rejected)

def label_of(*extra):
    """観点の決め方は説明文に残る。それを見て確かめる"""
    args = critique.build_parser().parse_args([str(NOVEL), "--seed", "3"]
                                              + list(extra))
    return critique.tasks(Manuscript.load(NOVEL), args)[0][3]


check("既定は実測で選ぶ", "実測" in label_of(), label_of())
check("実測の中身も残す", "名前" in label_of())
check("くじ引きに戻せる", "くじ引き" in label_of("--random-lenses"))
check("名指しはそう記録する", "指定" in label_of("--lens", "視点,熱量"))

print("\n== 全部の塊を読ませる ==")
code, out = run(str(NOVEL), "--size", "600", "--all", "--seed", "5")
check("塊の数だけ問いが並ぶ", out.count("\n---\n") == len(chunks),
      f"{out.count(chr(10) + '---' + chr(10))} 対 {len(chunks)}")
check("塊の間に区切りが入る", "=" * 64 in out)

first = critique.tasks(Manuscript.load(NOVEL),
                       critique.build_parser().parse_args(
                           [str(NOVEL), "--size", "600", "--all",
                            "--seed", "5"]))
check("塊ごとに観点が変わる",
      len({job[3] for job in first}) == len(first),
      str([job[3] for job in first]))
check("種を固定すれば同じ観点になる",
      [job[3] for job in first]
      == [job[3] for job in critique.tasks(
          Manuscript.load(NOVEL),
          critique.build_parser().parse_args(
              [str(NOVEL), "--size", "600", "--all", "--seed", "5"]))])
try:
    run(str(NOVEL), "--mode", "連想", "--all")
    refused = False
except SystemExit as stop:
    refused = "--all は発想と査読" in str(stop)
check("接続と連想では --all を断る", refused)

print("\n== モデルの呼び分け ==")
saved = os.environ.pop("ANTHROPIC_API_KEY", None)
try:
    try:
        critique.Api()
        guarded = False
    except SystemExit as stop:
        guarded = "ANTHROPIC_API_KEY" in str(stop)
    check("鍵が無ければ止まる", guarded)
    try:
        critique.Api()
    except SystemExit as stop:
        check("鍵の作り方を伝える", "console.anthropic.com" in str(stop))
        check("鍵を保存しないと明言する", "保存も表示もしない" in str(stop))
finally:
    if saved is not None:
        os.environ["ANTHROPIC_API_KEY"] = saved

os.environ["ANTHROPIC_API_KEY"] = "試験用の偽の鍵"
try:
    caller = critique.Api("試験用のモデル")
    check("鍵があれば作れる", caller.model == "試験用のモデル")
    check("鍵を控える", caller.key == "試験用の偽の鍵")
finally:
    if saved is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = saved

check("既定のモデルが定まっている",
      critique.API_MODEL.startswith("claude")
      and "Qwen" in critique.LOCAL_MODEL)
check("論評に足りる長さを取る", critique.MAX_TOKENS >= 2000,
      str(critique.MAX_TOKENS))

print("\n== 添削ファイルの記録 ==")
manuscript = Manuscript.load(NOVEL)


def records_for(*extra):
    args = critique.build_parser().parse_args([str(NOVEL)] + list(extra))
    return dict(critique.annotation_records(
        args, manuscript, labels=["1/1塊 視点／熱量 [実測]"]))


check("日付を残す", "日付" in records_for())
check("モードを残す", records_for("--mode", "査読")["モード"] == "査読")
check("原稿の大きさを残す", "段落" in records_for()["原稿"])
check("観点を残す", "視点／熱量" in records_for()["読ませた範囲と観点"])
check("語の取り出し方を残す", records_for()["語の取り出し"] == "正規表現")

check("APIなら API と分かる", "(API)" in records_for("--api")["モデル"])
check("APIの既定モデルを残す",
      critique.API_MODEL in records_for("--api")["モデル"])
check("手元なら 手元 と分かる", "(手元)" in records_for("--run")["モデル"])
check("手元の既定モデルを残す",
      critique.LOCAL_MODEL in records_for("--run")["モデル"])
check("名指ししたモデルを残す",
      "LiquidAI/LFM2-1.2B" in
      records_for("--run", "--model", "LiquidAI/LFM2-1.2B")["モデル"])
check("解かせていなければ不明と書く",
      "不明" in records_for()["モデル"])
check("申告されたモデルはそう書く",
      "(申告)" in records_for("--model", "外のモデル")["モデル"])

args = critique.build_parser().parse_args([str(NOVEL), "--check", "答え.txt"])
checked = dict(critique.annotation_records(args, manuscript,
                                           source="どこか/答え.txt"))
check("検査した答えのファイル名を残す", checked["検査した答え"] == "答え.txt")
check("パスは残さずファイル名だけにする",
      "どこか" not in checked["検査した答え"])
check("外の答えならモードは申告扱い", "(申告)" in checked["モード"])
check("範囲が分からなければその欄を出さない",
      "読ませた範囲と観点" not in checked)

print("\n== 段落番号の検査 ==")
with TemporaryDirectory() as folder:
    answer = Path(folder) / "answer.txt"

    answer.write_text("[0] と [1] は対応している。", encoding="utf-8")
    code, out = run(str(NOVEL), "--check", str(answer))
    check("実在する番号だけなら通る", code == 0, out.splitlines()[1])
    check("番号を本文に戻す",
          manuscript.paragraphs[0].text[:8] in out)

    answer.write_text("[0] は良い。[9999] は冗長である。", encoding="utf-8")
    code, out = run(str(NOVEL), "--check", str(answer))
    check("存在しない番号で落とす", code == 1)
    check("存在しない番号を名指しする", "[9999]" in out)
    check("番号を本文に戻せなくても印を付ける", "[9999?]" in out)

    # 渡していない範囲への言及。番号は実在するので、
    # 実在だけを見る検査では素通りしてしまう
    last = chunks[-1].paragraphs[-1].index
    first = chunks[0].paragraphs[0].index
    answer.write_text(f"[{last}] の効果は [{first}] に由来する。",
                      encoding="utf-8")
    code, out = run(str(NOVEL), "--check", str(answer))
    check("塊を指定しなければ素通りする", code == 0)
    check("素通りすることを断り書きする", "--chunk を渡すと" in out)

    code, out = run(str(NOVEL), "--check", str(answer),
                    "--size", "600", "--chunk", str(len(chunks)))
    check("塊を指定すれば範囲外を捕まえる", code == 1, out.splitlines()[1])
    check("範囲外の番号を名指しする", f"[{first}]" in out)
    check("範囲外だと分かる言葉で伝える", "見せていない範囲" in out)

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

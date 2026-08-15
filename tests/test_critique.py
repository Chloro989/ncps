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

print("\n== llama.cpp ==")
# 手元の RX 6700 XT では torch が動かない。llama.cpp なら Vulkan で
# AMDのGPUを使えるので、Colab に行かずに済む
caller = critique.Llama()
check("既定の窓口を持つ", caller.url == critique.LLAMA_URL)
check("窓口を差し替えられる",
      critique.Llama(url="http://例:9999/v1/chat/completions").url
      == "http://例:9999/v1/chat/completions")
check("モデル名を渡さなくても作れる", caller.model == "local")
check("モデル名を渡せる", critique.Llama("LFM2.5-1.2B-JP").model
      == "LFM2.5-1.2B-JP")
try:
    critique.Llama(url="http://127.0.0.1:1/v1/chat/completions")(
        "指示", "本文")
    reached = True
except SystemExit as stop:
    reached = False
    message = str(stop)
check("届かなければ止まる", not reached)
check("サーバの立て方を伝える", "llama-server" in message)
check("鍵は要らない", "ANTHROPIC_API_KEY" not in message)

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
check("llama.cpp ならそう分かる",
      "(llama.cpp)" in records_for("--llama")["モデル"])
check("llama.cpp でもモデル名を残せる",
      "LFM2.5-1.2B-JP" in
      records_for("--llama", "--model", "LFM2.5-1.2B-JP")["モデル"])
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

print("\n== 検分にかける ==")
manuscript = Manuscript.load(NOVEL)
ANSWER = ("- [0] の灯りは伏線になっていない。二度目を置くべきだ。\n"
          "- [1] と [3] の水は繋がるはずだが書かれていない。\n"
          "- [2] はもっと丁寧に書くとよい。\n")


class FakeJudge:
    def __init__(self, verdict):
        self.verdict = verdict
        self.asked = None

    def __call__(self, head, body, max_tokens=0):
        self.asked = (head, body)
        return self.verdict


def verify_with(verdict, *extra):
    args = critique.build_parser().parse_args(
        [str(NOVEL), "--verify", "--llama"] + list(extra))
    judge = FakeJudge(verdict)
    real = critique.verifier
    critique.verifier = lambda a: (judge, "試験")
    try:
        return critique.run_verify(args, manuscript, ANSWER, "本文"), judge
    finally:
        critique.verifier = real


(answer, how), judge = verify_with(
    "1: 残す 具体的\n2: 捨てる 本文に無い\n3: 捨てる 曖昧")
check("残った指摘だけになる", "灯りは伏線" in answer)
check("捨てた指摘は消える", "もっと丁寧に" not in answer)
check("経過を伝える", "1件を残し、2件を捨てた" in how, how[:60])
check("捨てた理由も出す", "本文に無い" in how)
check("本文を検分側にも渡す", "本文" in judge.asked[1])
check("疑う側に立たせている", "粗を探す側" in judge.asked[0])

(answer, how), _ = verify_with("1: 捨てる\n2: 捨てる\n3: 捨てる")
check("すべて捨てられたら元の答えを残す", "もっと丁寧に" in answer)
check("そのとき理由を述べる", "働きすぎている" in how, how[:60])

# 一件も判定できなかったら、検証は働かなかったということ。
# 全部の行に「判定されなかった」と貼るのは、何も分からなかったことを
# 分かったように見せるだけで害になる。実際にそうなった
(answer, how), _ = verify_with("読めない返事")
check("一件も判定できなければ元の答えをそのまま出す", answer == ANSWER)
check("行に注記を貼らない", "判定されなかった" not in answer)
check("働かなかったと伝える", "検証は働かなかった" in how, how[:60])
check("何件渡して何件読めたかを出す", "読み取れた判定は0件" in how)
check("どう直せばよいかを言う", "観点を減らす" in how)
check("返事の頭を見せる", "読めない返事" in how)

# 一部だけ読めたときは、読めなかったものを残して印を付ける
(answer, how), _ = verify_with("1: 残す よい")
check("一部でも判定できれば残った指摘を出す", "灯りは伏線" in answer)
check("読めなかったものは印を付けて残す", "判定されなかった" in answer)

(answer, how), _ = verify_with("1: 残す よい", "--verify-model", "別のモデル")
args = critique.build_parser().parse_args(
    [str(NOVEL), "--verify", "--llama", "--verify-model", "別のモデル"])
check("検分がまだなら未実行と書く",
      dict(critique.annotation_records(args, manuscript, ["x"]))["検証"]
      == "未実行")
args.verify_where = "llama.cpp http://例:8081 / 別のモデル"
check("検分した相手を記録に残す",
      "別のモデル" in dict(critique.annotation_records(
          args, manuscript, ["x"], outcome="2件を残し、1件を捨てた"))["検証"])
check("何件残したかも記録に残す",
      "2件を残し" in dict(critique.annotation_records(
          args, manuscript, ["x"], outcome="2件を残し、1件を捨てた"))["検証"])
check("検証していなければ記録に出さない",
      "検証" not in dict(critique.annotation_records(
          critique.build_parser().parse_args([str(NOVEL)]), manuscript, ["x"])))

print("\n== 検分する側の選び方 ==")


def judge_kind(*extra):
    args = critique.build_parser().parse_args([str(NOVEL), "--verify"]
                                              + list(extra))
    return critique.verifier(args)[1]


saved_key = os.environ.get("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = "試験用の偽の鍵"
try:
    check("既定は同じ経路 (API)", "API" in judge_kind("--api"))
    check("既定は同じ経路 (llama)", "llama.cpp" in judge_kind("--llama"))
    check("経路を変えられる",
          "API" in judge_kind("--llama", "--verify-with", "api"))
    check("検分に別のモデルを渡せる",
          critique.verifier(critique.build_parser().parse_args(
              [str(NOVEL), "--verify", "--api",
               "--verify-model", "claude-opus-5"]))[0].model
          == "claude-opus-5")

    # llama-server は起動時に読み込んだモデル1つだけを配る。
    # --verify-model に別の名前を書いてもサーバは無視するので、
    # 同じ窓口を指している限り検分するのは同じモデルになる。
    # 以前はそれを「検証: Qwen (llama)」と記録していて、事実と違っていた
    same = judge_kind("--llama", "--verify-model", "Qwen2.5-3B")
    check("同じ窓口なら同じモデルだと記録する", "書いた側と同じ" in same, same)
    check("そのとき別のモデル名を書かない", "Qwen2.5-3B" not in same, same)

    other = judge_kind("--llama", "--verify-llama-url",
                       "http://127.0.0.1:8081/v1/chat/completions")
    check("別の窓口なら同じとは書かない", "書いた側と同じ" not in other, other)
    check("どの窓口かを記録する", "8081" in other, other)

    caller, _ = critique.verifier(critique.build_parser().parse_args(
        [str(NOVEL), "--verify", "--llama",
         "--verify-llama-url", "http://127.0.0.1:8081/v1/chat/completions"]))
    check("別の窓口へ実際に繋ぎに行く", "8081" in caller.url, caller.url)
finally:
    if saved_key is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = saved_key

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

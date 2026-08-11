"""
実行系の試験。モデルは呼ばない。

    python tests/test_critique.py

この試験の主眼は、渡していない段落への言及を捕まえられるかにある。
実際に、番号は実在するが読ませていない段落を2件引き、
どちらも中身を取り違えた答えが出た。
番号の実在だけを見ていたときは、その2件を素通りさせていた。
"""

import io
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

code, out = run(str(AOZORA), "--mode", "接続", "--seed", "1")
check("接続モードが動く", code == 0 and "繋いでいない" in out)
code, out = run(str(AOZORA), "--mode", "連想", "--seed", "1", "--steps", "6")
check("連想モードが動く", code == 0 and "6歩" in out)

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

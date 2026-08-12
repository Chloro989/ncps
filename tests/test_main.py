"""
入口の試験。

    python tests/test_main.py

注意: `main.py test` を引数なしで呼ぶとこの試験も走り、その中で
また試験を呼ぶことになる。ここでは必ず試験名を明示して呼ぶ。
"""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import main as entry

FIXTURES = HERE / "fixtures"
NOVEL = FIXTURES / "sample_novel.txt"
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
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = entry.main(list(argv))
    return code, out.getvalue(), err.getvalue()


print("== 案内 ==")
for argv in ([], ["-h"], ["--help"], ["help"]):
    code, out, _ = run(*argv)
    check(f"{argv or '引数なし'} で案内を出す",
          code == 0 and "センチュリオン" in out and "read" in out)

code, out, err = run("foo")
check("知らない命令で終了値1", code == 1)
check("知らない命令を名指しする", "foo" in err)
check("案内も一緒に出す", "read" in out)

print("\n== 命令の一覧 ==")
check("五つある",
      set(entry.COMMANDS) == {"read", "ask", "check", "write", "test"},
      str(sorted(entry.COMMANDS)))
check("試験の一覧が空でない", len(entry.TESTS) >= 5)
check("試験の一覧が実在する",
      all((HERE / f"{name}.py").exists() for name in entry.TESTS),
      str([n for n in entry.TESTS if not (HERE / f"{n}.py").exists()]))

print("\n== read ==")
code, out, _ = run("read", str(NOVEL), "--size", "600")
check("正常に終わる", code == 0)
check("章の一覧が出る", "第一章" in out)
check("塊の一覧が出る", "塊0" in out)
check("反復の見出しが出る", "反復" in out)

print("\n== ask ==")
code, out, _ = run("ask", str(NOVEL), "--mode", "発想", "--seed", "3")
check("正常に終わる", code == 0)
check("発想モードの規則が入る", "本文に無いものを述べてよい" in out)
check("段落番号つきの本文が入る", "[0]" in out)

code, out, _ = run("ask", str(NOVEL), "--mode", "査読", "--seed", "3")
check("査読モードに切り替わる", "本文に無い要素について述べない" in out)

print("\n== check ==")
with TemporaryDirectory() as folder:
    answer = Path(folder) / "answer.txt"

    answer.write_text("[0] と [1] は響いている。", encoding="utf-8")
    code, out, _ = run("check", str(answer), str(NOVEL))
    check("実在する番号だけなら通る", code == 0)
    check("番号を本文に戻す", "[0「" in out)

    answer.write_text("[9999] は冗長である。", encoding="utf-8")
    code, out, _ = run("check", str(answer), str(NOVEL))
    check("存在しない番号で落とす", code == 1)

    code, out, err = run("check")
    check("答えを渡さなければ断る", code == 1 and "答えのファイル" in err)

print("\n== test ==")
code, out, _ = run("test", "test_review")
check("指定した試験を走らせる", code == 0 and "test_review" in out)
check("通過数を出す", "件通過" in out)
code, out, _ = run("test", "test_無い")
check("無い試験名で落とす", code == 1)

print("\n== READMEと実物が合っているか ==")
# 引数は増え続ける。書き忘れをここで止める
readme = (HERE.parent / "README.md").read_text(encoding="utf-8")
from centurion.critique import MODES, build_parser
from centurion.review import LENSES

flags = [option for action in build_parser()._actions
         if action.dest != "help"
         for option in action.option_strings
         if option.startswith("--")]
missing = [flag for flag in flags if flag not in readme]
check("すべての引数がREADMEにある", not missing, str(missing))

missing = [mode for mode in MODES if f"--mode {mode}" not in readme
           and f"`{mode}`" not in readme]
check("すべてのモードがREADMEにある", not missing, str(missing))

missing = [lens.key for lens in LENSES if lens.key not in readme]
check("すべての観点がREADMEにある", not missing, str(missing))

missing = [name for name in entry.COMMANDS if f"main.py {name}" not in readme]
check("すべての命令がREADMEにある", not missing, str(missing))

check("試験の件数がREADMEに書いてある",
      any(f"({total}件)" in readme for total in range(200, 1000)),
      "件数の表記が見つからない")

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

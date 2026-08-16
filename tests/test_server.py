"""
llama-server を画面から起こす仕組みの試験。

    python tests/test_server.py

一番大事なのは、何を起こせるかの線引き。
画面からの求めで子プロセスを起こすので、ここが緩いと
打ち間違いがそのまま妙なプロセスの起動になる。

実際の llama-server は起こさない。組み立てた引数と、
断り方だけを見る。
"""

import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from centurion import server

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


HAVE = {"unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
        "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"}


print("== 起こしてよいものの線引き ==")
check("手元にあるものは通す",
      server.check_model("unsloth/Qwen3.8-27B-GGUF:Q4_K_M", HAVE) == "")
check("手元に無くても形が正しければ通す",
      server.check_model("org/name-GGUF:Q4_K_M", HAVE) == "")
check("量子化を省いても通す", server.check_model("org/name", HAVE) == "")
check("空なら断る", "選ぶこと" in server.check_model("", HAVE))
check("空白だけでも断る", "選ぶこと" in server.check_model("   ", HAVE))

# 引数は並びで渡すので殻は挟まらないが、形を絞っておけば
# 打ち間違いがそのまま起動の失敗になるのを防げる
FISHY = ["a; whoami", "org/name && echo", "org/name | more",
         "$(whoami)", "`whoami`", "org/name\nrm", "../../etc/passwd",
         "org/name;rm -rf /", "-hf", "--host", "org/na me"]
for odd in FISHY:
    check(f"{odd!r} は断る",
          "起こせない" in server.check_model(odd, HAVE), odd)

check("斜線が無ければ断る",
      "起こせない" in server.check_model("Qwen3.8-27B", HAVE))


print("\n== 数値の検め ==")
check("大きすぎる層は丸める", server.whole("ngl", "999999") == 999)
check("負の層は0にする", server.whole("ngl", "-5") == 0)
check("数でなければ既定に戻す", server.whole("ctx", "あ") == 12288)
check("空でも既定に戻す", server.whole("port", "") == 8080)
check("None でも落ちない", server.whole("port", None) == 8080)
check("小さすぎるポートは丸める", server.whole("port", "80") == 1024)
check("普通の値はそのまま", server.whole("ngl", "36") == 36)
check("文字の数も読む", server.whole("ngl", 36) == 36)


print("\n== 引数の組み立て ==")
if not server.binary():
    check("llama-server が無いので組み立ては試さない", True)
else:
    made = server.build_command("unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
                                ngl=36, ctx=12288)
    check("並びで組む (文字列にしない)", isinstance(made, list))
    check("リポジトリ名なら -hf", "-hf" in made)
    check("-m は使わない", "-m" not in made)
    check("外に開かせない",
          made[made.index("--host") + 1] == "127.0.0.1")
    check("層を渡す", made[made.index("-ngl") + 1] == "36")
    check("文脈の長さを渡す", made[made.index("-c") + 1] == "12288")
    check("既定で flash attention を入れる", "-fa" in made)
    check("既定で KV を 8bit にする", "--cache-type-k" in made)
    check("エキスパートを指定しなければ旗を出さない",
          "--n-cpu-moe" not in made)

    moe = server.build_command("org/name-A3B-GGUF:Q4_K_M", cpu_moe=24)
    check("エキスパートを指定すれば旗を出す",
          moe[moe.index("--n-cpu-moe") + 1] == "24")

    plain = server.build_command("org/name", flash=False,
                                 quantize_cache=False)
    check("flash を外せる", "-fa" not in plain)
    check("KV の8bit化も外せる", "--cache-type-k" not in plain)

    # 数値は組み立ての中でも丸める。画面から素通りさせない
    wild = server.build_command("org/name", ngl="99999", ctx="あ",
                                port="1")
    check("組み立てでも層を丸める", wild[wild.index("-ngl") + 1] == "999")
    check("組み立てでも文脈を戻す", wild[wild.index("-c") + 1] == "12288")
    check("組み立てでもポートを丸める",
          wild[wild.index("--port") + 1] == "1024")

    with TemporaryDirectory() as folder:
        here = Path(folder) / "手元.gguf"
        here.write_bytes(b"x")
        local = server.build_command(str(here))
        check("実在するファイルなら -m", "-m" in local and "-hf" not in local,
              " ".join(local))


print("\n== 起こして止める ==")
# 本物の llama-server は起こさない。python を身代わりにして、
# 抱える・様子を見る・止める の筋道だけを確かめる
idle = server.Launcher()
check("最初は立っていない", not idle.alive())
check("最初は窓口も答えない", not idle.ready())
check("様子を訊ける", idle.status()["running"] is False)
check("止めても落ちない", idle.stop()["running"] is False)

fake = server.Launcher()
fake.process = subprocess.Popen(
    [sys.executable, "-c",
     "import sys,time\nprint('load_tensors: 試験', flush=True)\n"
     "time.sleep(30)"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    encoding="utf-8", bufsize=1)
fake.model = "身代わり"
import threading
threading.Thread(target=fake._drain, args=(fake.process.stdout,),
                 daemon=True).start()
try:
    check("立っていると分かる", fake.alive())
    for _ in range(50):
        if fake.lines:
            break
        time.sleep(0.05)
    check("起動ログを覚える",
          any("load_tensors" in line for line in fake.lines),
          str(list(fake.lines)))
    check("窓口が答えなければ ready ではない", not fake.ready())
    state = fake.stop()
    check("止められる", not fake.alive())
    check("止めたことを記録に残す",
          any("止めた" in line for line in state["log"]),
          str(state["log"][-2:]))
    check("止めた後は様子も止まっている", state["running"] is False)
finally:
    if fake.alive():
        fake.process.kill()

# 二つ立てない。llama.cpp は1プロセス1モデルなので、
# 二重に起こすとポートが衝突して原因が分かりにくくなる
busy = server.Launcher()
busy.process = subprocess.Popen([sys.executable, "-c", "import time;"
                                 "time.sleep(30)"])
busy.model = "先客"
try:
    try:
        busy.start("org/name")
        refused = ""
    except RuntimeError as problem:
        refused = str(problem)
    except FileNotFoundError:
        refused = "llama-server が無い"
    check("立っているのに起こそうとしたら断る",
          "先客" in refused or "llama-server が無い" in refused, refused)
finally:
    busy.process.kill()
    busy.process.wait(timeout=5)


print("\n== 詰まりの読み取り ==")
check("ファイルが無いと分かる",
      "-hf" in server.trouble(["E gguf_init_from_file: failed to open "
                               "GGUF file 'org/name' (No such file)"]))
check("VRAM 不足と分かる",
      "-ngl" in server.trouble(["ggml_vulkan: out of memory"]))
check("旗が古いと分かる",
      "版が古い" in server.trouble(["error: unknown argument: --n-cpu-moe"]))
check("普通のログには何も言わない",
      server.trouble(["load_tensors: Vulkan0 buffer size = 9000 MiB"]) == "")
check("空でも落ちない", server.trouble([]) == "")


print(f"\n{passed}件通過 / {failed}件失敗")
sys.exit(1 if failed else 0)

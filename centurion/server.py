"""
llama-server を画面から起こす。

## なぜ

llama-server は別の窓で立てるものだった。そのため

  - どの旗をどう組むかを毎回思い出す必要がある
  - -ngl を詰めるたびに窓を行き来する
  - 起動ログ (load_tensors: の配分) を見るのも別の窓

実際、-m と -hf を取り違えて何度も起動に失敗している。
画面から起こせれば、旗の組み立ては道具側の仕事になる。

## 安全について

これは**画面からの求めで子プロセスを起こす**仕組みなので、
何を起こせるかを絞る。

  1. shell=True を使わない。引数は必ず並びで渡す
  2. モデル名は、手元で見つけたものか、
     組織/名前:量子化 の形に合うものだけ通す
  3. 数値の旗は int に直し、範囲を検める
  4. --host は 127.0.0.1 に固定する。
     llama-server を外に開かせない

画面そのものが 127.0.0.1 にしか開いていないので、
これは外からの攻撃を防ぐというより、
打ち間違いで妙なものを起こさないための歯止めである。
"""

import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import critique

# 組織/名前 または 組織/名前:量子化。それ以外は通さない。
# 引数は並びで渡すので殻は挟まらないが、形を絞っておけば
# 打ち間違いがそのまま起動の失敗になるのを防げる
REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?$")

# 起動ログを何行まで覚えておくか。load_tensors: の配分が見えれば足りる
KEEP = 400

# 立ち上がるまでの待ち時間。大きいモデルは読み込みに数分かかる
READY_WAIT = 600
READY_STEP = 2

LIMITS = {
    "port": (1024, 65535, 8080),
    "ngl": (0, 999, 99),
    "ctx": (512, 1048576, 12288),
    "cpu_moe": (0, 999, 0),
}


def binary():
    """llama-server の在り処。無ければ空"""
    return shutil.which("llama-server") or ""


def whole(name, value):
    """数値の旗を int に直して、範囲に収める"""
    low, high, fallback = LIMITS[name]
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def check_model(name, allowed):
    """起こしてよいモデルか。駄目なら理由を返す"""
    name = (name or "").strip()
    if not name:
        return "モデルを選ぶこと"
    if name in allowed:
        return ""
    if REPO.match(name):
        return ""            # 手元に無いが、形は正しい。落としに行く
    return (f"この名前では起こせない: {name}\n"
            "手元にあるものを選ぶか、組織/名前:量子化 の形で書くこと")


def build_command(model, port=8080, ngl=99, ctx=12288, cpu_moe=0,
                  flash=True, quantize_cache=True, extra=()):
    """llama-server の引数を組む。ここが唯一の組み立て場所。

    -hf と -m を取り違えないよう、手元のファイルなら -m、
    リポジトリ名なら -hf を自分で選ぶ"""
    where = binary()
    if not where:
        raise FileNotFoundError(
            "llama-server が見つからない。\n"
            "  winget install ggml.llamacpp\n"
            "入れた後は端末を開き直すこと (PATH の反映のため)")

    # -m は手元のファイル、-hf はリポジトリ名。ここを取り違えると
    # failed to open GGUF file になる。実在するファイルかどうかで決める
    flag = "-m" if Path(model).expanduser().exists() else "-hf"

    command = [where, flag, model,
               "--host", "127.0.0.1",
               "--port", str(whole("port", port)),
               "-ngl", str(whole("ngl", ngl)),
               "-c", str(whole("ctx", ctx))]
    if flash:
        command += ["-fa", "on"]
    if quantize_cache:
        # KV キャッシュを 8bit に。VRAM がおよそ半分で済む。
        # flash attention が要るので、外すときは一緒に外す
        command += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    moe = whole("cpu_moe", cpu_moe)
    if moe:
        command += ["--n-cpu-moe", str(moe)]
    command += [str(word) for word in extra]
    return command


class Launcher:
    """llama-server を一つだけ抱える。

    二つ立てたいときは別のポートで別の Launcher を作る。
    ただし llama-server は1プロセス1モデルなので、
    普段は一つで足りる"""

    def __init__(self):
        self.process = None
        self.lines = deque(maxlen=KEEP)
        self.command = []
        self.model = ""
        self.port = LIMITS["port"][2]
        self.started = None
        self.lock = threading.Lock()

    # ----- 様子を見る -----

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def url(self):
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    def ready(self):
        """窓口が応じるか。読み込み中はまだ応じない"""
        return bool(critique.Llama(url=self.url()).loaded()) if self.alive() \
            else False

    def status(self):
        code = None if self.process is None else self.process.poll()
        return {
            "binary": binary(),
            "running": self.alive(),
            "model": self.model,
            "port": self.port,
            "url": self.url(),
            "command": " ".join(self.command),
            "since": self.started,
            "exit": code,
            "log": list(self.lines),
        }

    # ----- 起こす -----

    def _drain(self, stream):
        """llama-server の言うことを覚えておく。
        load_tensors: の配分が見えないと -ngl を詰められない"""
        for line in stream:
            self.lines.append(line.rstrip("\r\n"))
        stream.close()

    def start(self, model, **options):
        """起こす。既に立っていれば断る"""
        with self.lock:
            if self.alive():
                raise RuntimeError(
                    f"すでに {self.model} が立っている。"
                    "先に止めること (1プロセス1モデル)")
            command = build_command(model, **options)
            self.lines.clear()
            self.lines.append("$ " + " ".join(command))
            # 殻を挟まない。引数は並びのまま渡す
            self.process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=(subprocess.CREATE_NO_WINDOW
                               if sys.platform == "win32" else 0))
            self.command = command
            self.model = model
            self.port = whole("port", options.get("port", 8080))
            self.started = time.strftime("%H:%M:%S")
            threading.Thread(target=self._drain,
                             args=(self.process.stdout,),
                             daemon=True).start()
        return self.status()

    def wait_ready(self, limit=READY_WAIT):
        """窓口が応じるまで待つ。(応じたか, 何秒かかったか)"""
        began = time.monotonic()
        while time.monotonic() - began < limit:
            if not self.alive():
                return False, time.monotonic() - began
            if self.ready():
                return True, time.monotonic() - began
            time.sleep(READY_STEP)
        return False, time.monotonic() - began

    def stop(self, patience=10):
        """止める。行儀よく頼んでから、聞かなければ切る"""
        with self.lock:
            if not self.alive():
                self.process = None
                return self.status()
            self.process.terminate()
            try:
                self.process.wait(timeout=patience)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=patience)
            self.lines.append("(止めた)")
            self.started = None
            return self.status()


def trouble(lines):
    """起動ログから、よくある詰まりを読み取って一言にする。

    ログは長いので、要るところだけ拾って伝える"""
    text = "\n".join(lines)
    if "failed to open GGUF file" in text:
        return ("ファイルが見つからない。"
                "リポジトリ名なら -hf、手元のファイルなら -m を使う")
    if "out of memory" in text.lower() or "failed to allocate" in text:
        return "VRAM が足りない。-ngl を減らすか、-c を小さくすること"
    if "unknown argument" in text or "invalid argument" in text:
        return "llama.cpp が知らない旗がある。版が古い可能性がある"
    if "error while handling argument" in text:
        return "旗の書き方が合っていない"
    return ""


# 画面から使う一つきりの Launcher。
# llama.cpp は1プロセス1モデルなので、抱えるのも一つでよい
running = Launcher()

"""
原稿を読ませて、論評か発想を得る。

    python -m centurion.critique 原稿.txt                    # 発想の問いを出す
    python -m centurion.critique 原稿.txt --mode 査読
    python -m centurion.critique 原稿.txt --mode 接続
    python -m centurion.critique 原稿.txt --mode 連想
    python -m centurion.critique 原稿.txt --api               # その場で論評させる
    python -m centurion.critique 原稿.txt --api --all         # 全部の塊を順に
    python -m centurion.critique 原稿.txt --run               # 手元のモデルで
    python -m centurion.critique 原稿.txt --check 答え.txt    # 段落番号を検査する

既定では**プロンプトを出すだけ**で、モデルは呼ばない。
手元にGPUが無くても使えるようにするためで、出したものを好きなチャットへ
貼れば、性能の高いモデルで読ませられる。

--api か --run を付けると、その場で解かせて段落番号の検査まで通す。
  --api  Claude の API。鍵は環境変数 ANTHROPIC_API_KEY から読む。
         文芸の論評に耐える質が要るならこちら
  --run  手元(か Colab)のモデル。無料だが、3B級では論評の質が出ない

段落番号の検査だけは、答えを受け取ってから別に走らせられる。
「実在しない箇所への指摘」は、モデルが強くなっても消えないため。
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .answer import (annotate, anchoring, check_scores, find_quotes,
                     report_anchoring, report_quotes)
from .connect import (DREAM_WORK, MIN_CHARS, build_chain_prompt,
                      build_connection_prompt, distant_pairs, recurrences,
                      use_morphology)
from .manuscript import Manuscript
from . import verify
from . import review, rubric, wording
# review.LENSES と review.LENS_BY_KEY は prompts/観点.txt で差し替わるので、
# 名前で取り込まず review 越しに読む
from .review import (build_prompt, check_citations, choose_lenses, describe,
                     lenses_for, resolve, suggest_lenses, uses_lenses)

LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
API_MODEL = "claude-sonnet-5"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

# llama.cpp の llama-server。OpenAI と同じ形の窓口を出す。
#
# これが要る理由は速さではない。手元の RX 6700 XT では torch が動かず、
# --run はGPUを使えない。llama.cpp は Vulkan で AMD のGPUを使えるので、
# Colab に行かずに手元で回せるようになる。
#
# 注意: llama.cpp ではロジットに手を入れられないので、
# 小説を書かせる側の抑圧(type5設定)は使えない。
# ただし論評では抑圧を使っていないため、こちらは問題にならない
LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"

# 待ち時間の上限(秒)。既定を1時間にしてあるのは、GPUに載りきらない
# 大きいモデルを一部CPUで回すと生成が桁違いに遅くなるため。
# 12GBのGPUに27Bを載せると2〜3 tok/s まで落ち、
# --tokens 4000 の生成に30分では足りないことがある
LLAMA_TIMEOUT = 3600

# 分割された GGUF の連番。…-00002-of-00003.gguf の 00002 を取る
SPLIT_PART = re.compile(r"-(\d{5})-of-\d{5}\.gguf$")


def model_folders():
    """GGUF が置かれうる場所。llama.cpp の版によって違う。

    -hf で落としたものは、いまの llama.cpp では HuggingFace の hub
    キャッシュに入る。古い版は独自の llama.cpp フォルダを使っていた。
    片方だけを見ると、落ちているのに「何も無い」と表示することになる。
    実際に外して、載っているモデルを見つけられなかった"""
    places = [Path.home() / ".cache" / "huggingface" / "hub"]
    told = os.environ.get("HF_HOME", "").strip()
    if told:
        places.append(Path(told) / "hub")
    told = os.environ.get("LLAMA_CACHE", "").strip()
    if told:
        places.append(Path(told))
    elif sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData"
                                                  / "Local")
        places.append(Path(root) / "llama.cpp")
    elif sys.platform == "darwin":
        places.append(Path.home() / "Library" / "Caches" / "llama.cpp")
    else:
        root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
        places.append(Path(root) / "llama.cpp")
    return [place for place in places if place.exists()]


# HuggingFace の hub は models--組織--名前 という形で掘る。
# ここから -hf に渡せるリポジトリ名を戻せる
HF_FOLDER = re.compile(r"^models--(.+?)--(.+)$")
# ファイル名の末尾に付く量子化の名前。-hf の :Q4_K_M に使う
QUANT = re.compile(r"-(UD-)?(I?Q\d[A-Z_0-9]*|BF16|F16|F32)"
                   r"(?:-\d{5}-of-\d{5})?\.gguf$", re.IGNORECASE)


def repo_of(path):
    """HuggingFace のキャッシュの中なら、リポジトリ名を戻す"""
    for parent in path.parents:
        found = HF_FOLDER.match(parent.name)
        if found:
            return f"{found.group(1)}/{found.group(2)}"
    return ""


def quant_of(path):
    """ファイル名から量子化の名前を取る"""
    found = QUANT.search(path.name)
    if not found:
        return ""
    return (found.group(1) or "") + found.group(2)


def downloaded_models(folders=None):
    """手元に落ちている GGUF を並べる。

    書き並べた一覧ではなく実際の中身を見る。
    何を持っているか覚えていなくても選べるようにするため。

    分割ファイル (…-00002-of-00003.gguf) は1つ目だけを出す。
    llama.cpp は1つ目を渡せば残りを自分で見つける"""
    if folders is None:
        folders = model_folders()
    elif isinstance(folders, (str, Path)):
        folders = [Path(folders)]
    found, seen = [], set()
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.gguf")):
            part = SPLIT_PART.search(path.name)
            if part and part.group(1) != "00001":
                continue
            if path.name.startswith("mmproj"):
                continue      # 画像を読むための付属品で、単体では使えない
            if path.name in seen:
                continue
            seen.add(path.name)
            found.append(path)
    return found


def hf_name(path):
    """-hf にそのまま渡せる形。リポジトリが分からなければファイル名"""
    repo, quant = repo_of(path), quant_of(path)
    if repo:
        return f"{repo}:{quant}" if quant else repo
    return path.stem


def describe_downloaded(path):
    """一覧に出す一行。-hf に渡せる名前と大きさ"""
    try:
        size = path.stat().st_size / (1024 ** 3)
    except OSError:
        return hf_name(path)
    return f"{hf_name(path)} ({size:.1f}GB)"


# 画面の選択肢に出す名前。ここに無いものも自由に打ち込めるので、
# 一覧は「よく使うものの近道」であって制限ではない。
# llama.cpp の欄は、立っているサーバに訊いた名前と、
# 手元に落ちている GGUF で埋める (KNOWN_MODELS の分は控えの見本)
KNOWN_MODELS = {
    "api": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
    "llama": ["LFM2.5-1.2B-JP-202606", "LFM2.5-2.6B", "LFM2-8B-A1B",
              "Qwen2.5-3B-Instruct"],
    "run": ["Qwen/Qwen2.5-3B-Instruct", "LiquidAI/LFM2-1.2B",
            "LiquidAI/LFM2-2.6B", "Qwen/Qwen2.5-7B-Instruct"],
}
MAX_TOKENS = 4000          # 論評は長い。小説生成の150では話にならない
CHUNK_SIZE = 6000
MODES = ["発想", "査読", "採点", "接続", "連想"]


class Local:
    """手元(か Colab)のモデルに解かせる。一度読み込んで使い回す。

    抑圧(type5設定)は入れない — あれは小説の読みやすさで検証したもので、
    分析の文章に効くかは確かめていない"""

    def __init__(self, model_name=LOCAL_MODEL):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()

    def __call__(self, head, body, max_tokens=MAX_TOKENS):
        return self.chat([{"role": "system", "content": head},
                          {"role": "user", "content": body}], max_tokens)

    def chat(self, messages, max_tokens=MAX_TOKENS):
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=True,
                temperature=0.7, min_p=0.05, top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


class Llama:
    """llama.cpp の llama-server に解かせる。

    先に別の窓でサーバを立てておく。

        llama-server -hf LiquidAI/LFM2.5-1.2B-JP-202606-GGUF --port 8080

    -hf は HuggingFace のリポジトリ名、-m は手元のファイルパス。
    -m にリポジトリ名を渡すと failed to open GGUF file になる。
    リポジトリには複数の量子化が入っているので :Q4_K_M のように付ける。

    AMD のGPUを使うなら Vulkan 版の llama.cpp を入れること"""

    def __init__(self, model="", url=LLAMA_URL, timeout=LLAMA_TIMEOUT):
        self.model = model or "local"
        self.url = url
        self.timeout = timeout
        # まだ一度も解いていない印。0.0 を印に使うと、往復が速すぎて
        # 時計の刻み(Windows では約15ms)に埋もれたときに
        # 「解いていない」と誤って判じる
        self.elapsed = None
        self.produced = None

    def __call__(self, head, body, max_tokens=MAX_TOKENS):
        return self.chat([{"role": "system", "content": head},
                          {"role": "user", "content": body}], max_tokens)

    def chat(self, messages, max_tokens=MAX_TOKENS):
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "messages": messages,
        }).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload,
            headers={"content-type": "application/json"})
        started = time.monotonic()
        try:
            # timeout はキーワードで渡すこと。位置引数の2番目は data で、
            # そこに秒数を渡すと本文として送ろうとして TypeError になる
            with urllib.request.urlopen(request,
                                        timeout=self.timeout) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"llama-server が断った "
                             f"({problem.code}): {detail}")
        except TimeoutError:
            raise SystemExit(
                f"llama-server が {self.timeout}秒で返さなかった。\n"
                "  サーバ側では生成が続いているので、待てば終わる見込みがある。\n"
                f"  --llama-timeout {self.timeout * 2} で待ち時間を延ばすか、\n"
                "  --tokens を減らすか、GPUに載せる層を増やすこと")
        except urllib.error.URLError as problem:
            # 待ち時間切れは URLError に包まれて来ることがある
            if isinstance(problem.reason, TimeoutError):
                raise SystemExit(
                    f"llama-server が {self.timeout}秒で返さなかった。\n"
                    f"  --llama-timeout {self.timeout * 2} で延ばすか、"
                    "--tokens を減らすこと")
            raise SystemExit(
                f"llama-server に届かない ({self.url}): {problem.reason}\n"
                "先に別の窓でサーバを立てること:\n"
                "  llama-server -hf LiquidAI/LFM2.5-1.2B-JP-202606-GGUF"
                " --port 8080")
        self.elapsed = time.monotonic() - started
        self.produced = (answer.get("usage") or {}).get("completion_tokens")
        return answer["choices"][0]["message"]["content"]

    def loaded(self, timeout=3):
        """サーバに載っているモデルの名前を訊く。

        llama-server は起動時に読み込んだ1つだけを配り、
        こちらが送る model の値は無視する。つまり --model に書いた名前は
        ただの自己申告で、実際とずれていても誰も気づかない。
        窓口に直接訊けば、本当に載っているものが分かる。

        訊けなければ空文字。記録が嘘になるより、空のほうがよい"""
        endpoint = self.url.rsplit("/chat/completions", 1)[0] + "/models"
        try:
            with urllib.request.urlopen(endpoint, timeout=timeout) as answer:
                found = json.load(answer)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return ""
        names = [row.get("id", "") for row in found.get("data", [])]
        return names[0] if names else ""

    def speed(self):
        """直前の生成の速さ。層の割り振りを決める材料になる"""
        if self.elapsed is None:
            return ""
        line = f"{self.elapsed:.0f}秒"
        if self.produced and self.elapsed > 0:
            line += (f"で{self.produced}トークン "
                     f"({self.produced / self.elapsed:.1f} tok/s)")
        elif self.produced:
            line += f"で{self.produced}トークン"
        return line


class Api:
    """Claude の API に解かせる。

    鍵は環境変数 ANTHROPIC_API_KEY から読むだけで、ここには書かない。
    設定するのは使う人の仕事で、この道具は値を見ない。

    余計な依存を増やさないよう、標準ライブラリだけで叩く。
    Colab で pip install せずに使えるほうが手間が少ない"""

    def __init__(self, model=API_MODEL):
        self.model = model
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not self.key:
            raise SystemExit(
                "ANTHROPIC_API_KEY が設定されていない。\n"
                "  Windows: setx ANTHROPIC_API_KEY \"自分の鍵\" "
                "(設定後に端末を開き直す)\n"
                "  Colab:   import os; "
                "os.environ['ANTHROPIC_API_KEY'] = '自分の鍵'\n"
                "鍵は https://console.anthropic.com で作る。"
                "この道具は鍵を保存も表示もしない。")

    def __call__(self, head, body, max_tokens=MAX_TOKENS):
        return self.chat([{"role": "system", "content": head},
                          {"role": "user", "content": body}], max_tokens)

    def chat(self, messages, max_tokens=MAX_TOKENS):
        """この窓口は system を messages の外に置く形なので、
        先頭の system だけ取り分ける"""
        head = "".join(m["content"] for m in messages
                       if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_tokens,
            "system": head,
            "messages": rest,
        }).encode("utf-8")
        request = urllib.request.Request(API_URL, data=payload, headers={
            "x-api-key": self.key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        })
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"APIが断った ({problem.code}): {detail}")
        except urllib.error.URLError as problem:
            raise SystemExit(f"APIに届かない: {problem.reason}")
        return "".join(part.get("text", "") for part in answer["content"]
                       if part.get("type") == "text")


def pick_chunk(manuscript, size, number):
    chunks = manuscript.chunks(size=size, overlap=1)
    if not chunks:
        raise SystemExit("切り出せる段落が無い")
    if number is None:
        return chunks, chunks[0]
    if not 1 <= number <= len(chunks):
        raise SystemExit(f"塊は1〜{len(chunks)}の範囲で指定する")
    return chunks, chunks[number - 1]


def pick_lenses(chunk, args, rng):
    """観点を決める。(観点の並び, どう決めたかの一言) を返す。

    既定は原稿を測って選ぶ。名前の出てこない原稿には「固有」を、
    出来事を語っていない原稿には「分岐」を向ける。
    --lens で名指しすれば、測定を無視してそれを使う"""
    mode = args.mode
    usable = lenses_for(mode)
    if args.lens:
        keys = [key.strip() for key in args.lens.replace("／", "/")
                .replace("、", ",").replace("/", ",").split(",")
                if key.strip()]
        unknown = [key for key in keys if key not in review.LENS_BY_KEY]
        if unknown:
            raise SystemExit(
                f"知らない観点: {'、'.join(unknown)}\n"
                f"使えるのは: {'、'.join(l.key for l in review.LENSES)}")
        # 提案を求める観点を査読で使うと、「本文に無い要素について
        # 述べない」という規則と矛盾する。名指しでも断る
        wrong = [key for key in keys if key not in {l.key for l in usable}]
        if wrong:
            raise SystemExit(
                f"{mode}モードでは使えない観点: {'、'.join(wrong)}\n"
                f"  提案を求める観点は査読の規則と矛盾する。\n"
                f"  {mode}で使えるのは: "
                f"{'、'.join(l.key for l in usable)}")
        return [review.LENS_BY_KEY[key] for key in keys], "指定"
    if args.random_lenses:
        return choose_lenses(rng, count=args.lenses, mode=mode), "くじ引き"
    lenses, measured = suggest_lenses(chunk.paragraphs, count=args.lenses,
                                      rng=rng, mode=mode)
    return lenses, "実測 " + describe(measured)


LENS_ADVICE = 3


def warn_lenses(count):
    """観点を増やしすぎたら言う。

    判明16 — 3Bモデルは指示を増やすと一つひとつが薄まる。禁止語6個では
    効いたが8個にすると逆に増えた。実際に観点を8つ渡した論評が、
    どの観点にも踏み込めないまま散らかった出力になった"""
    if count > LENS_ADVICE:
        print(f"# 注意: 観点が{count}個。{LENS_ADVICE}個までを勧める — "
              "増やすと一つひとつが薄まることを実測している。"
              "広く見たいなら --all で回すか、種を変えて何度か回すこと",
              file=sys.stderr)


def warn_partial_score(chunks, chunk, args):
    """採点で原稿の一部しか見ていないときに言う。

    採点は作品全体に対する評価なのに、既定では先頭の塊しか渡らない。
    実測では5分割の1つ目だけを見て「合計点 16/35」と全体評価が出た。
    どこを見た点数なのかが分からないまま溜まると、比べられなくなる"""
    if args.mode != "採点" or len(chunks) <= 1 or args.all:
        return
    where = chunks.index(chunk) + 1
    whole = sum(len(p.text) for part in chunks
                for p in part.paragraphs[part.carried:])
    print(f"# 注意: {len(chunks)}つに分けたうちの{where}つ目だけを採点している。"
          f"作品全体の点数ではない。\n"
          f"#   全体を見せるなら --size {whole + 500} 以上にするか、"
          f"--all で塊ごとに採点すること",
          file=sys.stderr)


def compose(manuscript, args, chunk=None, nudge=0):
    """モードに応じて (指示, 本文, 見せた段落, 添える説明) を作る。

    nudge は塊ごとに観点を変えるためのずらし。
    全部の塊に同じ観点を当てると、同じ角度の指摘が並ぶだけになる。
    一度に渡す観点を減らして塊ごとに入れ替えるのが Phase 9 の処方"""
    rng = random.Random(None if args.seed is None else args.seed + nudge)

    if args.mode in ("発想", "査読", "採点"):
        chunks, picked = pick_chunk(manuscript, args.size, args.chunk)
        chunk = chunk or picked
        warn_partial_score(chunks, chunk, args)
        # 採点は7観点が固定なので、観点を選ばない
        lenses, how = (pick_lenses(chunk, args, rng)
                       if uses_lenses(args.mode) else ((), ""))
        place = (f"{len(chunks)}つに分けたうちの"
                 f"{chunks.index(chunk) + 1}つ目、{chunk}")
        head, body = build_prompt(
            chunk, lenses, mode=args.mode, title=manuscript.title,
            author=manuscript.author, note=args.note, place=place,
            severity=args.severity, directory=args.prompts)
        allowed = {p.index for p in chunk.paragraphs}
        what = (f"{'／'.join(l.key for l in lenses)} [{how}]"
                if lenses else f"{args.severity}のルーブリック")
        return (head, body, allowed,
                f"{chunks.index(chunk) + 1}/{len(chunks)}塊 {what}")

    if args.mode == "接続":
        # 反復があればそれを優先する。作者がすでに植えた種のほうが確度が高い
        found = recurrences(manuscript)
        motifs = [item for item in found if item.kind == "反復"]
        if motifs:
            item = motifs[rng.randrange(min(len(motifs), args.top))]
            pair = item.pair()
            label = f"反復 {item}"
        else:
            pairs = distant_pairs(manuscript, count=args.top, rng=rng)
            if not pairs:
                raise SystemExit(
                    too_short(manuscript)
                    or "繋げる対が見つからない。"
                       f"どの二点も{MIN_CHARS}文字より近いか、語彙が重なっている。")
            pair = pairs[0]
            label = f"遠い対 {pair}"
        extra = [DREAM_WORK[rng.randrange(len(DREAM_WORK))]] if args.dream else ()
        head, body = build_connection_prompt(manuscript, pair,
                                             note=args.note, extra=extra)
        allowed = None
        return head, body, allowed, label

    # 連想
    usable = [p for p in manuscript.paragraphs if len(p.text) >= 40]
    if not usable:
        raise SystemExit("連想の起点にできる段落が無い")
    start = usable[rng.randrange(len(usable))]
    head, body = build_chain_prompt(manuscript, start, steps=args.steps,
                                    note=args.note)
    return head, body, None, f"起点 [{start.index}]"


def build_parser():
    parser = argparse.ArgumentParser(
        prog="centurion.critique",
        description="原稿を読ませて、論評か発想を得る")
    parser.add_argument("path", nargs="?",
                        help="原稿のファイル。省くと、手元のPCなら選択の窓、"
                             "Colab ならアップロードの窓が開く。"
                             "manuscripts/ に置いたものはファイル名だけでよい")
    parser.add_argument("--mode", default="発想", choices=MODES,
                        help="何を訊くか (既定 発想)。"
                             "接続は3000文字以上の原稿でしか成り立たない")
    parser.add_argument("--severity", default=review.DEFAULT_SEVERITY,
                        choices=list(rubric.SEVERITIES),
                        help="査読と採点の厳しさ "
                             f"(既定 {review.DEFAULT_SEVERITY})。"
                             "育成は良い点を先に述べ評価3を健闘とする。"
                             "厳格は商業水準を評価1とする。"
                             "発想・接続・連想では使わない")
    parser.add_argument("--prompts", metavar="置き場",
                        help="プロンプトの文面を読む場所 "
                             f"(既定 {wording.HOME.name}/)。"
                             "main.py prompts で既定値を書き出せる")
    parser.add_argument("--size", type=int, default=CHUNK_SIZE,
                        help=f"1塊の上限文字数 (既定 {CHUNK_SIZE})")
    parser.add_argument("--chunk", type=int,
                        help="何番目の塊を読ませるか (既定 1つ目)")
    parser.add_argument("--lenses", type=int, default=3,
                        help="一度に渡す観点の数 (既定 3)")
    parser.add_argument("--lens", metavar="視点,熱量",
                        help="観点を名指しする。測定を無視してこれを使う")
    parser.add_argument("--random-lenses", action="store_true",
                        help="原稿を測らず、くじ引きで観点を選ぶ")
    parser.add_argument("--survey", action="store_true",
                        help="原稿の実測と、観点ごとの必要度を出す")
    parser.add_argument("--words", choices=["正規表現", "形態素"],
                        default="正規表現",
                        help="反復を探すときの語の取り出し方 (既定 正規表現)。"
                             "形態素には fugashi と unidic-lite が要る")
    parser.add_argument("--top", type=int, default=5,
                        help="接続モードで候補の上位いくつから選ぶか")
    parser.add_argument("--steps", type=int, default=4,
                        help="連想モードで何歩たどらせるか")
    parser.add_argument("--dream", action="store_true",
                        help="接続モードに夢の作業を一つ添える")
    parser.add_argument("--note", default="",
                        help="作者からの補足。狙いや訊きたいこと")
    parser.add_argument("--seed", type=int, help="観点や対の選び方を固定する")
    parser.add_argument("--list", action="store_true",
                        help="塊と反復の一覧だけを出す")
    parser.add_argument("--run", action="store_true",
                        help="その場で手元のモデルに解かせる。GPUが要る")
    parser.add_argument("--api", action="store_true",
                        help="その場で Claude の API に解かせる。"
                             "鍵は環境変数 ANTHROPIC_API_KEY から読む")
    parser.add_argument("--llama", action="store_true",
                        help="立ててある llama-server に解かせる。"
                             "AMDのGPUでも動く")
    parser.add_argument("--llama-timeout", type=int, default=LLAMA_TIMEOUT,
                        metavar="秒",
                        help=f"llama-server を待つ上限秒数 "
                             f"(既定 {LLAMA_TIMEOUT})。"
                             "GPUに載りきらないモデルを一部CPUで回すと"
                             "生成が桁違いに遅くなるので、そのときは延ばす")
    parser.add_argument("--models", action="store_true",
                        help="手元に落ちている GGUF と、立っている"
                             " llama-server に載っているモデルを並べる")
    parser.add_argument("--llama-url", default=LLAMA_URL,
                        help=f"llama-server の窓口 (既定 {LLAMA_URL})")
    parser.add_argument("--all", action="store_true",
                        help="全部の塊を順に読ませる (発想・査読のみ)")
    parser.add_argument("--verify", action="store_true",
                        help="出てきた指摘を、もう一度モデルに検分させる。"
                             "迷ったら捨てる側に立たせ、通ったものだけ残す")
    parser.add_argument("--verify-with", choices=["api", "llama", "run"],
                        help="検分を別の経路にやらせる (既定は同じ経路)")
    parser.add_argument("--verify-model",
                        help="検分に使うモデル (既定は同じモデル)")
    parser.add_argument("--verify-llama-url",
                        help="検分させる llama-server の窓口。"
                             "別のモデルに検分させるには、別のポートで"
                             "もう一つサーバを立ててここを指す")
    parser.add_argument("--model",
                        help=f"使うモデル (手元は {LOCAL_MODEL}、"
                             f"APIは {API_MODEL})")
    parser.add_argument("--tokens", type=int, default=MAX_TOKENS,
                        help=f"答えの長さの上限 (既定 {MAX_TOKENS})。"
                             "--api / --run のときだけ効く")
    parser.add_argument("--check", metavar="答え",
                        help="答えのファイルを読み、段落番号と引用を検査する")
    parser.add_argument("--out", metavar="添削.txt",
                        help="本文の各段落の下に指摘を貼った添削ファイルを書く")
    return parser


def too_short(manuscript):
    """遠い二点が取れない原稿かどうかを、理由つきで伝える文。取れるなら空"""
    if len(manuscript.text) >= MIN_CHARS * 2:
        return ""
    return (f"この原稿は{len(manuscript.text)}文字で、"
            f"遠い二点と呼べる間隔({MIN_CHARS}文字)が取れない。\n"
            "反復も対も出ないのは仕組みの不具合ではなく、繋ぐ先が無いため。\n"
            "接続と連想は長い原稿で使うもので、短い作品には査読と発想を使う。")


def show_list(manuscript, args):
    print(manuscript.summary())
    chunks = manuscript.chunks(size=args.size, overlap=1)
    print(f"\n{args.size}文字ずつに切ると {len(chunks)}塊")
    for chunk in chunks:
        print("  " + str(chunk))

    found = recurrences(manuscript)
    if not found:
        print("\n反復・主題 0件")
        print(too_short(manuscript)
              or "遠くで繰り返される稀な語が見つからない。")
        return
    for kind in ("反復", "主題"):
        rows = [item for item in found if item.kind == kind]
        print(f"\n{kind} {len(rows)}件" + (" (上位12)" if len(rows) > 12 else ""))
        for item in rows[:12]:
            print("  " + str(item))


def show_survey(manuscript, args):
    """原稿の実測と、そこから出た観点の必要度を並べる。
    なぜその観点が選ばれるのかを見えるようにするため"""
    from .review import needs

    _, chunk = pick_chunk(manuscript, args.size, args.chunk)
    score, measured = needs(chunk.paragraphs)
    print(f"{manuscript.title or '原稿'} / {chunk}")
    print("\n実測")
    labels = {
        "名前": "名前のある人や場所を含む段落",
        "会話": "会話文の段落",
        "出来事": "過去の出来事を2文以上語る段落",
        "感覚": "使われている感覚の種類",
        "一人称": "一人称を含む段落",
        "偏り": "段落の長さのばらつき",
        "混在": "ですます体とである体の混ざり具合",
        "轍": "常套語(宇宙・神秘・永遠…)の濃さ",
    }
    for key, value in measured.items():
        print(f"  {labels.get(key, key):<28} {value:>5.0%}")

    print("\n観点の必要度 (高いほどこの原稿に効くと見込まれる)")
    for lens in sorted(review.LENSES, key=lambda l: -score[l.key]):
        print(f"  {score[lens.key]:>5.0%}  【{lens.key}】{lens.group}")
    print("\n上から群を散らして選ぶ。"
          "--lens 視点,熱量 で名指しすれば測定を無視する")


def run_check(manuscript, args):
    """答えの段落番号を検査する。

    --chunk を渡すと、そのとき見せた範囲の外を指した指摘も見つける。
    番号が実在することと、モデルがその段落を読んでいたことは別で、
    見せていない段落への言及は中身を確かめずに書いたものになる。
    実際にこの検査を作った日に、範囲外の段落を2件引いて
    どちらも中身を取り違えた例が出た"""
    answer = open(args.check, encoding="utf-8").read()
    allowed = None
    if args.chunk is not None:
        _, chunk = pick_chunk(manuscript, args.size, args.chunk)
        allowed = {p.index for p in chunk.paragraphs}

    real, missing, outside = check_citations(answer, manuscript,
                                             allowed=allowed)
    total = len(real) + len(missing) + len(outside)
    print(f"示された段落番号 {total}件")
    print(f"  実在する {len(real)}件: {sorted(set(real))}")
    if missing:
        print(f"  × 存在しない {len(missing)}件: {sorted(set(missing))}")
        print("    実在しない箇所への指摘。捨てること")
    if outside:
        print(f"  △ 見せていない範囲 {len(outside)}件: {sorted(set(outside))}")
        print("    番号は実在するが、この読みでは渡していない段落。")
        print("    中身を確かめずに書いている可能性が高い")
    if not missing and not outside:
        print("  すべて渡した範囲の中にある"
              if allowed else "  存在しない番号は無し")
    if allowed is None:
        print("  ※ --chunk を渡すと、見せていない範囲への言及も検出できる")

    quotes = find_quotes(answer, manuscript)
    print()
    print(report_quotes(quotes))

    # 外で解かせた採点も検算する。ここが --mode 採点 を渡す唯一の理由
    counted = score_lines(answer, args.mode)
    bad = bool(missing or outside or any(not q.ok for q in quotes)
               or len(counted) > 1)
    if counted:
        print()
        print("\n".join(counted))

    if args.out:
        write_annotated(args.out, answer, manuscript,
                        annotation_records(args, manuscript,
                                           source=args.check),
                        mode=args.mode)
        return 1 if bad else 0

    print("\n--- 番号を本文に戻したもの ---")
    print(resolve(answer, manuscript))
    return 1 if bad else 0


def show_models(args):
    """何が使えるのかを実測で並べる。

    書き並べた一覧では、落としてあるものも、いま載っているものも
    分からない。置き場を見て、サーバに訊く"""
    loaded = Llama(url=args.llama_url).loaded()
    print(f"llama-server ({args.llama_url})")
    if loaded:
        print(f"  載っているのは {loaded}")
    else:
        print("  立っていないか、名前を答えない")

    places = model_folders()
    print()
    print("置き場")
    for place in places:
        print(f"  {place}")
    if not places:
        print("  見つからない (まだ何も落としていない)")

    found = downloaded_models()
    print()
    print(f"落ちている GGUF {len(found)}件")
    for path in found:
        mark = "→" if loaded and hf_name(path) == loaded else " "
        print(f"  {mark} {describe_downloaded(path)}")
    if found:
        print()
        print("使うには、別の窓でサーバを立て直すこと:")
        print(f"  llama-server -hf {hf_name(found[0])} --port 8080 -ngl 99")
    return 0


def used_model(args):
    """どのモデルに解かせたかを一行で。
    貼り付けた答えを検査するときは、こちらには分からない"""
    if args.api:
        return f"{args.model or API_MODEL} (API)"
    if args.llama:
        # サーバに直接訊いた名前があればそれを使う。--model は自己申告で、
        # llama-server はその値を無視するので、書いた名前と載っている
        # モデルが違っていても誰も気づかない。実際に取り違えて記録していた
        real = getattr(args, "llama_loaded", "")
        if real:
            same = args.model and args.model not in real
            return (f"{real} (llama.cpp が申告)"
                    + (f" ※--model には {args.model} と書かれていた"
                       if same else ""))
        return f"{args.model or '不明'} (llama.cpp・名前は未確認)"
    if args.run:
        return f"{args.model or LOCAL_MODEL} (手元)"
    if args.model:
        return f"{args.model} (申告)"
    return "不明 (外で解かせた答え)"


def used_wording(args):
    """外部ファイルの文面を使ったなら、どれを使ったかを一行で。

    prompts/ にある全部を並べると嘘になる。一回の実行で読むのは
    そのモードと厳しさのファイルだけで、他は使われていない"""
    if not uses_lenses(args.mode) and args.mode not in ("採点",):
        return ""          # 接続と連想は prompts/ を見ない
    folder = wording.home(args.prompts)
    where = args.prompts or folder.name
    files = []
    name = review.heading_name(args.mode, args.severity)
    if wording.path(name, args.prompts).exists():
        files.append(f"{name}.txt")
    if wording.path("観点", args.prompts).exists() and uses_lenses(args.mode):
        files.append("観点.txt")
    return f"{where}/ の {'、'.join(files)}" if files else ""


def annotation_records(args, manuscript, labels=(), source="", outcome=""):
    """添削ファイルの見出しに残すもの。
    どのモードで、どのモデルに、どの範囲を、どの観点で読ませたか。

    貼り付けた答えを検査した場合は、こちらでは範囲も観点も分からない。
    その代わり、どのファイルを検査したかを残す"""
    records = [
        ("日付", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("モード", args.mode if not source else f"{args.mode} (申告)"),
        ("モデル", used_model(args)),
    ]
    # 厳しさは査読と採点でしか効かない。効かないモードで書くと誤解を招く
    if args.mode in ("査読", "採点"):
        records.append(("厳しさ", args.severity))
    used = used_wording(args)
    if used:
        records.append(("文面", used))
    records += [
        ("原稿", f"{len(manuscript.text)}文字 / "
                 f"{len(manuscript.paragraphs)}段落"),
    ]
    if source:
        records.append(("検査した答え", Path(source).name))
    if labels:
        records.append(("読ませた範囲と観点", " / ".join(labels)))
    if args.verify:
        # 何に検分させたかは verifier() が決める。ここで --verify-model を
        # そのまま書くと、llama.cpp では嘘になる (サーバは名前を無視するため)。
        # 何件残って何件捨てられたかまで残す — 誰が検分したかだけでは、
        # 働いたのか素通りしたのかが後から分からない
        records.append(("検証", getattr(args, "verify_where", "未実行")
                        + (f" — {outcome}" if outcome else "")))
    records.append(("語の取り出し", args.words))
    return records


def write_annotated(path, answer, manuscript, records=(), mode=""):
    # 採点なら、添削ファイルの見出しにも検算を残す
    axes = rubric.AXES if mode == "採点" else ()
    text = annotate(answer, manuscript, records=records, axes=axes)
    with open(path, "w", encoding="utf-8") as out:
        out.write(text + "\n")
    print(f"# 添削を {path} に書いた", file=sys.stderr)


def tasks(manuscript, args):
    """読ませる仕事の並びを作る。--all なら塊の数だけ並ぶ"""
    if not args.all:
        return [compose(manuscript, args)]
    if args.mode not in ("発想", "査読", "採点"):
        raise SystemExit("--all は発想・査読・採点でだけ使える")
    chunks, _ = pick_chunk(manuscript, args.size, None)
    return [compose(manuscript, args, chunk=chunk, nudge=index)
            for index, chunk in enumerate(chunks)]


def verifier(args):
    """検分する側を用意する。(解き手, 記録に残す説明) を返す。

    llama.cpp は起動時に読み込んだモデル1つだけを配る。--model / --verify-model
    はサーバに送られるが無視される。つまり同じ窓口を指している限り、
    --verify-model に別の名前を書いても**検分するのは同じモデル**である。
    以前はそれを「検証: Qwen (llama)」と記録していて、事実と違っていた。

    別のモデルに検分させるには、別のポートでもう一つ llama-server を立て、
    --verify-llama-url でそちらを指すこと"""
    if args.verify_with == "api" or (args.verify_with is None and args.api):
        model = args.verify_model or args.model or API_MODEL
        return Api(model), f"{model} (API)"
    if args.verify_with == "llama" or (args.verify_with is None and args.llama):
        url = args.verify_llama_url or args.llama_url
        same = url == args.llama_url and args.llama
        if same:
            where = f"llama.cpp {url} に載っているモデル (書いた側と同じ)"
        else:
            where = f"llama.cpp {url}" + (f" / {args.verify_model}"
                                          if args.verify_model else "")
        return Llama(args.verify_model or "", url,
                     args.llama_timeout), where
    model = args.verify_model or args.model or LOCAL_MODEL
    return Local(model), f"{model} (手元)"


def warn_same_judge(args):
    """書いた側と検分する側が同じモデルになっていたら言う。
    同じモデルは自分の答えを通しがちで、検証にならない"""
    if not args.verify or not args.llama:
        return
    if args.verify_with not in (None, "llama"):
        return
    if (args.verify_llama_url or args.llama_url) != args.llama_url:
        return
    message = ("# 注意: 検分するのも同じ llama-server、"
               "つまり書いた側と同じモデルになる。"
               "自分の答えは通しがちなので検証になりにくい")
    if args.verify_model:
        message += (f"\n#   --verify-model {args.verify_model} は"
                    "サーバに無視される。載っているモデルは起動時に決まっている")
    message += ("\n#   別のモデルに検分させるには、別のポートでもう一つ"
                "サーバを立てて --verify-llama-url で指すこと")
    print(message, file=sys.stderr)


def run_verify(args, manuscript, answer, body_text):
    """指摘を検分にかけ、残ったものだけで答えを組み直す。
    (組み直した答え, 経過の説明) を返す"""
    findings = verify.split_findings(answer, manuscript)
    if not findings:
        return answer, "検証: 検分できる指摘が見当たらない"

    solve, where = verifier(args)
    args.verify_where = where          # 記録に残すために控える
    head, body = verify.build_prompt(manuscript, findings, body_text,
                                     title=manuscript.title)
    judged = solve(head, body, args.tokens)
    verdicts = verify.parse_verdicts(judged, findings)
    kept, dropped, unjudged = verify.sift(findings, verdicts)

    # 一件も判定できなかったなら、検証は働かなかった。
    # 全部の行に「判定されなかった」と貼るのは、
    # 何も分からなかったことを分かったように見せるだけで害になる
    if not kept and not dropped:
        return answer, (
            f"検証は働かなかった ({where} が判定の形で答えなかった)。"
            f"元の答えをそのまま出す。\n"
            f"  指摘{len(findings)}件を渡したが、読み取れた判定は0件。\n"
            f"  観点を減らすか、検分を別の(より大きい)モデルにさせること。\n"
            f"  返ってきた先頭: {judged.strip()[:60]}…")

    if not kept and not unjudged:
        return answer, ("検証: すべて捨てられた。"
                        "検分が働きすぎている疑いがあるので元の答えを残す\n"
                        + verify.report(kept, dropped, unjudged))
    return (verify.rebuild(kept, unjudged),
            f"検証を {where} で行った\n"
            + verify.report(kept, dropped, unjudged))


def verify_outcome(note):
    """検証の経過から、記録に残す一行を作る"""
    for line in (note or "").splitlines():
        if "件を残し" in line or "働かなかった" in line or "すべて捨て" in line:
            return line.replace("検証: ", "").replace("検証を ", "").strip()
    return ""


def report(answer, manuscript, allowed, mode=""):
    """答えを検査して、気になるものだけ伝える。

    番号の実在と、引用の中身の両方を見る。
    番号だけの検査では、Qwen が引用14件中8件を取り違えた答えを
    「実在18件」として素通りさせた。
    採点モードならさらに点数を検算する — LFM2.5 が
    4+3+5+5+4+5+4=30 を「合計点 16/35」と書いた"""
    real, missing, outside = check_citations(answer, manuscript,
                                             allowed=allowed)
    parts = [f"実在{len(real)}件"]
    if missing:
        parts.append(f"存在しない{len(missing)}件 {sorted(set(missing))}"
                     " ← この指摘は捨てること")
    if outside:
        parts.append(f"見せていない範囲{len(outside)}件 {sorted(set(outside))}"
                     " ← 中身を確かめずに書いている")
    print("# 段落番号 " + " / ".join(parts), file=sys.stderr)

    # 引用の照合は「引用があるもの」しか見られない。
    # どこも指さない指摘ばかりの答えは、照合を素通りして
    # 「不一致0件」と出てしまう。錨の数を先に見る
    anchored, total = anchoring(answer, manuscript)
    for line in report_anchoring(anchored, total).splitlines():
        print("# " + line.strip(), file=sys.stderr)

    quotes = find_quotes(answer, manuscript)
    for line in report_quotes(quotes).splitlines():
        print("# " + line, file=sys.stderr)

    counted = score_lines(answer, mode)
    for line in counted:
        print("# " + line, file=sys.stderr)

    return bool(missing or outside or [q for q in quotes if not q.ok]
                or (total and anchored / total < 0.5)
                or len(counted) > 1)


def score_lines(answer, mode):
    """採点モードなら検算の行を返す。それ以外では空"""
    return check_scores(answer, rubric.AXES) if mode == "採点" else []


def main(argv=None):
    args = build_parser().parse_args(argv)
    if review.load_wording(args.prompts):
        print(f"# 観点を {wording.path('観点', args.prompts)} から読んだ",
              file=sys.stderr)
    if args.words == "形態素":
        try:
            use_morphology(True)
        except ImportError:
            raise SystemExit(
                "形態素解析には fugashi と unidic-lite が要る。\n"
                "  pip install fugashi unidic-lite")
        print("# 語の取り出しに形態素解析を使う", file=sys.stderr)
    if args.models:
        return show_models(args)
    manuscript = Manuscript.load(args.path)

    if args.check:
        return run_check(manuscript, args)
    if args.list:
        show_list(manuscript, args)
        return 0
    if args.survey:
        show_survey(manuscript, args)
        return 0
    if uses_lenses(args.mode) and not args.lens:
        warn_lenses(args.lenses)
    warn_same_judge(args)

    jobs = tasks(manuscript, args)

    if not args.run and not args.api and not args.llama:
        # 貼りやすい形で出す。指示と本文の境目を残す
        for index, (head, body, _, label) in enumerate(jobs):
            print(f"# {args.mode}モード / {label}", file=sys.stderr)
            if index:
                print("\n" + "=" * 64 + "\n")
            print(head)
            print()
            print("---")
            print()
            print(body)
        print("\n# 答えを得たら check で段落番号を検査すること",
              file=sys.stderr)
        return 0

    if args.api:
        model = args.model or API_MODEL
        print(f"# {model} に訊いています…", file=sys.stderr)
        solve = Api(model)
    elif args.llama:
        solve = Llama(args.model or "", args.llama_url,
                      args.llama_timeout)
        # 何が載っているかをサーバに訊く。--model は無視されるので、
        # ここで訊かないと記録が自己申告のままになる
        args.llama_loaded = solve.loaded()
        aboard = args.llama_loaded or "名前を答えない"
        print(f"# {args.llama_url} の llama-server に訊いています "
              f"(載っているのは {aboard})", file=sys.stderr)
    else:
        model = args.model or LOCAL_MODEL
        print(f"# {model} を読み込んでいます…", file=sys.stderr)
        solve = Local(model)

    collected = []
    for index, (head, body, allowed, label) in enumerate(jobs):
        print(f"# {args.mode}モード / {label}", file=sys.stderr)
        if index:
            print("\n" + "=" * 64 + "\n")
        answer = solve(head, body, args.tokens)
        # 何 tok/s 出ているかを見せる。層の割り振りを詰めるときの材料になる
        pace = solve.speed() if hasattr(solve, "speed") else ""
        if pace:
            print(f"# {pace}", file=sys.stderr)
        outcome = ""
        if args.verify:
            print("# 検分にかけています…", file=sys.stderr)
            answer, how = run_verify(args, manuscript, answer, body)
            for line in how.splitlines():
                print("# " + line, file=sys.stderr)
            outcome = verify_outcome(how)
        print(answer)
        report(answer, manuscript, allowed, args.mode)
        collected.append((answer, label, outcome))

    if args.out:
        write_annotated(
            args.out, "\n\n".join(answer for answer, _, _ in collected),
            manuscript,
            annotation_records(
                args, manuscript, [label for _, label, _ in collected],
                outcome=" / ".join(o for _, _, o in collected if o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

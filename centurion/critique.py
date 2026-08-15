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
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from .answer import annotate, find_quotes, report_quotes
from .connect import (DREAM_WORK, MIN_CHARS, build_chain_prompt,
                      build_connection_prompt, distant_pairs, recurrences,
                      use_morphology)
from .manuscript import Manuscript
from .review import (LENS_BY_KEY, LENSES, build_prompt, check_citations,
                     choose_lenses, describe, resolve, suggest_lenses)

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

# 画面の選択肢に出す名前。ここに無いものも自由に打ち込めるので、
# 一覧は「よく使うものの近道」であって制限ではない。
# llama.cpp の欄は記録用の名札で、実際に何が載っているかはサーバ側で決まる
KNOWN_MODELS = {
    "api": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
    "llama": ["LFM2.5-1.2B-JP-202606", "LFM2.5-2.6B", "LFM2-8B-A1B",
              "Qwen2.5-3B-Instruct"],
    "run": ["Qwen/Qwen2.5-3B-Instruct", "LiquidAI/LFM2-1.2B",
            "LiquidAI/LFM2-2.6B", "Qwen/Qwen2.5-7B-Instruct"],
}
MAX_TOKENS = 4000          # 論評は長い。小説生成の150では話にならない
CHUNK_SIZE = 6000
MODES = ["発想", "査読", "接続", "連想"]


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

    GGUF を手元に落としてあるなら -m 置き場所.gguf でもよい。
    AMD のGPUを使うなら Vulkan 版の llama.cpp を入れること"""

    def __init__(self, model="", url=LLAMA_URL):
        self.model = model or "local"
        self.url = url

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
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as problem:
            detail = problem.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"llama-server が断った "
                             f"({problem.code}): {detail}")
        except urllib.error.URLError as problem:
            raise SystemExit(
                f"llama-server に届かない ({self.url}): {problem.reason}\n"
                "先に別の窓でサーバを立てること:\n"
                "  llama-server -hf LiquidAI/LFM2.5-1.2B-JP-202606-GGUF"
                " --port 8080")
        return answer["choices"][0]["message"]["content"]


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
    if args.lens:
        keys = [key.strip() for key in args.lens.replace("／", "/")
                .replace("、", ",").replace("/", ",").split(",")
                if key.strip()]
        unknown = [key for key in keys if key not in LENS_BY_KEY]
        if unknown:
            raise SystemExit(
                f"知らない観点: {'、'.join(unknown)}\n"
                f"使えるのは: {'、'.join(l.key for l in LENSES)}")
        return [LENS_BY_KEY[key] for key in keys], "指定"
    if args.random_lenses:
        return choose_lenses(rng, count=args.lenses), "くじ引き"
    lenses, measured = suggest_lenses(chunk.paragraphs, count=args.lenses,
                                      rng=rng)
    return lenses, "実測 " + describe(measured)


def compose(manuscript, args, chunk=None, nudge=0):
    """モードに応じて (指示, 本文, 見せた段落, 添える説明) を作る。

    nudge は塊ごとに観点を変えるためのずらし。
    全部の塊に同じ観点を当てると、同じ角度の指摘が並ぶだけになる。
    一度に渡す観点を減らして塊ごとに入れ替えるのが Phase 9 の処方"""
    rng = random.Random(None if args.seed is None else args.seed + nudge)

    if args.mode in ("発想", "査読"):
        chunks, picked = pick_chunk(manuscript, args.size, args.chunk)
        chunk = chunk or picked
        lenses, how = pick_lenses(chunk, args, rng)
        place = (f"{len(chunks)}つに分けたうちの"
                 f"{chunks.index(chunk) + 1}つ目、{chunk}")
        head, body = build_prompt(
            chunk, lenses, mode=args.mode, title=manuscript.title,
            author=manuscript.author, note=args.note, place=place)
        allowed = {p.index for p in chunk.paragraphs}
        return (head, body, allowed,
                f"{chunks.index(chunk) + 1}/{len(chunks)}塊 "
                f"{'／'.join(l.key for l in lenses)} [{how}]")

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
    parser.add_argument("--llama-url", default=LLAMA_URL,
                        help=f"llama-server の窓口 (既定 {LLAMA_URL})")
    parser.add_argument("--all", action="store_true",
                        help="全部の塊を順に読ませる (発想・査読のみ)")
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
        "轍": "常套語(宇宙・神秘・永遠…)の濃さ",
    }
    for key, value in measured.items():
        print(f"  {labels.get(key, key):<28} {value:>5.0%}")

    print("\n観点の必要度 (高いほどこの原稿に効くと見込まれる)")
    for lens in sorted(LENSES, key=lambda l: -score[l.key]):
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

    if args.out:
        write_annotated(args.out, answer, manuscript,
                        annotation_records(args, manuscript,
                                           source=args.check))
        return 1 if missing or outside or any(not q.ok for q in quotes) else 0

    print("\n--- 番号を本文に戻したもの ---")
    print(resolve(answer, manuscript))
    return 1 if missing or outside or any(not q.ok for q in quotes) else 0


def used_model(args):
    """どのモデルに解かせたかを一行で。
    貼り付けた答えを検査するときは、こちらには分からない"""
    if args.api:
        return f"{args.model or API_MODEL} (API)"
    if args.llama:
        return f"{args.model or '不明'} (llama.cpp)"
    if args.run:
        return f"{args.model or LOCAL_MODEL} (手元)"
    if args.model:
        return f"{args.model} (申告)"
    return "不明 (外で解かせた答え)"


def annotation_records(args, manuscript, labels=(), source=""):
    """添削ファイルの見出しに残すもの。
    どのモードで、どのモデルに、どの範囲を、どの観点で読ませたか。

    貼り付けた答えを検査した場合は、こちらでは範囲も観点も分からない。
    その代わり、どのファイルを検査したかを残す"""
    records = [
        ("日付", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("モード", args.mode if not source else f"{args.mode} (申告)"),
        ("モデル", used_model(args)),
        ("原稿", f"{len(manuscript.text)}文字 / "
                 f"{len(manuscript.paragraphs)}段落"),
    ]
    if source:
        records.append(("検査した答え", Path(source).name))
    if labels:
        records.append(("読ませた範囲と観点", " / ".join(labels)))
    records.append(("語の取り出し", args.words))
    return records


def write_annotated(path, answer, manuscript, records=()):
    text = annotate(answer, manuscript, records=records)
    with open(path, "w", encoding="utf-8") as out:
        out.write(text + "\n")
    print(f"# 添削を {path} に書いた", file=sys.stderr)


def tasks(manuscript, args):
    """読ませる仕事の並びを作る。--all なら塊の数だけ並ぶ"""
    if not args.all:
        return [compose(manuscript, args)]
    if args.mode not in ("発想", "査読"):
        raise SystemExit("--all は発想と査読でだけ使える")
    chunks, _ = pick_chunk(manuscript, args.size, None)
    return [compose(manuscript, args, chunk=chunk, nudge=index)
            for index, chunk in enumerate(chunks)]


def report(answer, manuscript, allowed):
    """答えを検査して、気になるものだけ伝える。

    番号の実在と、引用の中身の両方を見る。
    番号だけの検査では、Qwen が引用14件中8件を取り違えた答えを
    「実在18件」として素通りさせた"""
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

    quotes = find_quotes(answer, manuscript)
    for line in report_quotes(quotes).splitlines():
        print("# " + line, file=sys.stderr)
    return bool(missing or outside or [q for q in quotes if not q.ok])


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.words == "形態素":
        try:
            use_morphology(True)
        except ImportError:
            raise SystemExit(
                "形態素解析には fugashi と unidic-lite が要る。\n"
                "  pip install fugashi unidic-lite")
        print("# 語の取り出しに形態素解析を使う", file=sys.stderr)
    manuscript = Manuscript.load(args.path)

    if args.check:
        return run_check(manuscript, args)
    if args.list:
        show_list(manuscript, args)
        return 0
    if args.survey:
        show_survey(manuscript, args)
        return 0

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
        print(f"# {args.llama_url} の llama-server に訊いています…",
              file=sys.stderr)
        solve = Llama(args.model or "", args.llama_url)
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
        print(answer)
        report(answer, manuscript, allowed)
        collected.append((answer, label))

    if args.out:
        write_annotated(
            args.out, "\n\n".join(answer for answer, _ in collected),
            manuscript,
            annotation_records(args, manuscript,
                               [label for _, label in collected]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
ブラウザから使うサーバの試験。

    python tests/test_web.py

モデルは呼ばない。確かめるのは配管と、外に出してはいけないもの —
manuscripts/ の外を開かせないこと、本文をそのまま HTML に流さないこと、
画面からの指定が CLI と同じ引数に化けること。
"""

import html
import io
import json
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import centurion.critique as critique
import centurion.web as web
from centurion.manuscript import Manuscript

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ○ {name}")
    else:
        failed += 1
        print(f"  × {name}" + (f" — {detail}" if detail else ""))


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as answer:
        return answer.read().decode("utf-8")


def post(path, body):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as answer:
        return json.loads(answer.read().decode("utf-8"))


print("== 引数への組み替え ==")
# 画面と端末で振る舞いがずれないよう、指定は CLI と同じ argv に組む
argv = web.to_argv({"name": "あっちゃぐり.txt", "mode": "査読", "size": 3000,
                    "chunk": 2, "lensmode": "named", "lens": "視点,熱量",
                    "lenses": 2, "note": "狙い", "engine": "llama",
                    "model": "LFM2.5"})
check("モードが入る", "--mode" in argv and "査読" in argv)
check("塊の指定が入る", argv[argv.index("--chunk") + 1] == "2")
check("名指しした観点が入る", argv[argv.index("--lens") + 1] == "視点,熱量")
check("補足が入る", argv[argv.index("--note") + 1] == "狙い")
check("解かせ方が入る", "--llama" in argv)
check("モデル名が入る", argv[argv.index("--model") + 1] == "LFM2.5")

argv = web.to_argv({"name": "あっちゃぐり.txt", "lensmode": "random"})
check("くじ引きを選べる", "--random-lenses" in argv)
check("観点を名指ししなければ渡さない", "--lens" not in argv)
argv = web.to_argv({"name": "あっちゃぐり.txt", "lensmode": "named", "lens": " "})
check("空の名指しは渡さない", "--lens" not in argv)
argv = web.to_argv({"name": "あっちゃぐり.txt", "engine": ""})
check("解かせ方を選ばなければ何も付かない",
      not {"--llama", "--api", "--run"} & set(argv))

check("実際に CLI が受け取れる形になっている",
      __import__("centurion.critique", fromlist=["x"])
      .build_parser().parse_args(
          web.to_argv({"name": "あっちゃぐり.txt", "mode": "接続",
                       "engine": "api"})).mode == "接続")

print("\n== 置き場の外を開かせない ==")
for name in ("../README.md", "..\\README.md", "/etc/passwd",
             "C:\\Windows\\win.ini", "無い原稿.txt"):
    try:
        web.resolve(name)
        blocked = False
    except (ValueError, OSError):
        blocked = True
    check(f"{name} を断る", blocked)

check("置き場の説明書は原稿として出さない",
      "README.md" not in web.listing(), str(web.listing()))
check("原稿は出す", any(n.endswith(".txt") for n in web.listing()),
      str(web.listing()))

print("\n== 本文を HTML に流し込むとき ==")
# 原稿は作者のもので、< や & が入りうる。そのまま流すと画面が壊れる
risky = Manuscript("　彼は<b>と書いた。それから&nbsp;と続けた。\n\n"
                   "　<script>alert(1)</script> という一行もある。")
drawn = web.render(risky, "[0] は良い。", [("モード", "発想")])
check("山括弧を逃がす", "&lt;b&gt;" in drawn, drawn[:120])
check("素のタグを出さない", "<script>" not in drawn)
check("アンパサンドを逃がす", "&amp;nbsp;" in drawn)
check("記録が入る", "モード: 発想" in drawn)
check("中身の無い記録は出さない",
      "空:" not in web.render(risky, "", [("空", "")]))

drawn = web.render(risky, "[9999] 「本文にどこにも無い一文である。」", risky
                   and [("モード", "発想")])
check("食い違う指摘に印が付く", web.MARK_BAD in drawn or "tip bad" in drawn,
      drawn[-200:])

drawn = web.render(risky, "", [("モード", "発想")], prompt="問いの本文")
check("問いだけのときは問いを出す", "問いの本文" in drawn)
check("そのときは貼り方を添える", "貼り" in drawn)

print("\n== サーバを立てて叩く ==")
threading.Thread(target=web.serve, args=(PORT, False), daemon=True).start()
for _ in range(40):
    try:
        get("/api/manuscripts")
        break
    except Exception:
        time.sleep(0.1)

page = get("/")
check("画面が返る", "センチュリオン" in page and "<html" in page)
check("画面に依存の読み込みが無い", "http://" not in page.split("</style>")[0])

names = json.loads(get("/api/manuscripts"))
check("原稿の一覧が返る", isinstance(names, list) and names, str(names))

data = json.loads(get(f"/api/manuscript?name={quote(names[0])}&size=6000"))
check("原稿の姿が返る", "title" in data and "summary" in data)
check("塊が返る", isinstance(data["chunks"], list) and data["chunks"])
check("実測が返る", len(data["survey"]) >= 5)
check("必要度が返る", len(data["needs"]) >= 10)
check("必要度は高い順", data["needs"][0][1] >= data["needs"][-1][1])

broken = json.loads(get("/api/manuscript?name=" + quote("../README.md")))
check("置き場の外は断って理由を返す", "error" in broken, str(broken)[:80])

answer = post("/api/ask", {"name": names[0], "mode": "発想", "size": 6000,
                           "chunk": 1, "lensmode": "auto", "lenses": 3,
                           "engine": ""})
check("問いを組んで返す", "html" in answer, str(answer)[:120])
check("発想モードの規則が入る",
      "本文に無いものを述べてよい" in answer.get("html", ""))

answer = post("/api/ask", {"name": names[0], "lensmode": "named",
                           "lens": "無い観点", "engine": ""})
check("知らない観点は理由を返す", "error" in answer, str(answer)[:80])

# 手元で llama-server が動いていることがあるので、確実に閉じた口を指す
answer = post("/api/ask", {"name": names[0], "engine": "llama",
                           "lensmode": "auto",
                           "llama_url": "http://127.0.0.1:1/v1/chat/completions"})
check("llama-server が居なければ理由を返す",
      "error" in answer and "llama-server" in answer["error"],
      str(answer)[:120])
check("そのとき画面を壊さない", "html" not in answer)

try:
    urllib.request.urlopen(BASE + "/" + quote("無い道"), timeout=10)
    missing = False
except urllib.error.HTTPError as problem:
    missing = problem.code == 404
check("知らない道は404", missing)

print("\n== 貼り付けた本文 ==")
# その場で試したいときに、いちいちファイルへ保存させるのは手間が多い
PASTED = "　雨が降った。傘をさした。\n\n　彼は黙っていた。遠くで鐘が鳴った。"
data = post("/api/manuscript", {"text": PASTED, "size": 6000})
check("貼り付けから姿を読める", data.get("summary", "").startswith("32文字"),
      str(data)[:90])
check("貼り付けと分かる題が付く", data["title"] == "貼り付けた原稿")
check("段落に分かれる", "2段落" in data["summary"])

data = post("/api/ask", {"text": PASTED, "engine": "", "lensmode": "auto",
                         "chunk": 1, "size": 6000})
check("貼り付けから問いを組める", "html" in data, str(data)[:90])
check("貼り付けた本文が問いに入る", "雨が降った" in data["html"])

data = post("/api/manuscript", {"text": "   ", "name": "無い原稿.txt"})
check("貼り付けが空なら置き場を見に行く", "error" in data, str(data)[:70])

print("\n== モデルの一覧 ==")
data = json.loads(get("/api/models"))
check("解かせ方ごとに一覧が返る",
      set(data["models"]) == {"api", "llama", "run"}, str(list(data["models"])))
check("どれも空でない", all(data["models"].values()))
check("APIの一覧は claude で始まる",
      all(m.startswith("claude") for m in data["models"]["api"]))
check("人格も返る", "センチュリオン" in data["persona"])

print("\n== チャット ==")
data = post("/api/chat", {"messages": [], "engine": "llama"})
check("何も書かれていなければ断る", "error" in data, str(data)[:70])

data = post("/api/chat", {
    "messages": [{"role": "user", "content": "こんにちは"}],
    "engine": "llama", "model": "試験",
    "llama_url": "http://127.0.0.1:1/v1/chat/completions"})
check("届かなければ理由を返す", "error" in data, str(data)[:70])

# 何がモデルへ渡るかを、偽の解き手で受け止めて確かめる
sent = {}


class Fake:
    def chat(self, messages, max_tokens=0):
        sent["messages"] = messages
        return "答えました"


real = web.solver_for
web.solver_for = lambda args: Fake()
try:
    out = web.run_chat({"messages": [{"role": "user", "content": "一つ目"},
                                     {"role": "assistant", "content": "はい"},
                                     {"role": "user", "content": "二つ目"}],
                        "system": "あなたはセンチュリオン。", "engine": "llama"})
    check("答えを返す", out == {"reply": "答えました"}, str(out))
    check("人格を先頭に置く", sent["messages"][0]["role"] == "system")
    check("人格の中身が入る",
          sent["messages"][0]["content"] == "あなたはセンチュリオン。")
    check("これまでのやり取りを全部渡す",
          [m["content"] for m in sent["messages"][1:]]
          == ["一つ目", "はい", "二つ目"])

    web.run_chat({"messages": [{"role": "user", "content": "素のまま"}],
                  "system": "  ", "engine": "llama"})
    check("人格が空なら入れない",
          all(m["role"] != "system" for m in sent["messages"]))

    web.run_chat({"messages": [{"role": "user", "content": "ふつう"},
                               {"role": "変な役", "content": "混ぜもの"},
                               {"role": "user", "content": ""}],
                  "engine": "llama"})
    check("知らない役と空の発言は落とす",
          [m["content"] for m in sent["messages"]] == ["ふつう"],
          str(sent["messages"]))
finally:
    web.solver_for = real

print("\n== 画面に出ているもの ==")
page = get("/")
for want, why in [
        ("tab-make", "創作のタブ"),
        ("manuscripts/", "置き場の名前をそのまま出す"),
        ('rows="12"', "貼り付け欄を広く取る"),
        ('rows="7"', "チャットの入力欄を広く取る"),
        ('id="words"', "語の取り出し方を選べる"),
        ('id="examples"', "補足の書き方の例"),
        ("試作です", "創作が試作だと断る"),
        ('id="save"', "添削を保存できる"),
        ("abbr", "実測の意味を出す")]:
    check(why, want in page, want)
check("創作で抑圧が効く経路を明示する", "抑圧が効く" in page)
check("効かない経路も明示する", "抑圧なし" in page)

print("\n== 補足の例と実測の説明 ==")
data = json.loads(get("/api/models"))
check("補足の例を返す", len(data["examples"]) >= 5)
check("例は具体的な迷いになっている",
      all(len(text) > 15 for text in data["examples"]))
check("実測の説明を返す", len(data["help"]) >= 6)
check("測った項目に説明が付いている",
      set(data["help"]) >= {"名前", "会話", "出来事", "感覚", "轍"},
      str(sorted(data["help"])))

print("\n== 画面から呼ぶ関数の名前 ==")
# インラインの onclick は名前を 要素→document→window の順に探す。
# document.write が実在するので、write() と名付けると
# 引数なしの document.write() が呼ばれてページが白紙になる。実際になった
bare = re.findall(r'on(?:click|change)="(?!window\.)([A-Za-z_]\w*)\(', page)
check("画面からの呼び出しは window. を付ける", not bare, str(bare))
check("白紙になる名前を使っていない", 'onclick="write(' not in page)
check("書かせる釦は別名を呼ぶ", "window.runWrite()" in page)

# model3 の選択欄がチャット側を書き換えていた
check("創作の選択欄がある", 'id="model-pick3"' in page)
check("選択欄は三つとも別々",
      len(re.findall(r'id="model-pick\d?"', page)) == 3,
      str(re.findall(r'id="model-pick\d?"', page)))
check("創作の解かせ方が自分の欄を埋めに行く",
      "window.fillModels('engine3','model3')" in page)

print("\n== 厳しさと採点 ==")
check("採点モードが選べる", "<option>採点</option>" in page)
check("厳しさの欄がある", 'id="severity"' in page)
for level in ("育成", "標準", "厳格"):
    check(f"{level}が選べる", f'value="{level}"' in page)
check("既定は標準", 'value="標準" selected' in page)
check("どれがどういう基準かを画面で言う", "評価1が商業水準" in page)
# 効かない欄を出したままにすると、設定したつもりのものが黙って無視される
check("モードで欄を出し分ける", "function modeChanged()" in page)
check("最初にも一度出し分ける", "modeChanged();" in page)
check("厳しさが効くのは査読と採点だけ",
      'SEVERITY_MODES = ["査読", "採点"]' in page)
check("観点が効くのは発想と査読だけ",
      'LENS_MODES = ["発想", "査読"]' in page)

argv = web.to_argv({"name": "あっちゃぐり.txt", "mode": "採点",
                    "severity": "厳格"})
check("厳しさが引数に入る",
      argv[argv.index("--severity") + 1] == "厳格")
check("省いても既定が入る",
      web.to_argv({"name": "あっちゃぐり.txt"})[
          web.to_argv({"name": "あっちゃぐり.txt"}).index("--severity") + 1]
      == "標準")

print("\n== 相手が先に切ったとき ==")
# 画面を読み込み直すと途中の求めが打ち切られる。そこへ書こうとして
# ConnectionAbortedError (WinError 10053) が出るのは異常ではない。
# 素通しにすると、開き直すたびに長い traceback が流れて
# 本当の不具合が埋もれる
noise = io.StringIO()
saved_err = sys.stderr
sys.stderr = noise
try:
    for _ in range(4):
        raw = socket.create_connection(("127.0.0.1", PORT), timeout=5)
        raw.sendall(b"GET /api/models HTTP/1.1\r\nHost: x\r\n\r\n")
        raw.close()          # 返事を待たずに切る
    time.sleep(0.8)
finally:
    sys.stderr = saved_err
check("切られても黙って終わる", "Traceback" not in noise.getvalue(),
      noise.getvalue()[-200:])
check("その後もサーバは応じる", json.loads(get("/api/cache")) is not None)

# 走査は数秒だけ覚えておく。画面を一度開くと
# /api/models と /api/llama が続けて来るので、素直に書くと何度も走る
web.rescan()
began = time.perf_counter()
web.scanned()
cold = time.perf_counter() - began
began = time.perf_counter()
web.scanned()
warm = time.perf_counter() - began
check("二度目の走査は覚えている", warm < cold or warm < 0.001,
      f"{cold * 1000:.1f}ms 対 {warm * 1000:.1f}ms")
check("捨てられる", (web.rescan() or web._scan["結果"]) is None)


print("\n== 画面の script が壊れていないか ==")
# 文字列の途中に本物の改行が入ると、その一行で script 全体が死ぬ。
# 画面は真っ白にならず、ただ何も動かなくなるので気づきにくい。
# 実際に log.join("\n") が改行そのものに化けて、
# window.show is not a function になった


def unclosed(source):
    """改行をまたいだ文字列を探す。(行番号, その行) の並びを返す。

    バッククォートは複数行にまたがってよいので、そこは数えない"""
    problems = []
    quote_char = None
    line_number = 1
    index = 0
    while index < len(source):
        letter = source[index]
        if letter == "\\" and quote_char:
            index += 2
            continue
        if letter == "\n":
            if quote_char in ('"', "'"):
                problems.append((line_number,
                                 source.splitlines()[line_number - 1]))
                quote_char = None
            line_number += 1
        elif quote_char:
            if letter == quote_char:
                quote_char = None
        elif letter in ('"', "'", "`"):
            quote_char = letter
        elif letter == "/" and source[index:index + 2] == "//":
            # 行注釈。中の引用符を数えない
            while index < len(source) and source[index] != "\n":
                index += 1
            continue
        index += 1
    return problems


script = page.split("<script>")[-1].split("</script>")[0]
broken = unclosed(script)
check("文字列が改行をまたいでいない", not broken,
      "; ".join(f"{n}行目 {line.strip()[:50]}" for n, line in broken[:3]))

# 画面から呼ぶ関数が、実際に定義されているか。
# 呼び出しだけあって定義が無いと、押した時に初めて壊れる
called = set(re.findall(r'onclick="window\.(\w+)\(', page))
called |= set(re.findall(r'onchange="window\.(\w+)\(', page))
defined = set(re.findall(r"^(?:async )?function (\w+)", script, re.M))
check("画面から呼ぶ関数がすべて定義されている", called <= defined,
      str(sorted(called - defined)))


print("\n== サーバーの面 ==")
check("サーバーのタブがある", 'id="tab-serve"' in page)
check("サーバーの面がある", 'id="pane-serve"' in page)
check("添削より左に置く",
      page.index('id="tab-serve"') < page.index('id="tab-work"'))
check("面の切り替えに入っている", '"serve", "work"' in page)
check("最初は添削を出す", 'id="tab-work" class="on"' in page)
check("起こす欄は添削から出ている",
      page.index('id="pane-serve"') < page.index("llama-server を起こす")
      < page.index('id="pane-work"'))

print("\n== 覚えているもの ==")
# 一度読んだ原稿を覚える。反復語の探索が重く、
# 形態素解析を使うとさらに重い
web.cache.forget()
body = {"name": names[0], "size": 6000}
first = post("/api/manuscript", body)
check("一度目は覚えていない", first.get("cached") is False, str(first)[:70])
second = post("/api/manuscript", body)
check("二度目は覚えている", second.get("cached") is True)
check("中身は変わらない",
      first["summary"] == second["summary"]
      and first["recurrences"] == second["recurrences"])

# 塊の大きさを変えれば別の解析。覚えていたものを使い回すと
# 違う切り方の答えが返る
other = post("/api/manuscript", {"name": names[0], "size": 800})
check("塊の大きさが変われば読み直す", other.get("cached") is False)
check("切り方も実際に変わる", other["chunks"] != first["chunks"],
      f'{other["chunks"]} 対 {first["chunks"]}')

# 語の取り出し方でも反復語が変わる
web.cache.forget()
post("/api/manuscript", body)
again = post("/api/manuscript", {"name": names[0], "size": 6000,
                                 "words": "形態素"})
check("語の取り出し方が変われば読み直す", again.get("cached") is False)

pasted = post("/api/manuscript", {"text": PASTED, "size": 6000})
check("貼り付けも覚える",
      post("/api/manuscript", {"text": PASTED,
                               "size": 6000}).get("cached") is True,
      str(pasted)[:60])
check("違う貼り付けは読み直す",
      post("/api/manuscript", {"text": PASTED + "　もう一段落。",
                               "size": 6000}).get("cached") is False)

state = json.loads(get("/api/cache"))
check("覚えている数を訊ける", state["原稿"]["覚えている数"] >= 1, str(state))
check("当たりの数も出る", state["解析"]["当たり"] >= 1, str(state))

emptied = post("/api/cache/forget", {})
check("忘れさせられる", emptied["原稿"]["覚えている数"] == 0, str(emptied))
check("忘れた後は読み直す",
      post("/api/manuscript", body).get("cached") is False)

check("画面に忘れる釦がある", "window.forgetAll()" in page)
check("覚えている数を画面に出す", 'id="cache-state"' in page)


print("\n== llama-server を画面から起こす ==")
for want, why in [('id="wake-model"', "手元のモデルから選べる"),
                  ('id="wake-name"', "名前を直に書ける"),
                  ('id="wake-ngl"', "層の数を決められる"),
                  ('id="wake-moe"', "エキスパートの数を決められる"),
                  ('id="wake-ctx"', "文脈の長さを決められる"),
                  ('id="wake-port"', "ポートを決められる"),
                  ('id="wake-log"', "起動ログを見られる"),
                  ("window.wake()", "起こす釦がある"),
                  ("window.sleepLlama()", "止める釦がある")]:
    check(why, want in page, want)
check("MoE かどうかを助言する", "function wakeHint()" in page)

# 起こしてよいものの線引きは server 側にあるが、
# 画面からの入口でも断れないと意味がない
for odd in ["a; whoami", "悪い名前", ""]:
    answer = web.start_llama({"model": odd})
    check(f"{odd!r} は起こさない", "error" in answer, str(answer)[:60])
check("断る理由を伝える",
      "組織/名前" in web.start_llama({"model": "変"})["error"])

# 起こしたポートへ論評を送る。既定へ送ると別のモデルに届く
saved = web.launcher.running
try:
    class Pretending:
        model = "身代わり"

        def alive(self):
            return True

        def url(self):
            return "http://127.0.0.1:9999/v1/chat/completions"

        def status(self):
            return {"running": True, "model": self.model, "log": [],
                    "command": "", "since": None, "exit": None,
                    "port": 9999, "url": self.url(), "binary": ""}

    web.launcher.running = Pretending()
    argv = web.to_argv({"name": "あっちゃぐり.txt", "engine": "llama"})
    check("起こした窓口へ送る",
          argv[argv.index("--llama-url") + 1]
          == "http://127.0.0.1:9999/v1/chat/completions",
          str(argv))
    told = web.to_argv({"name": "あっちゃぐり.txt", "engine": "llama",
                        "llama_url": "http://127.0.0.1:7777/v1/x"})
    check("画面から指定があればそちらを優先する",
          told[told.index("--llama-url") + 1] == "http://127.0.0.1:7777/v1/x")
finally:
    web.launcher.running = saved

check("立っていなければ窓口を足さない",
      "--llama-url" not in web.to_argv({"name": "あっちゃぐり.txt", "engine": "llama"}))


print("\n== 検証 ==")
for want, why in [('id="verify"', "検分の切り替えがある"),
                  ("verify-with", "検分させる相手を選べる"),
                  ("verify-model", "別のモデル名を渡せる"),
                  ("疑う側に立たせ", "何をするのか説明する"),
                  ("自分の答えを通しがち", "別のモデルを勧める理由を書く")]:
    check(why, want in page, want)

argv = web.to_argv({"name": "あっちゃぐり.txt", "verify": True,
                    "verifyWith": "api", "verifyModel": "claude-opus-5"})
check("検分の指定が引数に届く", "--verify" in argv)
check("経路の指定が届く", argv[argv.index("--verify-with") + 1] == "api")
check("モデルの指定が届く",
      argv[argv.index("--verify-model") + 1] == "claude-opus-5")
parsed = critique.build_parser().parse_args(argv)
check("CLI が受け取れる形になっている",
      parsed.verify and parsed.verify_with == "api"
      and parsed.verify_model == "claude-opus-5")
check("切っていれば付かない",
      "--verify" not in web.to_argv({"name": "あっちゃぐり.txt"}))

print("\n== 語の取り出し方 ==")
argv = web.to_argv({"name": "あっちゃぐり.txt", "words": "形態素"})
check("画面の指定が引数に届く", argv[argv.index("--words") + 1] == "形態素")
check("既定は正規表現",
      web.to_argv({"name": "あっちゃぐり.txt"})[
          web.to_argv({"name": "あっちゃぐり.txt"}).index("--words") + 1]
      == "正規表現")

print("\n== 添削の書き出し ==")
data = post("/api/annotated", {"text": "　雨が降った。傘をさした。",
                              "answer": "[0] 「雨が降った。傘をさした。」は良い。",
                              "label": "試験", "engine": ""})
check("題から名前を作る", data["name"].endswith("_添削.txt"), data.get("name"))
check("記録が入る", "モード:" in data["text"])
check("本文が入る", "雨が降った" in data["text"])
check("照合の結果が入る", "引用の照合" in data["text"])

print("\n== 創作 ==")
data = post("/api/write", {"topic": "", "engine": "llama"})
check("お題が無ければ断る", "error" in data, str(data)[:60])

data = post("/api/write", {
    "topic": "朝の匂いについて書いて", "engine": "llama",
    "llama_url": "http://127.0.0.1:1/v1/chat/completions"})
check("届かなければ理由を返す", "error" in data, str(data)[:60])


class FakeWriter:
    def chat(self, messages, max_tokens=0):
        return "書きました。"


real = web.solver_for
web.solver_for = lambda args: FakeWriter()
try:
    data = web.run_write({"topic": "朝の匂いについて書いて", "engine": "llama",
                          "times": 3, "fluid": True})
    check("回数ぶん書かせる", len(data["written"]) == 3)
    check("書き出しを付ける", data["written"][0]["text"].startswith("そうですね、"))
    check("そのとき選んだ姿勢を返す", len(data["written"][0]["stance"]) == 2)
    check("禁止語も返す", len(data["written"][0]["banned"]) == 3)
    check("抑圧が効かないことを断る", "抑圧なし" in data["note"])

    data = web.run_write({"topic": "お題", "engine": "llama", "fluid": False})
    check("流動を切れる", data["written"][0]["stance"] == [])
    check("そのとき固定の人格を使う",
          "詩情と物語だけを愛する" in data["prompt"])

    data = web.run_write({"topic": "お題", "engine": "llama", "times": 99})
    check("回数に上限がある", len(data["written"]) <= 5)
finally:
    web.solver_for = real

print(f"\n{passed}件通過 / {failed}件失敗")
raise SystemExit(1 if failed else 0)

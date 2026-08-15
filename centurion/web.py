"""
ブラウザから使うための小さなサーバ。

    python main.py web

## 作りの方針

**標準ライブラリだけで書く。** 原稿を読ませる側は pip install なしで
動くようにしてあり、画面のために依存を増やすのは筋が違う。

**127.0.0.1 にだけ開く。** manuscripts/ には未発表の原稿が入る。
同じ網の中の別の機械から読めてはいけない。

**鍵はブラウザに渡さない。** ANTHROPIC_API_KEY はサーバ側の
環境変数のままで、画面にも通信にも出さない。

**引数の解釈は CLI と同じものを使う。** 画面からの指定を argv に組んで
critique.build_parser() に通すので、画面と端末で振る舞いがずれない。

## 長く待つことについて

論評は数分かかる。押した瞬間に返らないので、画面には「訊いています」と
出したまま待つ。一人で使う道具なので、それで足りる。
"""

import html
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import critique
from .answer import MARK_BAD, MARK_OK, annotate, attach, find_quotes
from .connect import recurrences
from .manuscript import MANUSCRIPTS, Manuscript
from .prompts import build_fluid
from .review import needs

HOST = "127.0.0.1"
PORT = 8765
SUFFIXES = {".txt", ".md"}

PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>centurion</title>
<style>
 :root { color-scheme: light dark; --line:#8884; --accent:#4a9; --bad:#d55; }
 body { font-family: "Yu Gothic UI", "Hiragino Sans", system-ui, sans-serif;
        margin: 0; line-height: 1.75; }
 header { padding: 10px 20px; border-bottom: 1px solid var(--line);
          display: flex; gap: 18px; align-items: baseline; flex-wrap: wrap; }
 /* 外の字を読みに行かないよう、手元にある筆記体だけを並べる */
 .mark { font-family: "Segoe Script", "Brush Script MT", "Lucida Handwriting",
                      "Apple Chancery", "URW Chancery L", cursive;
         font-size: 30px; font-style: italic; letter-spacing: .5px;
         margin: 0; line-height: 1; }
 .mark .c { font-size: 38px; }
 nav { display: flex; gap: 4px; }
 nav button { border: 1px solid transparent; border-bottom: none;
              border-radius: 7px 7px 0 0; padding: 6px 18px; opacity: .6; }
 nav button.on { border-color: var(--line); opacity: 1; font-weight: 600; }
 main { display: grid; grid-template-columns: 300px 1fr; min-height: 88vh; }
 main.wide { grid-template-columns: 1fr; }
 aside { padding: 16px 20px; border-right: 1px solid var(--line); font-size: 13px; }
 section { padding: 16px 28px; }
 fieldset { border: 1px solid var(--line); border-radius: 6px; margin: 0 0 14px;
            padding: 10px 12px; }
 legend { font-size: 12px; opacity: .75; padding: 0 4px; }
 label { display: block; margin: 5px 0; font-size: 13px; }
 select, input, button, textarea {
   font: inherit; padding: 4px 8px; border-radius: 5px;
   border: 1px solid #8886; background: transparent; color: inherit; }
 textarea { width: 100%; box-sizing: border-box; resize: vertical;
            font-size: 13px; line-height: 1.7; }
 button { cursor: pointer; }
 button.go { padding: 7px 20px; font-weight: 600; }
 button.go:disabled { opacity: .5; cursor: progress; }
 .note { font-size: 12px; opacity: .7; }
 .para { margin: 0 0 4px; }
 .num { opacity: .45; font-size: 11px; margin-right: 6px;
        font-variant-numeric: tabular-nums; }
 .tip { margin: 2px 0 10px 22px; padding: 5px 10px; font-size: 13px;
        border-left: 3px solid var(--accent); background: #4a91;
        border-radius: 0 4px 4px 0; }
 .tip.bad { border-left-color: var(--bad); background: #d551; }
 .head { white-space: pre-wrap; font-size: 12px; opacity: .8;
         border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px;
         margin-bottom: 16px; }
 pre.prompt { white-space: pre-wrap; font-size: 12px; background: #8881;
              padding: 12px; border-radius: 6px; }
 .item { margin: 3px 0; font-size: 12px; }
 .bar { display: inline-block; height: 7px; background: var(--accent);
        border-radius: 3px; vertical-align: middle; margin-right: 5px; }
 .turn { margin: 0 0 14px; padding: 9px 14px; border-radius: 8px;
         white-space: pre-wrap; }
 .turn.me { background: #4a91; margin-left: 15%; }
 .turn.it { background: #8881; margin-right: 15%; }
 .who { font-size: 11px; opacity: .6; display: block; margin-bottom: 3px; }
 .hide { display: none; }
</style></head><body>
<header>
  <p class="mark"><span class="c">c</span>enturion</p>
  <nav>
    <button id="tab-work" class="on" onclick="show('work')">添削</button>
    <button id="tab-talk" onclick="show('talk')">チャット</button>
  </nav>
  <span id="state" class="note"></span>
</header>

<main id="pane-work">
<aside id="side" class="note">原稿を選ぶか、貼り付けて「読む」を押す</aside>
<section>
  <fieldset><legend>原稿</legend>
    <label>置き場から
      <select id="doc"></select>
      <button onclick="load()">読む</button>
    </label>
    <label>または、ここに貼り付ける (貼ってあればこちらを使う)
      <textarea id="pasted" rows="4"
        placeholder="原稿をそのまま貼り付けて「読む」"></textarea></label>
  </fieldset>
  <fieldset><legend>何を訊くか</legend>
    <label>モード
      <select id="mode">
        <option>発想</option><option>査読</option>
        <option>接続</option><option>連想</option>
      </select>
    </label>
    <label>観点
      <select id="lensmode">
        <option value="auto">原稿を測って選ぶ</option>
        <option value="random">くじ引き</option>
        <option value="named">自分で決める</option>
      </select>
      <input id="lens" placeholder="視点,熱量" size="18">
      <input id="lenses" type="number" value="3" min="1" max="8" size="2">個
    </label>
    <label>読ませる範囲
      <select id="chunk"></select>
      <input id="size" type="number" value="6000" step="500" size="5">文字ずつ
    </label>
    <label>作者からの補足
      <input id="note" placeholder="狙いや訊きたいこと" size="42"></label>
  </fieldset>
  <fieldset><legend>誰に解かせるか</legend>
    <label><select id="engine" onchange="fillModels('engine','model')">
      <option value="">問いを出すだけ (貼って使う)</option>
      <option value="llama">llama.cpp (手元・AMDでも動く)</option>
      <option value="api">Claude の API (鍵が要る)</option>
      <option value="run">transformers (NVIDIAが要る)</option>
    </select>
    <select id="model-pick" onchange="$('model').value=this.value"></select>
    <input id="model" placeholder="モデル名を直に書く" size="26"></label>
  </fieldset>
  <button class="go" id="go" onclick="ask()">実行</button>
  <span id="busy" class="note"></span>
  <div id="out"></div>
</section>
</main>

<main id="pane-talk" class="wide hide">
<section>
  <fieldset><legend>誰と話すか</legend>
    <label><select id="engine2" onchange="fillModels('engine2','model2')">
      <option value="llama">llama.cpp (手元・AMDでも動く)</option>
      <option value="api">Claude の API (鍵が要る)</option>
      <option value="run">transformers (NVIDIAが要る)</option>
    </select>
    <select id="model-pick2" onchange="$('model2').value=this.value"></select>
    <input id="model2" placeholder="モデル名を直に書く" size="26"></label>
    <label>人格 (空なら素のまま)
      <textarea id="system" rows="2"
        placeholder="あなたはセンチュリオン。生粋の文系で…"></textarea></label>
    <button onclick="usePersona()">センチュリオンの人格を入れる</button>
    <button onclick="clearTalk()">会話を捨てる</button>
  </fieldset>
  <div id="log"></div>
  <textarea id="say" rows="3" placeholder="ここに書いて Ctrl+Enter で送る"
            onkeydown="if(event.ctrlKey&&event.key==='Enter')talk()"></textarea>
  <p><button class="go" id="go2" onclick="talk()">送る</button>
     <span id="busy2" class="note"></span></p>
</section>
</main>

<script>
const $ = id => document.getElementById(id);
let current = null, models = {}, persona = "", history = [];

function show(which) {
  for (const name of ["work", "talk"]) {
    $("pane-" + name).classList.toggle("hide", name !== which);
    $("tab-" + name).classList.toggle("on", name === which);
  }
}

function escape(text) {
  return text.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function fillModels(from, into) {
  const list = models[$(from).value] || [];
  const pick = $(into === "model" ? "model-pick" : "model-pick2");
  pick.innerHTML = "<option value=''>選ぶ…</option>"
    + list.map(m => `<option>${escape(m)}</option>`).join("");
}

async function boot() {
  const data = await (await fetch("/api/models")).json();
  models = data.models; persona = data.persona;
  fillModels("engine", "model"); fillModels("engine2", "model2");
  const names = await (await fetch("/api/manuscripts")).json();
  $("doc").innerHTML = names.map(n => `<option>${escape(n)}</option>`).join("")
    || "<option value=''>置き場が空です</option>";
  if (names.length) load();
}

function source() {
  const pasted = $("pasted").value.trim();
  return pasted ? {text: pasted} : {name: $("doc").value};
}

async function load() {
  $("state").textContent = "読んでいます…";
  const data = await (await fetch("/api/manuscript", {
    method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({...source(), size: +$("size").value})})).json();
  if (data.error) { $("state").textContent = data.error; return; }
  current = data;
  $("state").textContent = "";
  $("chunk").innerHTML = data.chunks.map((c, i) =>
    `<option value="${i + 1}">${i + 1}/${data.chunks.length} ${escape(c)}</option>`).join("");
  const bars = rows => rows.map(([k, v]) =>
    `<div class="item"><span class="bar" style="width:${Math.round(v * 60)}px"></span>`
    + `${escape(k)} ${Math.round(v * 100)}%</div>`).join("");
  $("side").innerHTML =
    `<b>${escape(data.title)}</b><br>${escape(data.summary)}<br><br>`
    + `<b>実測</b><br>${bars(data.survey)}`
    + `<br><b>観点の必要度</b><br>${bars(data.needs.slice(0, 8))}`
    + (data.recurrences.length
        ? `<br><b>反復</b><br>` + data.recurrences.map(r =>
            `<div class="item">${escape(r)}</div>`).join("")
        : `<br><span class="note">${escape(data.no_recurrence || "")}</span>`);
}

async function ask() {
  if (!current) { alert("先に原稿を読んでください"); return; }
  $("go").disabled = true;
  $("busy").textContent = $("engine").value
    ? "訊いています… (数分かかります)" : "組んでいます…";
  $("out").innerHTML = "";
  try {
    const data = await (await fetch("/api/ask", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({...source(), mode: $("mode").value,
        size: +$("size").value, chunk: +$("chunk").value,
        lensmode: $("lensmode").value, lens: $("lens").value,
        lenses: +$("lenses").value, note: $("note").value,
        engine: $("engine").value, model: $("model").value})})).json();
    $("out").innerHTML = data.error
      ? `<div class="tip bad">${escape(data.error)}</div>` : data.html;
  } catch (problem) {
    $("out").innerHTML = `<div class="tip bad">${escape(String(problem))}</div>`;
  } finally { $("go").disabled = false; $("busy").textContent = ""; }
}

function usePersona() { $("system").value = persona; }
function clearTalk() { history = []; $("log").innerHTML = ""; }

function draw() {
  $("log").innerHTML = history.map(turn =>
    `<div class="turn ${turn.role === "user" ? "me" : "it"}">`
    + `<span class="who">${turn.role === "user" ? "あなた" : "モデル"}</span>`
    + escape(turn.content) + `</div>`).join("");
  $("log").lastElementChild?.scrollIntoView({block: "nearest"});
}

async function talk() {
  const said = $("say").value.trim();
  if (!said) return;
  history.push({role: "user", content: said});
  $("say").value = ""; draw();
  $("go2").disabled = true; $("busy2").textContent = "考えています…";
  try {
    const data = await (await fetch("/api/chat", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({messages: history, system: $("system").value,
        engine: $("engine2").value, model: $("model2").value})})).json();
    history.push({role: "assistant",
                  content: data.error ? "— " + data.error : data.reply});
    draw();
  } finally { $("go2").disabled = false; $("busy2").textContent = ""; }
}
boot();
</script></body></html>
"""


def persona():
    """チャットの人格として使えるセンチュリオンのプロンプト。
    毎回の流動ではなく、会話の間ずっと同じものを使う"""
    return build_fluid()[0]


def listing():
    """manuscripts/ にある原稿。置き場そのものの説明書は原稿ではないので外す"""
    if not MANUSCRIPTS.exists():
        return []
    return sorted(path.name for path in MANUSCRIPTS.iterdir()
                  if path.suffix.lower() in SUFFIXES
                  and not path.name.startswith(".")
                  and path.name.lower() != "readme.md")


def resolve(name):
    """manuscripts/ の中だけを開く。外を指す名前は断る。

    Path(name).name で名前だけ取り出すと ../README.md が README.md に
    化けて、断ったつもりで別のファイルを開いてしまう。
    区切りを含む名前はその場で断る"""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"原稿の名前として使えない: {name}")
    path = (MANUSCRIPTS / name).resolve()
    if path.parent != MANUSCRIPTS.resolve() or not path.exists():
        raise ValueError(f"manuscripts/ に {name} が無い")
    return path


def obtain(body):
    """置き場のファイルか、貼り付けられた本文か。
    貼ってあればそちらを使う — その場で試したいときに、
    いちいちファイルに保存させるのは手間が多い"""
    pasted = (body.get("text") or "").strip()
    if pasted:
        return Manuscript(pasted, title="貼り付けた原稿")
    return Manuscript.load(resolve(body.get("name", "")))


def structure(body, size):
    manuscript = obtain(body)
    chunks = manuscript.chunks(size=size, overlap=1)
    score, measured = needs(manuscript.paragraphs)
    found = recurrences(manuscript)
    return {
        "title": manuscript.title or "原稿",
        "summary": f"{len(manuscript.text)}文字 / "
                   f"{len(manuscript.paragraphs)}段落 / "
                   f"{len(manuscript.chapters)}章",
        "chunks": [f"段落{c.span[0]}〜{c.span[1]}" for c in chunks],
        "survey": sorted(measured.items(), key=lambda kv: -kv[1]),
        "needs": sorted(score.items(), key=lambda kv: -kv[1]),
        "recurrences": [str(item) for item in found[:10]],
        "no_recurrence": critique.too_short(manuscript),
    }


def to_argv(body, path=None):
    """画面からの指定を、CLI と同じ argv に組む。
    解釈を一箇所にまとめておけば、画面と端末で振る舞いがずれない。

    貼り付けた本文には置き場のファイルが無いので、
    そのときは呼び手が仮置きの path を渡す"""
    argv = [str(path or resolve(body.get("name", ""))),
            "--mode", body.get("mode", "発想"),
            "--size", str(body.get("size", 6000)),
            "--chunk", str(body.get("chunk", 1)),
            "--lenses", str(body.get("lenses", 3))]
    if body.get("lensmode") == "random":
        argv.append("--random-lenses")
    elif body.get("lensmode") == "named" and body.get("lens", "").strip():
        argv += ["--lens", body["lens"].strip()]
    if body.get("note", "").strip():
        argv += ["--note", body["note"].strip()]
    if body.get("model", "").strip():
        argv += ["--model", body["model"].strip()]
    engine = body.get("engine", "")
    if engine in ("llama", "api", "run"):
        argv.append(f"--{engine}")
    if body.get("llama_url", "").strip():
        argv += ["--llama-url", body["llama_url"].strip()]
    return argv


def render(manuscript, answer, records, prompt=None):
    """答えを画面用の HTML にする。
    本文は作者のもので、< や & が入りうるので必ず逃がす"""
    parts = ["<div class='head'>"
             + "\n".join(f"{name}: {value}" for name, value in records
                         if value) + "</div>"]
    if prompt is not None:
        parts.append("<p class='note'>これを好きなチャットに貼り、"
                     "答えを保存してから <code>check</code> にかけてください。</p>")
        parts.append(f"<pre class='prompt'>{html.escape(prompt)}</pre>")
        return "".join(parts)

    quotes = find_quotes(answer, manuscript)
    bad = {q.line for q in quotes if not q.ok}
    preamble, notes = attach(answer, manuscript)
    if preamble:
        # 段落に紐づかない指摘にも印を付ける。
        # 存在しない番号を挙げた指摘はここへ落ちるので、
        # ここを素通しにすると一番怪しいものが無印で並ぶ
        parts.append("<h3>段落を指していない指摘</h3>")
        for line in preamble:
            kind = "tip bad" if line in bad else "tip"
            mark = MARK_BAD if line in bad else ""
            parts.append(f"<div class='{kind}'>{mark}"
                         f"{html.escape(line)}</div>")
    parts.append("<h3>本文と指摘</h3>")
    for paragraph in manuscript.paragraphs:
        parts.append(f"<p class='para'><span class='num'>"
                     f"{paragraph.index}</span>"
                     f"{html.escape(paragraph.text)}</p>")
        for line in notes.get(paragraph.index, []):
            kind = "tip bad" if line in bad else "tip"
            mark = MARK_BAD if line in bad else MARK_OK
            parts.append(f"<div class='{kind}'>{mark} "
                         f"{html.escape(line)}</div>")
    return "".join(parts)


def solver_for(args):
    if args.api:
        return critique.Api(args.model or critique.API_MODEL)
    if args.llama:
        return critique.Llama(args.model or "", args.llama_url)
    return critique.Local(args.model or critique.LOCAL_MODEL)


def run_chat(body):
    """普通の会話。原稿も観点も通さず、そのまま渡す"""
    engine = body.get("engine", "llama")
    argv = ["_", f"--{engine}"] if engine in ("llama", "api", "run") else ["_"]
    if body.get("model", "").strip():
        argv += ["--model", body["model"].strip()]
    if body.get("llama_url", "").strip():
        argv += ["--llama-url", body["llama_url"].strip()]
    args = critique.build_parser().parse_args(argv)

    messages = [turn for turn in body.get("messages", [])
                if turn.get("role") in ("user", "assistant")
                and turn.get("content")]
    if not messages:
        raise ValueError("何も書かれていない")
    head = (body.get("system") or "").strip()
    if head:
        messages = [{"role": "system", "content": head}] + messages
    return {"reply": solver_for(args).chat(messages, args.tokens)}


def run_ask(body):
    manuscript = obtain(body)
    args = critique.build_parser().parse_args(
        to_argv(body, path=MANUSCRIPTS / "貼り付け.txt"
                if body.get("text", "").strip() else None))
    if args.words == "形態素":
        critique.use_morphology(True)
    jobs = critique.tasks(manuscript, args)
    head, prompt_body, allowed, label = jobs[0]

    if not (args.api or args.run or args.llama):
        records = critique.annotation_records(args, manuscript, [label])
        return {"html": render(manuscript, "", records,
                               prompt=head + "\n\n---\n\n" + prompt_body)}

    answer = solver_for(args)(head, prompt_body, args.tokens)
    records = critique.annotation_records(args, manuscript, [label])
    if body.get("save"):
        out = MANUSCRIPTS / f"{manuscript.title}_添削.txt"
        out.write_text(annotate(answer, manuscript, records) + "\n",
                       encoding="utf-8")
    return {"html": render(manuscript, answer, records)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *ignored):
        pass                                   # 一件ごとの記録は出さない

    def reply(self, payload, kind="application/json"):
        if kind == "application/json":
            payload = json.dumps(payload, ensure_ascii=False)
        raw = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", f"{kind}; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        route = urlparse(self.path)
        try:
            if route.path == "/":
                return self.reply(PAGE, "text/html")
            if route.path == "/api/manuscripts":
                return self.reply(listing())
            if route.path == "/api/models":
                return self.reply({"models": critique.KNOWN_MODELS,
                                   "persona": persona()})
            if route.path == "/api/manuscript":
                query = parse_qs(route.query)
                return self.reply(structure(
                    {"name": query.get("name", [""])[0]},
                    int(query.get("size", ["6000"])[0])))
        except Exception as problem:
            return self.reply({"error": str(problem)})
        self.send_error(404)

    def do_POST(self):
        route = urlparse(self.path).path
        works = {"/api/ask": run_ask, "/api/chat": run_chat,
                 "/api/manuscript": lambda body: structure(
                     body, int(body.get("size", 6000)))}
        if route not in works:
            return self.send_error(404)
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            return self.reply(works[route](body))
        except SystemExit as stop:
            return self.reply({"error": str(stop)})
        except Exception as problem:
            return self.reply({"error": f"{type(problem).__name__}: {problem}"})


def serve(port=PORT, open_page=True):
    MANUSCRIPTS.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, port), Handler)
    address = f"http://{HOST}:{port}/"
    print(f"センチュリオンを {address} で開いています", file=sys.stderr)
    print("止めるには Ctrl+C", file=sys.stderr)
    if open_page:
        threading.Timer(0.5, webbrowser.open, [address]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n止めました", file=sys.stderr)
    finally:
        server.server_close()
    return 0


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="centurion.web",
        description="ブラウザから使う。127.0.0.1 にだけ開く")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--no-open", action="store_true",
                        help="ブラウザを自動で開かない")
    args = parser.parse_args(argv)
    return serve(args.port, not args.no_open)


if __name__ == "__main__":
    raise SystemExit(main())

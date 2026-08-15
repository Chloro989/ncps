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
from .review import LENSES, describe, needs

HOST = "127.0.0.1"
PORT = 8765
SUFFIXES = {".txt", ".md"}

PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>センチュリオン</title>
<style>
 :root { color-scheme: light dark; }
 body { font-family: "Yu Gothic UI", "Hiragino Sans", system-ui, sans-serif;
        margin: 0; line-height: 1.75; }
 header { padding: 12px 20px; border-bottom: 1px solid #8884;
          display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
 h1 { font-size: 15px; margin: 0 12px 0 0; font-weight: 600; }
 main { display: grid; grid-template-columns: 300px 1fr; min-height: 90vh; }
 aside { padding: 16px 20px; border-right: 1px solid #8884; font-size: 13px; }
 section { padding: 16px 28px; }
 fieldset { border: 1px solid #8884; border-radius: 6px; margin: 0 0 14px;
            padding: 10px 12px; }
 legend { font-size: 12px; opacity: .75; padding: 0 4px; }
 label { display: block; margin: 4px 0; font-size: 13px; }
 select, input, button { font: inherit; padding: 4px 8px; border-radius: 5px;
                         border: 1px solid #8886; background: transparent;
                         color: inherit; }
 button { cursor: pointer; }
 button.go { padding: 7px 18px; font-weight: 600; }
 button.go:disabled { opacity: .5; cursor: progress; }
 .note { font-size: 12px; opacity: .7; }
 .para { margin: 0 0 4px; }
 .num { opacity: .45; font-size: 11px; margin-right: 6px;
        font-variant-numeric: tabular-nums; }
 .tip { margin: 2px 0 10px 22px; padding: 5px 10px; font-size: 13px;
        border-left: 3px solid #4a9; background: #4a91; border-radius: 0 4px 4px 0; }
 .tip.bad { border-left-color: #d55; background: #d551; }
 .head { white-space: pre-wrap; font-size: 12px; opacity: .8;
         border: 1px solid #8884; border-radius: 6px; padding: 8px 12px;
         margin-bottom: 16px; }
 pre.prompt { white-space: pre-wrap; font-size: 12px; background: #8881;
              padding: 12px; border-radius: 6px; }
 .item { margin: 3px 0; font-size: 12px; }
 .bar { display: inline-block; height: 7px; background: #4a9; border-radius: 3px;
        vertical-align: middle; margin-right: 5px; }
</style></head><body>
<header>
  <h1>センチュリオン</h1>
  <select id="doc"></select>
  <button onclick="load()">読む</button>
  <span id="state" class="note"></span>
</header>
<main>
<aside id="side" class="note">原稿を選んで「読む」を押す</aside>
<section>
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
    <label><select id="engine">
      <option value="">問いを出すだけ (貼って使う)</option>
      <option value="llama">llama.cpp (手元・AMDでも動く)</option>
      <option value="api">Claude の API (鍵が要る)</option>
      <option value="run">transformers (NVIDIAが要る)</option>
    </select>
    <input id="model" placeholder="モデル名 (任意)" size="26"></label>
  </fieldset>
  <button class="go" id="go" onclick="ask()">実行</button>
  <span id="busy" class="note"></span>
  <div id="out"></div>
</section>
</main>
<script>
const $ = id => document.getElementById(id);
let current = null;

async function boot() {
  const names = await (await fetch("/api/manuscripts")).json();
  $("doc").innerHTML = names.map(n => `<option>${n}</option>`).join("")
    || "<option>manuscripts/ が空です</option>";
  if (names.length) load();
}

async function load() {
  const name = $("doc").value;
  $("state").textContent = "読んでいます…";
  const url = `/api/manuscript?name=${encodeURIComponent(name)}`
            + `&size=${$("size").value}`;
  const data = await (await fetch(url)).json();
  if (data.error) { $("state").textContent = data.error; return; }
  current = data;
  $("state").textContent = "";
  $("chunk").innerHTML = data.chunks
    .map((c, i) => `<option value="${i + 1}">${i + 1}/${data.chunks.length} ${c}</option>`)
    .join("");
  $("side").innerHTML =
    `<b>${data.title}</b><br>${data.summary}<br><br>`
    + `<b>実測</b><br>`
    + data.survey.map(([k, v]) =>
        `<div class="item"><span class="bar" style="width:${Math.round(v * 60)}px"></span>${k} ${Math.round(v * 100)}%</div>`).join("")
    + `<br><b>観点の必要度</b><br>`
    + data.needs.slice(0, 8).map(([k, v]) =>
        `<div class="item"><span class="bar" style="width:${Math.round(v * 60)}px"></span>${k} ${Math.round(v * 100)}%</div>`).join("")
    + (data.recurrences.length
        ? `<br><b>反復</b><br>` + data.recurrences.map(r => `<div class="item">${r}</div>`).join("")
        : `<br><span class="note">${data.no_recurrence || ""}</span>`);
}

async function ask() {
  if (!current) { alert("先に原稿を読んでください"); return; }
  $("go").disabled = true;
  $("busy").textContent = $("engine").value ? "訊いています… (数分かかります)" : "組んでいます…";
  $("out").innerHTML = "";
  const body = {
    name: $("doc").value, mode: $("mode").value, size: +$("size").value,
    chunk: +$("chunk").value, lensmode: $("lensmode").value,
    lens: $("lens").value, lenses: +$("lenses").value,
    note: $("note").value, engine: $("engine").value, model: $("model").value,
  };
  try {
    const data = await (await fetch("/api/ask", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(body)})).json();
    $("out").innerHTML = data.error
      ? `<div class="tip bad">${data.error}</div>` : data.html;
  } catch (problem) {
    $("out").innerHTML = `<div class="tip bad">${problem}</div>`;
  } finally {
    $("go").disabled = false; $("busy").textContent = "";
  }
}
boot();
</script></body></html>
"""


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


def structure(name, size):
    manuscript = Manuscript.load(resolve(name))
    chunks = manuscript.chunks(size=size, overlap=1)
    score, measured = needs(manuscript.paragraphs)
    found = recurrences(manuscript)
    return {
        "title": manuscript.title or name,
        "summary": f"{len(manuscript.text)}文字 / "
                   f"{len(manuscript.paragraphs)}段落 / "
                   f"{len(manuscript.chapters)}章",
        "chunks": [f"段落{c.span[0]}〜{c.span[1]}" for c in chunks],
        "survey": sorted(measured.items(), key=lambda kv: -kv[1]),
        "needs": sorted(score.items(), key=lambda kv: -kv[1]),
        "recurrences": [str(item) for item in found[:10]],
        "no_recurrence": critique.too_short(manuscript),
    }


def to_argv(body):
    """画面からの指定を、CLI と同じ argv に組む。
    解釈を一箇所にまとめておけば、画面と端末で振る舞いがずれない"""
    argv = [str(resolve(body["name"])),
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


def run_ask(body):
    args = critique.build_parser().parse_args(to_argv(body))
    if args.words == "形態素":
        critique.use_morphology(True)
    manuscript = Manuscript.load(args.path)
    jobs = critique.tasks(manuscript, args)
    head, prompt_body, allowed, label = jobs[0]

    if not (args.api or args.run or args.llama):
        records = critique.annotation_records(args, manuscript, [label])
        return {"html": render(manuscript, "", records,
                               prompt=head + "\n\n---\n\n" + prompt_body)}

    if args.api:
        solve = critique.Api(args.model or critique.API_MODEL)
    elif args.llama:
        solve = critique.Llama(args.model or "", args.llama_url)
    else:
        solve = critique.Local(args.model or critique.LOCAL_MODEL)

    answer = solve(head, prompt_body, args.tokens)
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
            if route.path == "/api/manuscript":
                query = parse_qs(route.query)
                return self.reply(structure(
                    query.get("name", [""])[0],
                    int(query.get("size", ["6000"])[0])))
        except Exception as problem:
            return self.reply({"error": str(problem)})
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            return self.send_error(404)
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            return self.reply(run_ask(body))
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

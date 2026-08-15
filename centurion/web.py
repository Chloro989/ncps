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

from . import critique, review
from .answer import MARK_BAD, MARK_OK, annotate, attach, find_quotes
from .connect import recurrences
from .manuscript import MANUSCRIPTS, Manuscript
from .prompts import FIXED_PROMPT, PREFILL, build_fluid
from .review import needs

HOST = "127.0.0.1"
PORT = 8765
SUFFIXES = {".txt", ".md"}

# 「作者からの補足」に何を書けばいいのか分かりにくいので、例を出す。
# 抽象的な願いより、狙いと迷いを具体的に書いたほうが答えが変わる
NOTE_EXAMPLES = [
    "語り手の距離感を試しています。近すぎないか見てください。",
    "冒頭の三段落だけで読者を引き込みたいのですが、届いていますか。",
    "この人物を好きになってほしいのですが、嫌われていませんか。",
    "説明を削りました。削りすぎて分からなくなっていませんか。",
    "同じ場面を三度書き直しています。何が足りないのか自分で分かりません。",
    "終わり方を決めかねています。ここからの行き先を挙げてください。",
    "文体を変えました。前より読みにくくなっていませんか。",
]

# 左の欄に出す数字が何なのかを、画面の中で説明する
SURVEY_HELP = {
    "名前": "名前のある人や場所を含む段落の割合",
    "会話": "会話文の段落の割合",
    "出来事": "過去の出来事を2文以上語る段落の割合",
    "感覚": "使われている感覚の種類 (音・におい・手触り・温度・光 の5つ中)",
    "一人称": "私・僕・俺を含む段落の割合",
    "偏り": "段落の長さのばらつき",
    "混在": "ですます体とである体の混ざり具合",
    "轍": "常套語 (宇宙・神秘・永遠…) の濃さ",
}

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
 .turn.me { background: #4a91; margin-left: 12%; }
 .turn.it { background: #8881; margin-right: 12%; }
 .who { font-size: 11px; opacity: .6; display: block; margin-bottom: 3px; }
 .hide { display: none; }
 details { margin: 6px 0; font-size: 13px; }
 summary { cursor: pointer; opacity: .75; }
 details button { display: block; margin: 4px 0; text-align: left;
                  width: 100%; font-size: 12px; }
 .warn { border-left: 3px solid #c94; background: #c941; padding: 7px 12px;
         border-radius: 0 4px 4px 0; font-size: 13px; margin: 0 0 12px; }
 .made { border: 1px solid var(--line); border-radius: 7px; padding: 10px 14px;
         margin: 0 0 12px; white-space: pre-wrap; }
 abbr { text-decoration: none; border-bottom: 1px dotted currentColor; }
</style></head><body>
<header>
  <p class="mark"><span class="c">c</span>enturion</p>
  <nav>
    <button id="tab-work" class="on" onclick="window.show('work')">添削</button>
    <button id="tab-make" onclick="window.show('make')">創作</button>
    <button id="tab-talk" onclick="window.show('talk')">チャット</button>
  </nav>
  <span id="state" class="note"></span>
</header>

<main id="pane-work">
<aside id="side" class="note">原稿を選ぶか、貼り付けて「読む」を押す</aside>
<section>
  <fieldset><legend>原稿</legend>
    <label><code>manuscripts/</code> から
      <select id="doc"></select>
      <button onclick="window.load()">読む</button>
    </label>
    <label>または、ここに貼り付ける (貼ってあればこちらを使う)
      <textarea id="pasted" rows="12"
        placeholder="原稿をそのまま貼り付けて「読む」"></textarea></label>
  </fieldset>
  <fieldset><legend>何を訊くか</legend>
    <label>モード
      <select id="mode" onchange="window.modeChanged()">
        <option>発想</option><option>査読</option><option>採点</option>
        <option>接続</option><option>連想</option>
      </select>
    </label>
    <label id="severity-row">厳しさ
      <select id="severity">
        <option value="育成">育成 — 良い点を先に。評価3が「健闘」</option>
        <option value="標準" selected>標準 — 良い点と問題点を同じ精度で</option>
        <option value="厳格">厳格 — 卓越性だけを問う。評価1が商業水準</option>
      </select>
      <span class="note">査読と採点にだけ効く</span>
    </label>
    <label id="lens-row">観点
      <select id="lensmode">
        <option value="auto">原稿を測って選ぶ</option>
        <option value="random">くじ引き</option>
        <option value="named">自分で決める</option>
      </select>
      <input id="lens" placeholder="視点,熱量" size="18">
      <input id="lenses" type="number" value="3" min="1" max="8" size="2"
             onchange="window.lensWarn()">個
      <span id="lenswarn" class="note"></span>
    </label>
    <label>読ませる範囲
      <select id="chunk"></select>
      <input id="size" type="number" value="6000" step="500" size="5">文字ずつ
    </label>
    <label>反復を探すときの語の取り出し
      <select id="words">
        <option value="正規表現">正規表現 (依存なし)</option>
        <option value="形態素">形態素解析 (fugashi が要る)</option>
      </select></label>
    <label>作者からの補足
      <textarea id="note" rows="3"
        placeholder="狙いや訊きたいこと。ここが具体的だと答えが変わる"></textarea></label>
    <details><summary>書き方の例</summary><div id="examples"></div></details>
  </fieldset>
  <fieldset><legend>誰に解かせるか</legend>
    <label><select id="engine" onchange="window.fillModels('engine','model')">
      <option value="">問いを出すだけ (貼って使う)</option>
      <option value="llama">llama.cpp (手元・AMDでも動く)</option>
      <option value="api">Claude の API (鍵が要る)</option>
      <option value="run">transformers (NVIDIAが要る)</option>
    </select>
    <select id="model-pick" onchange="$('model').value=this.value"></select>
    <input id="model" placeholder="モデル名を直に書く" size="26"></label>
  </fieldset>
  <fieldset><legend>検証 (出てきた指摘をもう一度検分させる)</legend>
    <label><input type="checkbox" id="verify"> 検分にかける
      <span class="note">疑う側に立たせ、通った指摘だけを残す。
      時間と費用は倍になる</span></label>
    <label>検分させる相手
      <select id="verify-with">
        <option value="">同じ経路・同じモデル</option>
        <option value="llama">llama.cpp</option>
        <option value="api">Claude の API</option>
        <option value="run">transformers</option>
      </select>
      <input id="verify-model" placeholder="別のモデル名 (任意)" size="24">
    </label>
    <p class="note">別のモデルに検分させたほうが効く。
      同じモデルは自分の答えを通しがちになる。</p>
  </fieldset>
  <button class="go" id="go" onclick="window.ask()">実行</button>
  <button id="save" class="hide" onclick="window.saveAnnotated()">添削を保存</button>
  <span id="busy" class="note"></span>
  <div id="out"></div>
</section>
</main>

<main id="pane-make" class="wide hide">
<section>
  <p class="warn"><b>試作です。</b>
    目的の後半「発想の飛躍」は達成できていません。出てくるのは
    読みやすく常套からわずかに外れた文章であって、飛躍ではありません。
    抑圧 (type5設定) は盲検36件で12戦12勝と確かめてありますが、
    <b>ロジットに手を入れる仕組みなので transformers でしか効きません</b>。
    llama.cpp と API では流動プロンプトだけになります。</p>
  <fieldset><legend>何を書かせるか</legend>
    <label>お題
      <input id="topic" placeholder="古い本を開いたときの手触りを書いて"
             size="52"></label>
    <label><input type="number" id="times" value="1" min="1" max="5" size="2">
      回書かせる</label>
  </fieldset>
  <fieldset><legend>どう書かせるか</legend>
    <label><select id="engine3" onchange="window.fillModels('engine3','model3')">
      <option value="llama">llama.cpp (抑圧なし)</option>
      <option value="api">Claude の API (抑圧なし)</option>
      <option value="run">transformers (抑圧が効く)</option>
    </select>
    <select id="model-pick3" onchange="$('model3').value=this.value"></select>
    <input id="model3" placeholder="モデル名を直に書く" size="26"></label>
    <label><input type="checkbox" id="fluid" checked>
      流動プロンプト (生成ごとに姿勢2つと禁止語3つを選び直す)</label>
    <label><input type="checkbox" id="suppress" checked>
      分岐点での抑圧 (transformers のときだけ効く)</label>
  </fieldset>
  <button class="go" id="go3" onclick="window.runWrite()">書かせる</button>
  <span id="busy3" class="note"></span>
  <div id="made"></div>
</section>
</main>

<main id="pane-talk" class="wide hide">
<section>
  <fieldset><legend>誰と話すか</legend>
    <label><select id="engine2" onchange="window.fillModels('engine2','model2')">
      <option value="llama">llama.cpp (手元・AMDでも動く)</option>
      <option value="api">Claude の API (鍵が要る)</option>
      <option value="run">transformers (NVIDIAが要る)</option>
    </select>
    <select id="model-pick2" onchange="$('model2').value=this.value"></select>
    <input id="model2" placeholder="モデル名を直に書く" size="26"></label>
    <label>人格 (空なら素のまま)
      <textarea id="system" rows="4"
        placeholder="あなたはセンチュリオン。生粋の文系で…"></textarea></label>
    <button onclick="window.usePersona()">センチュリオンの人格を入れる</button>
    <button onclick="window.clearTalk()">会話を捨てる</button>
  </fieldset>
  <div id="log"></div>
  <textarea id="say" rows="7" placeholder="ここに書いて Ctrl+Enter で送る"
            onkeydown="if(event.ctrlKey&&event.key==='Enter')window.talk()"></textarea>
  <p><button class="go" id="go2" onclick="window.talk()">送る</button>
     <span id="busy2" class="note"></span></p>
</section>
</main>

<script>
const $ = id => document.getElementById(id);
let current = null, models = {}, persona = "", history = [];
let helps = {}, lastAnswer = null;

function show(which) {
  for (const name of ["work", "make", "talk"]) {
    $("pane-" + name).classList.toggle("hide", name !== which);
    $("tab-" + name).classList.toggle("on", name === which);
  }
}

function escape(text) {
  return text.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function fillModels(from, into) {
  // into は model / model2 / model3。選択欄は model-pick / -pick2 / -pick3。
  // ここを三項で書いていたとき、model3 がチャット側の欄を書き換えていた
  const list = models[$(from).value] || [];
  const pick = $("model-pick" + into.slice("model".length));
  if (!pick) return;
  pick.innerHTML = "<option value=''>選ぶ…</option>"
    + list.map(m => `<option>${escape(m)}</option>`).join("");
}

async function boot() {
  const data = await (await fetch("/api/models")).json();
  models = data.models; persona = data.persona; helps = data.help || {};
  fillModels("engine", "model"); fillModels("engine2", "model2");
  fillModels("engine3", "model3");
  modeChanged();
  $("examples").innerHTML = (data.examples || []).map(text =>
    `<button onclick="$('note').value=${JSON.stringify(text)}">`
    + escape(text) + `</button>`).join("");
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
  const bars = (rows, tips) => rows.map(([k, v]) =>
    `<div class="item"><span class="bar" style="width:${Math.round(v * 60)}px"></span>`
    + (tips && helps[k] ? `<abbr title="${escape(helps[k])}">${escape(k)}</abbr>`
                        : escape(k))
    + ` ${Math.round(v * 100)}%</div>`).join("");
  $("side").innerHTML =
    `<b>${escape(data.title)}</b><br>${escape(data.summary)}<br><br>`
    + `<b>実測</b><br><span class="note">この原稿を測った値。`
    + `名前の上にカーソルを置くと意味が出ます</span>`
    + bars(data.survey, true)
    + `<br><b>観点の必要度</b><br><span class="note">実測から導いた`
    + `「この原稿に効きそうな観点」。高いものから選ばれます</span>`
    + bars(data.needs.slice(0, 8))
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
  $("out").innerHTML = ""; $("save").classList.add("hide"); lastAnswer = null;
  try {
    const data = await (await fetch("/api/ask", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify(askBody())})).json();
    $("out").innerHTML = data.error
      ? `<div class="tip bad">${escape(data.error)}</div>` : data.html;
    if (data.answer) {
      lastAnswer = {answer: data.answer, label: data.label};
      $("save").classList.remove("hide");
    }
  } catch (problem) {
    $("out").innerHTML = `<div class="tip bad">${escape(String(problem))}</div>`;
  } finally { $("go").disabled = false; $("busy").textContent = ""; }
}

function askBody() {
  return {...source(), mode: $("mode").value, size: +$("size").value,
    chunk: +$("chunk").value, lensmode: $("lensmode").value,
    lens: $("lens").value, lenses: +$("lenses").value, note: $("note").value,
    words: $("words").value, engine: $("engine").value,
    model: $("model").value, verify: $("verify").checked,
    verifyWith: $("verify-with").value, verifyModel: $("verify-model").value};
}

async function saveAnnotated() {
  if (!lastAnswer) return;
  const data = await (await fetch("/api/annotated", {
    method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({...askBody(), ...lastAnswer})})).json();
  if (data.error) { alert(data.error); return; }
  const link = document.createElement("a");
  link.href = URL.createObjectURL(
    new Blob([data.text], {type: "text/plain;charset=utf-8"}));
  link.download = data.name; link.click();
  URL.revokeObjectURL(link.href);
}

// 名前を write にしてはいけない。インラインの onclick は名前を
// 要素→document→window の順に探すので、document.write が先に見つかり、
// 引数なしで呼ばれてページが白紙になる。
// 同じ罠を避けるため、画面から呼ぶものはすべて window. を付けて呼んでいる
async function runWrite() {
  const topic = $("topic").value.trim();
  if (!topic) { alert("お題を書いてください"); return; }
  $("go3").disabled = true; $("busy3").textContent = "書いています…";
  $("made").innerHTML = "";
  try {
    const data = await (await fetch("/api/write", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({topic, times: +$("times").value,
        engine: $("engine3").value, model: $("model3").value,
        fluid: $("fluid").checked, suppress: $("suppress").checked})})).json();
    if (data.error) {
      $("made").innerHTML = `<div class="tip bad">${escape(data.error)}</div>`;
      return;
    }
    $("made").innerHTML = `<p class="note">${escape(data.note)}</p>`
      + data.written.map(one =>
          `<div class="made">${escape(one.text)}</div>`
          + `<p class="note">姿勢: ${escape(one.stance.join(" / ") || "なし")}`
          + ` ／ 禁止語: ${escape(one.banned.join("・") || "なし")}`
          + (one.diverted ? ` ／ 抑圧 ${one.diverted}箇所` : "") + `</p>`).join("");
  } catch (problem) {
    $("made").innerHTML = `<div class="tip bad">${escape(String(problem))}</div>`;
  } finally { $("go3").disabled = false; $("busy3").textContent = ""; }
}

function lensWarn() {
  const many = +$("lenses").value > 3;
  $("lenswarn").textContent = many
    ? "3個までを勧めます。増やすと一つひとつが薄まることを実測しています"
    : "";
  $("lenswarn").style.color = many ? "#c94" : "";
}

// モードによって効く欄が変わる。効かない欄を出したままにすると、
// 設定したつもりのものが無視されて理由が分からなくなる
const LENS_MODES = ["発想", "査読"];
const SEVERITY_MODES = ["査読", "採点"];

function modeChanged() {
  const mode = $("mode").value;
  $("lens-row").style.display = LENS_MODES.includes(mode) ? "" : "none";
  $("severity-row").style.display =
    SEVERITY_MODES.includes(mode) ? "" : "none";
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
            "--words", body.get("words", "正規表現"),
            "--mode", body.get("mode", "発想"),
            "--severity", body.get("severity", review.DEFAULT_SEVERITY),
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
    if body.get("verify"):
        argv.append("--verify")
        if body.get("verifyWith") in ("api", "llama", "run"):
            argv += ["--verify-with", body["verifyWith"]]
        if body.get("verifyModel", "").strip():
            argv += ["--verify-model", body["verifyModel"].strip()]
    return argv


def render(manuscript, answer, records, prompt=None, verified=""):
    """答えを画面用の HTML にする。
    本文は作者のもので、< や & が入りうるので必ず逃がす"""
    parts = ["<div class='head'>"
             + "\n".join(f"{name}: {value}" for name, value in records
                         if value) + "</div>"]
    if verified:
        parts.append(f"<div class='head'>{html.escape(verified)}</div>")
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


def run_write(body):
    """小説を書かせる。試作。

    抑圧(type5設定)はロジットに手を入れる仕組みなので、
    transformers でしか使えない。llama.cpp と API では
    流動プロンプトだけになる — そのことは画面にも出す"""
    topic = (body.get("topic") or "").strip()
    if not topic:
        raise ValueError("お題を書いてください")
    engine = body.get("engine", "llama")
    times = max(1, min(int(body.get("times", 1)), 5))
    fluid = body.get("fluid", True)
    suppress = body.get("suppress", True)

    prompt, stance, banned = (build_fluid() if fluid
                              else (FIXED_PROMPT, [], []))
    written = []

    if engine == "run":
        from .generate import MODEL_NAME, Centurion
        # 空の欄をそのまま渡すと model_name=None で読み込みに行って落ちる
        writer = Centurion(model_name=(body.get("model") or "").strip()
                           or MODEL_NAME,
                           suppress=suppress, fluid=fluid)
        for _ in range(times):
            reply = writer.say(topic, remember=False)
            written.append({"text": reply.text,
                            "stance": reply.stance, "banned": reply.banned,
                            "diverted": len(reply.diverted)})
        note = ("抑圧あり (type5設定)" if suppress else "抑圧なし")
    else:
        argv = ["_", f"--{engine}"]
        if body.get("model", "").strip():
            argv += ["--model", body["model"].strip()]
        if body.get("llama_url", "").strip():
            argv += ["--llama-url", body["llama_url"].strip()]
        args = critique.build_parser().parse_args(argv)
        solve = solver_for(args)
        for _ in range(times):
            written.append({
                "text": PREFILL + solve.chat(
                    [{"role": "system", "content": prompt},
                     {"role": "user", "content": topic}], 400).strip(),
                "stance": stance, "banned": banned, "diverted": 0})
        note = "抑圧なし (この経路ではロジットに手を入れられない)"

    return {"written": written, "note": note,
            "prompt": prompt, "stance": stance, "banned": banned}


def annotated_text(body):
    """添削ファイルの中身をそのまま返す。ブラウザから保存させる"""
    manuscript = obtain(body)
    args = critique.build_parser().parse_args(
        to_argv(body, path=MANUSCRIPTS / "貼り付け.txt"
                if body.get("text", "").strip() else None))
    records = critique.annotation_records(
        args, manuscript, [body.get("label", "")])
    title = manuscript.title or "原稿"
    return {"name": f"{title}_添削.txt",
            "text": annotate(body.get("answer", ""), manuscript, records)}


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
    note = ""
    if args.verify:
        answer, note = critique.run_verify(args, manuscript, answer,
                                           prompt_body)
    records = critique.annotation_records(
        args, manuscript, [label], outcome=critique.verify_outcome(note))
    return {"html": render(manuscript, answer, records, verified=note),
            "answer": answer, "label": label}


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
                                   "persona": persona(),
                                   "examples": NOTE_EXAMPLES,
                                   "help": SURVEY_HELP})
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
                 "/api/write": run_write, "/api/annotated": annotated_text,
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

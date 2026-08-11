"""
センチュリオンの本体。

積んであるのは、盲検で確かめられた2つだけ。

  分岐点での抑圧 (type5設定)  盲検36件で12戦12勝。
      素のモデルの読める率58%を100%にした。中国語もラテン文字も出さない
  流動プロンプト              一騎打ちで20勝12敗 (p=0.108)。
      有意には届いていないが、固定プロンプトが両勝ちしたお題は1つもなかった

試して効かなかったものは積んでいない —
未来エントロピーによる重み付け、表現への介入(層24)、
液体回路による姿勢の漂い。詳細は results/REPORT.md。
"""

import random
import re

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessor, LogitsProcessorList)

from .prompts import FIXED_PROMPT, PREFILL, build_fluid

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# 生成の設定。温度を上げても連想の経路は変わらず文法だけ壊れるので、
# 温度は1.0のまま min_p で裾を切る
MAX_TOKENS = 200
TEMPERATURE = 1.0
MIN_P = 0.05
TOP_P = 1.0

# 抑圧 (type5設定)。エントロピーが高い箇所だけを分岐点とみなし、
# 上位2個を 2.0 押し下げる。
# 押し下げ幅に round() を使わないこと — 量子化すると選択圧が伝わらない
ENTROPY_GATE = 3.5
SUPPRESS_TOP_K = 2
SUPPRESS_STRENGTH = 2.0

SENTENCE_END = "。！？」』"


class BranchDiverter(LogitsProcessor):
    """分岐点でのみ、上位候補を押し下げて別の道へ逸らす。

    エントロピーは全候補について計算する。この処理は min_p より前に
    走るので、この時点では裾が切られていない(実測で確認済み)"""

    def __init__(self, tokenizer, gate=ENTROPY_GATE, top_k=SUPPRESS_TOP_K,
                 strength=SUPPRESS_STRENGTH):
        self.tokenizer = tokenizer
        self.gate = gate
        self.top_k = top_k
        self.strength = strength
        self.diverted = []          # 抑圧した語の記録(観察用)

    def reset(self):
        self.diverted = []

    def __call__(self, input_ids, scores):
        probs = torch.softmax(scores.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        gated = entropy >= self.gate
        if not bool(gated.any()):
            return scores

        top = scores.topk(self.top_k, dim=-1).indices
        for row in torch.nonzero(gated, as_tuple=False).flatten().tolist():
            self.diverted.append(
                "".join(self.tokenizer.decode([i]) for i in top[row]))
            scores[row, top[row]] -= self.strength
        return scores


def trim(text):
    """最後の文末で切る。
    トークン上限で途中で切れた文が残ると、読み物として明確に劣化する。
    文末が一つも無ければ何もしない(短い応答を丸ごと捨てないため)"""
    cut = max((text.rfind(mark) for mark in SENTENCE_END), default=-1)
    return text[:cut + 1] if cut >= 0 else text


class Reply:
    """1回の応答。本文と、そのとき使った指示"""

    def __init__(self, text, stance, banned, diverted):
        self.text = text
        self.stance = stance        # そのとき選ばれた姿勢の句
        self.banned = banned        # そのとき禁じた語
        self.diverted = diverted    # 抑圧した候補(観察用)

    def __str__(self):
        return self.text


class Centurion:
    """文章を保ったまま、常套から少し外れた語りを返す。

    使い方:
        centurion = Centurion()
        print(centurion.say("青色にまつわる話を聞かせて"))

    会話にすると前のやり取りを覚える:
        for reply in centurion.converse(["朝の匂いについて書いて",
                                         "沈黙について書いて"]):
            print(reply)
    """

    def __init__(self, model_name=MODEL_NAME, suppress=True, fluid=True,
                 prefill=PREFILL, max_tokens=MAX_TOKENS, trim_tail=True,
                 device=None, seed=None):
        self.model_name = model_name
        self.suppress = suppress
        self.fluid = fluid
        self.prefill = prefill
        self.max_tokens = max_tokens
        self.trim_tail = trim_tail
        self.seed = seed
        self.history = []           # (お題, 応答) の並び
        # 姿勢と禁止語の抽選は Python の乱数、本文の抽選は torch の乱数。
        # 種を渡されたら両方を固定しないと同じ文章にならない
        self.rng = random.Random(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        self.diverter = BranchDiverter(self.tokenizer)

        if seed is not None:
            torch.manual_seed(seed)

    def forget(self):
        """会話の記憶を捨てる。種を渡してあれば抽選も最初に戻す"""
        self.history = []
        if self.seed is not None:
            self.rng.seed(self.seed)
            torch.manual_seed(self.seed)

    def _system_prompt(self):
        if not self.fluid:
            return FIXED_PROMPT, [], []
        return build_fluid(self.rng)

    def _prefix(self, system_prompt, topic):
        messages = [{"role": "system", "content": system_prompt}]
        for past_topic, past_reply in self.history:
            messages.append({"role": "user", "content": past_topic})
            messages.append({"role": "assistant", "content": past_reply})
        messages.append({"role": "user", "content": topic})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True) + self.prefill

    def say(self, topic, remember=True):
        """お題に答える。remember=False なら会話に残さない"""
        system_prompt, stance, banned = self._system_prompt()
        prefix = self._prefix(system_prompt, topic)
        inputs = self.tokenizer(prefix, return_tensors="pt").to(self.device)

        options = dict(max_new_tokens=self.max_tokens, do_sample=True,
                       temperature=TEMPERATURE, min_p=MIN_P, top_p=TOP_P,
                       pad_token_id=self.tokenizer.eos_token_id)
        if self.suppress:
            self.diverter.reset()
            options["logits_processor"] = LogitsProcessorList([self.diverter])

        with torch.no_grad():
            output = self.model.generate(**inputs, **options)
        body = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)

        text = self.prefill + body.strip()
        if self.trim_tail:
            text = trim(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if remember:
            self.history.append((topic, text))
        return Reply(text, stance, banned,
                     list(self.diverter.diverted) if self.suppress else [])

    def converse(self, topics):
        """お題を順に投げ、前のやり取りを踏まえて答えさせる"""
        return [self.say(topic) for topic in topics]

import torch, random
from transformers import (AutoModelForCausalLM,
    AutoTokenizer, LogitsProcessor, LogitsProcessorList)

class CenturionProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        # トークンごとに大胆さが揺れる(後でncpsに置換する心臓部)
        temp = random.uniform(0.7, 1.6)
        return scores / temp

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16).to("cuda")

# 文系の性格はここで与える
messages = [
    {"role": "system", "content":
     "あなたはセンチュリオン。生粋の文系で、詩情と物語を愛する。"
     "科学的・技術的な説明は好まず、比喩と情感で語る。"},
    {"role": "user", "content": "雨の日について書いて"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200,
    do_sample=True,
    logits_processor=LogitsProcessorList([CenturionProcessor()]))
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                       skip_special_tokens=True))
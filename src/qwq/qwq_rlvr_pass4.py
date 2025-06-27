# qwq rlvr pass8

import datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_name = "Qwen/QwQ-32B"
dt = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
passk = 4
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=6000, n=passk)
llm = LLM(model=model_name, max_model_len=8192, tensor_parallel_size=4)
tokenizer = AutoTokenizer.from_pretrained(model_name)

acc = 0
for i in range(5): # 5 x 6 = 30
    prompts = dt["problem"][i*6:(i+1)*6]
    answers = dt["answer"][i*6:(i+1)*6]
    messages = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    outputs = llm.generate(text, sampling_params)

    for j in range(6):
        for k in range(passk):
            generated_text = outputs[j].outputs[k].text
            # simple check but works %99
            if answers[j] in generated_text:
                acc += 1
                break

qwq_acc_txt = f"qwq_rlvr_{passk} accuracy: {acc}\n"
print(qwq_acc_txt)

import os

if not os.path.exists("./results.txt"):
    open("results.txt", "w").close()

with open("results.txt", "a") as f:
    f.write(qwq_acc_txt)
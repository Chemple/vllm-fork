from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
    "The life is", 
    "The vLLM is"
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# meta-llama/Llama-2-7b-hf 权重已下载到本地的 ckpts 路径
# 如果还没下载，可以直接 
llm = LLM(model="Qwen/Qwen2.5-Coder-3B")
# llm = LLM(model="./ckpts/llama-2-7b-hf")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
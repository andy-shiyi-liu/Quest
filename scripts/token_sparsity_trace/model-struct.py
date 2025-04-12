from transformers import AutoTokenizer
from quest import LlamaForCausalLM
import torch

MODEL_PATH = "/fact_home/shiyiliu/coding/Quest/data/model/Llama-2-7b-chat-hf"
DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
torch.set_default_dtype(DTYPE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE)
model.quest_init(page_size=16, max_seq_len=8192, token_budget=1024)
# 打印模型结构（注意：大模型结构可能很长）
# print(model)

# # 或者查看模型配置
# print(model.config)

# # 要查看MoE特定结构，可以检查模型中的专家层：
# for name, module in model.named_modules():
#     if "expert" in name.lower() or "moe" in name.lower():
#         print(f"Layer name: {name}")
#         print(f"Module: {module}")

# from transformers import AutoTokenizer, AutoModelForCausalLM
from torchinfo import summary

# # 加载模型和分词器
# model_name = "Qwen/Qwen1.5-MoE-A2.7B"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# 生成一个虚拟输入
inputs = tokenizer("Hello, world!", return_tensors="pt").to(DEVICE)

# 打印模型摘要
summary(
    model,
    input_data=(inputs["input_ids"], inputs["attention_mask"]),
    dtypes=["torch.LongTensor", "torch.LongTensor"],
    verbose=1,
    depth=10
)
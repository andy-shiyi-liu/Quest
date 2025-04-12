from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型名称
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 下载并加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存模型到指定路径
save_directory = "/fact_home/shiyiliu/coding/Quest/data/model/Llama-2-7b-chat-hf"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

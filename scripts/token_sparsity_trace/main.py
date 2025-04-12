import subprocess
import sys
from pathlib import Path
import json
from transformers import AutoTokenizer
import torch
import argparse
from inference import inference
from datasets import load_dataset
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from parse import parse_output


def dtype_type(dtype_str):
    """Convert string to torch.dtype"""
    dtype_str = dtype_str.lower()
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    raise argparse.ArgumentTypeError(f"Unsupported dtype: {dtype_str}")


RUNTIME_CFGS = ["quest", "hg"]

parser = argparse.ArgumentParser()

# Model and hardware configuration
parser.add_argument(
    "--model_path",
    type=str,
    default="/fact_home/shiyiliu/coding/Quest/data/model/Llama-2-7b-chat-hf",
    help="Path to pretrained model",
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)"
)
parser.add_argument(
    "--dtype",
    type=dtype_type,
    default="float16",
    help="Data type for computation (float16, float32, bfloat16)",
)

# Quest-specific configuration
parser.add_argument(
    "--page_size", type=int, default=16, help="Page size for memory management"
)
parser.add_argument(
    "--max_seq_len",
    type=int,
    default=8192,
    help="Maximum sequence length supported",
)
parser.add_argument(
    "--token_budget",
    type=int,
    default=1024,
    help="Token budget for memory allocation",
)

# Generation configuration
parser.add_argument(
    "--max_length",
    type=int,
    default=2048,
    help="Maximum length for generated sequence",
)

# Runtime method selection
parser.add_argument(
    "--method",
    choices=RUNTIME_CFGS,
    default="quest",
    help="Runtime method to use (quest or hg)",
)

# Prompt configuration
parser.add_argument(
    "--dataset",
    type=str,
    default="openai/gsm8k",
    help="Dataset for generation",
)

# Output configuration
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Directory to save the output JSON file",
)
parser.add_argument(
    "--dump_n_sample",
    type=int,
    default=100,
    help="Dump every N samples to a new file",
)

args = parser.parse_args()

# Initialize based on CLI arguments
MODEL_PATH = args.model_path
DEVICE = torch.device(args.device)
DTYPE = args.dtype
PAGE_SIZE = args.page_size
MAX_SEQ_LEN = args.max_seq_len
DUMP_N_SAMPLE = args.dump_n_sample
DATASET = args.dataset
OUTPUT_DIR = Path(script_dir / args.output_dir / DATASET)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    torch.set_default_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if args.method == "quest":
        from quest import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE
        )
        # Initialize Quest Controller with CLI parameters
        model.quest_init(
            page_size=PAGE_SIZE, max_seq_len=MAX_SEQ_LEN, token_budget=args.token_budget
        )
        print(f"Page Size: {PAGE_SIZE}\nMax Seq Len: {MAX_SEQ_LEN}")
    else:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE
        )

    # save inference config
    with open(OUTPUT_DIR / "inference_config.json", "w") as f:
        json.dump(
            {
                "model_path": MODEL_PATH,
                # "device": DEVICE,
                # "dtype": DTYPE,
                "page_size": PAGE_SIZE,
                "max_seq_len": MAX_SEQ_LEN,
                "token_budget": args.token_budget,
                "max_length": args.max_length,
                "method": args.method,
                "dataset": DATASET,
            },
            f,
        )

    dataset = load_dataset(DATASET, "main")
    dataset = dataset["train"]["question"]

    # 文件管理配置
    total_samples = len(dataset)
    current_sample = 0
    file_counter = 0
    file_template = "traces_{}.json"

    current_file = (OUTPUT_DIR / file_template.format(file_counter)).open("w")

    for prompt in tqdm(dataset):
        current_sample += 1
        buffer = StringIO()
        with redirect_stdout(buffer):
            inference(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=args.max_length,
                device=DEVICE,
            )
        output_text = buffer.getvalue()
        # Parse the output
        result = parse_output(output_text)

        # 写入文件
        current_file.write(json.dumps(result) + "\n")

        # 切换文件条件：达到分片数量且不是最后一个样本
        if current_sample % DUMP_N_SAMPLE == 0 and current_sample < total_samples:
            current_file.close()
            file_counter += 1
            current_file = (OUTPUT_DIR / file_template.format(file_counter)).open("w")

        if args.method == "quest":
            model.quest_clear()
        torch.cuda.empty_cache()

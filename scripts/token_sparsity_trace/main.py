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
    "--device", type=str, default="cuda", help="Device to use (e.g., cuda, cpu)"
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

# Job control
parser.add_argument(
    "--n_sample",
    type=int,
    default=25,
    help="Dump every N samples to a new file",
)
parser.add_argument(
    "--job_id",
    type=int,
    default=0,
    help="Job ID for slurm",
)

args = parser.parse_args()

# Initialize based on CLI arguments
MODEL_PATH = args.model_path
DEVICE = torch.device(args.device)
DTYPE = args.dtype
PAGE_SIZE = args.page_size
MAX_SEQ_LEN = args.max_seq_len
DATASET = args.dataset
if Path(args.output_dir).is_absolute():
    OUTPUT_DIR = Path(args.output_dir)
else:
    OUTPUT_DIR = Path(script_dir / args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLE = args.n_sample
JOB_ID = args.job_id

if __name__ == "__main__":
    dataset = load_dataset(DATASET, "main")
    dataset = dataset["train"]["question"]

    if N_SAMPLE * (JOB_ID + 1) > len(dataset):
        # nothing to do
        exit(0)
    dataset = dataset[N_SAMPLE * JOB_ID : N_SAMPLE * (JOB_ID + 1)]

    torch.set_default_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if args.method == "quest":
        from quest import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            MODEL_PATH, device_map=DEVICE, torch_dtype=DTYPE
        ).eval()
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
    if not (OUTPUT_DIR / "inference_config.json").exists():
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

    # 文件管理配置
    file_template = "traces_{}.json"

    current_file = (OUTPUT_DIR / file_template.format(JOB_ID)).open("a")

    for prompt in tqdm(dataset):
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

        if args.method == "quest":
            # print("total_seq_len: ", result["output_seq_len"] + result["input_seq_len"])
            model.quest_clear()
        torch.cuda.empty_cache()

    current_file.close()
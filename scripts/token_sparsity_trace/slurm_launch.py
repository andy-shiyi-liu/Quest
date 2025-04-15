import argparse
import sys
from pathlib import Path
import torch
from datasets import load_dataset
import os

script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

from parse import parse_output

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
    type=str,
    default="float16",
    help="Data type for computation (float16, float32, bfloat16)",
)

# Quest-specific configuration
parser.add_argument(
    "--page_size", type=int, default=8, help="Page size for memory management"
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
    default=64,
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
    default="output/openai-gsm8k",
    help="Directory to save the output JSON file",
)

# Job control
parser.add_argument(
    "--n_sample",
    type=int,
    default=20,
    help="Dump every N samples to a new file",
)
parser.add_argument(
    "--job_name",
    type=str,
    default="quest-token-trace",
    help="Name of the job to be run",
)
parser.add_argument(
    "--cpus-per-task",
    type=int,
    default=8,
    help="Number of CPUs per task",
)
parser.add_argument(
    "--gpus",
    type=int,
    default=1,
    help="Number of GPUs to use",
)
parser.add_argument(
    "--partition",
    type=str,
    default="gpu-scavenger",
    help="Partition to submit the job to",
)
parser.add_argument(
    "--mem",
    type=str,
    default="32G",
    help="Memory allocation for the job",
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

SLURM_OUTPUT_DIR = OUTPUT_DIR / "slurm_output"
SLURM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLURM_SCRIPT_DIR = OUTPUT_DIR / "slurm_scripts"
SLURM_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

PY_SCRIPT_PATH = Path(script_dir / "main.py")

N_SAMPLE = args.n_sample

slurm_script = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus={gpus}
#SBATCH --partition={partition}
#SBATCH --mem={mem}
#SBATCH --output={slurm_output}.out
"""
python_launch_script = """
source ~/.bashrc && conda activate quest && python {PY_SCRIPT_PATH} \\
    --model_path {model_path} \\
    --device {device} \\
    --dtype {dtype} \\
    --page_size {page_size} \\
    --max_seq_len {max_seq_len} \\
    --token_budget {token_budget} \\
    --max_length {max_length} \\
    --method {method} \\
    --dataset {dataset} \\
    --output_dir {output_dir} \\
    --n_sample {n_sample} \\
    --job_id {job_id}
"""


if __name__ == "__main__":
    dataset = load_dataset(DATASET, "main")
    dataset = dataset["train"]["question"]
    datasetLen = len(dataset)

    nJobs = datasetLen // N_SAMPLE
    if datasetLen % N_SAMPLE != 0:
        nJobs += 1

    for jobID in range(nJobs):
        slurm_script_content = slurm_script.format(
            job_name=args.job_name,
            cpus_per_task=args.cpus_per_task,
            gpus=args.gpus,
            partition=args.partition,
            mem=args.mem,
            slurm_output=SLURM_OUTPUT_DIR / f"{args.job_name}_{jobID}.log",
        )

        python_script_content = python_launch_script.format(
            PY_SCRIPT_PATH=PY_SCRIPT_PATH,
            model_path=MODEL_PATH,
            device=DEVICE,
            dtype=DTYPE,
            page_size=PAGE_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            token_budget=args.token_budget,
            max_length=args.max_length,
            method=args.method,
            dataset=DATASET,
            output_dir=OUTPUT_DIR,
            n_sample=N_SAMPLE,
            job_id=jobID,
        )

        script = slurm_script_content + python_script_content
        script_name = f"{args.job_name}_{jobID}.sh"
        script_path = SLURM_SCRIPT_DIR / script_name
        with open(script_path, "w") as f:
            f.write(script)

        # launch the job
        os.system(f"chmod +x {script_path} && sbatch {script_path}")
        # os.system(f"chmod +x {script_path} && {script_path}")

import re


def parse_output(output_text):
    result = {"layers": [], "output": ""}

    header_pattern = re.compile(
        r"Prompt: \s*([\s\S]*?)\n"
        r"Input Sequence Length: (\d+)",
        re.MULTILINE,
    )

    header_match = header_pattern.search(output_text)
    assert header_match, "No match found for header pattern"
    result.update(
        {
            "prompt": header_match.group(1).strip(),
            "input_seq_len": int(header_match.group(2)),
        }
    )

    layer_blocks = re.findall(
        r"(layer_idx: \d+\n.*?(?=\n----+\n|\Z))", output_text, flags=re.DOTALL
    )

    layer_pattern = re.compile(
        r"layer_idx: (\d+)\n"
        r"paged_kv_indices:\s*tensor\(([\s\S]*?)\)\n"
        r"paged_kv_indptr:\s*tensor\(([\s\S]*?)\)\n"
        r"paged_kv_last_page_len:\s*(\d+)\n"
        r"paged_kv_last_page_idx:\s*(\d+)",
        re.MULTILINE,
    )

    for block in layer_blocks:
        match = layer_pattern.search(block)
        if match:
            result["layers"].append(
                {
                    "layer_idx": int(match.group(1)),
                    "paged_kv_indices": parse_tensor(match.group(2)),
                    "paged_kv_indptr": parse_tensor(match.group(3)),
                    "paged_kv_last_page_len": int(match.group(4)),
                    "paged_kv_last_page_idx": int(match.group(5)),
                }
            )

    stats_pattern = re.compile(
        r"Total Output Token Size:\s*(\d+)\n"
        r"Generated Token Size \(excluding input\):\s*(\d+)",
        flags=re.MULTILINE,
    )
    stats_match = stats_pattern.search(output_text)
    assert stats_match, "No match found for stats pattern"

    result.update(
        {
            "output_seq_len": int(stats_match.group(2)),
        }
    )

    # 修正后的输出处理（支持大写Output和多行内容）
    output_pattern = re.compile(
        r"\n([Oo]utput):\s*([\s\S]*?)(?=\n----+|\Z)", flags=re.IGNORECASE | re.DOTALL
    )
    output_match = output_pattern.search(output_text)
    if output_match:
        # 清理输出内容
        raw_output = output_match.group(2).strip()
        # 移除可能存在的结尾分隔线
        cleaned_output = re.sub(r"\n-+\s*$", "", raw_output).strip()
        result["output"] = cleaned_output

    return result


def parse_tensor(tensor_str):
    clean_str = tensor_str.replace("tensor(", "").replace(")", "")
    rows = re.findall(r"\[.*?\]", clean_str.replace("\n", ""))
    return [[int(num) for num in re.findall(r"\d+", row)] for row in rows]


if __name__ == "__main__":
    with open(
        "/fact_home/shiyiliu/coding/Quest/scripts/token_sparsity_trace/tmp/test.log",
        "r",
    ) as f:
        output_text = f.read()
    parsed_output = parse_output(output_text)

    with open(
        "/fact_home/shiyiliu/coding/Quest/scripts/token_sparsity_trace/tmp/parsed_output.json",
        "w",
    ) as f:
        import json

        json.dump(parsed_output, f)

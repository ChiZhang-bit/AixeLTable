import json

file_path = "new_result/tabfact/gpt4o/write_prompt.jsonl"

total_input_tokens = 0
total_output_tokens = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        total_input_tokens += data.get("input_tokens", 0)
        total_output_tokens += data.get("output_tokens", 0)

print(f"📥 input_tokens 总和: {total_input_tokens}")
print(f"📤 output_tokens 总和: {total_output_tokens}")
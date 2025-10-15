import json
import re
import sys
from processing_format import get_row_description, get_col_description, get_row_flattened
from generate_solution_plan import get_solution_plan
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from get_sub_table import retrieve_final_subtable, retrieve_final_subtable_add
from utils.processing import clean_table, index_table
from generate_answer import generate_final_answer, generate_noplan_answer
from utils.processing import sample_table_rows
import tiktoken
from utils.processing import list_to_markdown, sample_table_rows

input_tokens, output_tokens = 0, 0


def count_tokens(text, model="gpt-3.5-turbo"):
    """
    Count the number of tokens in the input text.
    :param text: Input text
    :param model: Model used for tokenization (affects tokenizer)
    :return: Number of tokens
    """
    # Get the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    
    # Encode the text and return the token count
    tokenized = encoding.encode(text)
    return len(tokenized)


def get_col_template(table, prompt):
    """Randomly sample table rows and format a column prompt."""
    header, sampled_rows = sample_table_rows(table)
    markdown_header = "| " + " | ".join(header) + " |\n"
    markdown_rows = ""
    for row in sampled_rows:
        markdown_rows += "| " + " | ".join(row) + " |\n"

    # Generate the column prompt
    prompt = prompt.format(header=markdown_header, sampled_rows=markdown_rows)
    return prompt


# Example: for computing col prompt token cost
# def process_single_table(d, col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt):
#     """Compute column prompt token cost."""
#     item = json.loads(d)
#     table = item["table_text"]
#     cleaned_table = clean_table(table)
#     col_template_prompt = get_col_template(cleaned_table, col_prompt)
#     col_template_prompt_token = count_tokens(col_template_prompt)
#     return col_template_prompt_token


def process_single_table(col_prompt, plan_prompt, final_reasoning_prompt, noplan_reasoning_prompt):
    """Compute solution plan token cost."""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]

    cleaned_table = clean_table(table)
    # Generate solution plan
    header, sampled_rows = sample_table_rows(cleaned_table)

    markdown_table = list_to_markdown(header, sampled_rows)
    input_prompt = plan_prompt.format(question=question, table=markdown_table)
    token = count_tokens(input_prompt)

    return token


def old():
    """Estimate token usage for final reasoning stage."""
    # Load input data
    with open("result/final_reasoning/wtq/gpt4o-mini/read_result_1113_test_gpt4omini.json", 'r') as f:
        data = json.load(f)

    # Load the reasoning prompt
    with open("prompt/final_reasoning.md", "r") as f:
        final_reasoning_prompt = f.read()

    count = 0
    input_tokens = 0

    for key, value in data.items():
        tmp = []
        answer = value["pred_answer"]
        input_tokens += count_tokens(answer)

        if "final_sub_table" not in value.keys():
            continue

        solution_plan = value["solution_plan"]
        Flag = False
        for s in solution_plan:
            tmp.append(s["Top k"])
        if "all" in tmp:
            continue
        else:
            final_sub_table = value["final_sub_table"]
            tableprompt = list_to_markdown(final_sub_table[0], final_sub_table[1:])
            prompt = final_reasoning_prompt + str(solution_plan) + tableprompt
            count += 1
            input_tokens += count_tokens(prompt)

    print(input_tokens)
    print(count)
    print(input_tokens / len(data))
    return None


def dp_count():
    """Estimate token usage for plan-free reasoning prompts."""
    # Load input data
    with open("dataset/wikitq/test/4096/4096.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/noplan_reasoning.md", "r") as f:
        noplan_reasoning_prompt = f.read()

    count = 0
    input_tokens = 0

    for index, d in enumerate(data):
        item = json.loads(d)
        table = item["table_text"]
        question = item["statement"]
        cleaned_table = clean_table(table)
        indexed_cleaned_table = index_table(cleaned_table)

        col_headers = indexed_cleaned_table[0]
        table = indexed_cleaned_table[1:]
        table_md = "| " + " | ".join(col_headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(col_headers)) + " |\n"
        for row in table:
            table_md += "| " + " | ".join(map(str, row)) + " |\n"

        formatted_prompt = noplan_reasoning_prompt.format(question=question, table=table_md)

        input_tokens += count_tokens(formatted_prompt)
        count += 1

    print(input_tokens)
    print(count)
    print(input_tokens / count)

    return None


old()

import json
import random
import sys
import re
from tqdm import tqdm
from utils.request_gpt import request_gpt_chat
from utils.processing import list_to_markdown, sample_table_rows
from concurrent.futures import ThreadPoolExecutor, as_completed


def clean_header(header):
    """Clean column names so each contains only letters, digits, and underscores. If empty, return 'null'."""
    cleaned_header = []
    for column_name in header:
        if not column_name.strip():  # Check for empty or whitespace-only
            cleaned_name = 'null'
        else:
            # Replace non-alphanumeric/underscore characters
            cleaned_name = re.sub(r'\W+', '_', column_name)
            # Collapse consecutive underscores and strip leading/trailing underscores
            cleaned_name = re.sub(r'_+', '_', cleaned_name).strip('_')
        cleaned_header.append(cleaned_name)
    return cleaned_header


def load_question_type_map(label_file):
    """Read the label file and return a dict mapping question -> type."""
    question_type_map = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            question_type_map[item["question"]] = item["type"]
    return question_type_map


def get_dag(table, question, question_type, dag_prompt_template):
    header, sampled_rows = sample_table_rows(table)
    markdown_table = list_to_markdown(header, sampled_rows)
    # input = dag_prompt.format(question=question, table=markdown_table)
    # print(input)
    # print("****************")

    # Load few-shot examples according to question_type
    few_shot_file_map = {
        'parallel': 'prompt/fewshot/fewshot_parallel.txt',
        'sequential': 'prompt/fewshot/fewshot_sequential.txt',
        'hybrid': 'prompt/fewshot/fewshot_hybrid.txt',
        'tabfact': 'prompt/fewshot/fewshot_tabfact.txt'
    }

    if question_type not in few_shot_file_map:
        raise ValueError(f"Unsupported question_type: {question_type}")

    fewshot_path = few_shot_file_map[question_type]

    with open(fewshot_path, 'r', encoding='utf-8') as f:
        fewshot_examples = f.read()

    dag_prompt = dag_prompt_template.format(
        fewshot=fewshot_examples.strip(),
        question=question,
        table=markdown_table
    )

    # Maximum retry attempts
    max_attempts = 10
    for attempt in range(max_attempts):
        dag = request_gpt_chat(dag_prompt)
        # print(dag)

        # Validate the generated DAG
        dag_dict = validate_dag(dag)
        if dag_dict:
            return dag_dict
        else:
            print(f"Attempt {attempt + 1}: Generated DAG does not match the expected format, retrying...")

    raise ValueError("Failed to generate DAG in the expected format after multiple attempts.")


def validate_dag(dag_plan):
    """
    Validate whether the generated DAG plan meets the requirements:
    1. Valid JSON format;
    2. Each node contains the fields: NodeID, Sub-Level-Question, Action, Top k, Next;
    3. The Action for all nodes must be either "Retrieval" or "Reasoning";
    4. Terminal nodes (no successors) must have Action = "Reasoning";
    5. The graph is acyclic (a DAG).
    """
    try:
        dag_plan = dag_plan.strip().strip("```").strip()
        match = re.search(r'(\[\s*\{.*\}\s*\])', dag_plan, re.DOTALL)
        if match:
            dag = json.loads(match.group(1))
        else:
            return None

        # Check structure and required fields
        required_fields = {'NodeID', 'Sub-Level-Question', 'Action', 'Top k', 'Next'}
        if not (isinstance(dag, list) and all(isinstance(node, dict) and required_fields.issubset(node.keys()) for node in dag)):
            return None

        node_dict = {node['NodeID']: node for node in dag}

        # Validate Action field values
        if not all(node['Action'] in ['Retrieval', 'Reasoning'] for node in dag):
            return None

        # Terminal nodes (empty Next) must be Reasoning
        for node in dag:
            if len(node['Next']) == 0 and node['Action'] != 'Reasoning':
                return None

        # Detect cycles via DFS (topological validation)
        visited = set()
        on_stack = set()

        def has_cycle(node_id):
            if node_id in on_stack:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            on_stack.add(node_id)
            for next_id in node_dict[node_id]['Next']:
                if next_id not in node_dict or has_cycle(next_id):
                    return True
            on_stack.remove(node_id)
            return False

        if any(has_cycle(node['NodeID']) for node in dag):
            return None

        return dag

    except json.JSONDecodeError:
        pass

    return None


def process_single_table(index, d, dag_prompt):
    """Process a single table and return the result record."""
    item = json.loads(d)
    table = item["table_text"]
    question = item["statement"]

    # Clean header
    header = table[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table[1:]

    question_type_map = load_question_type_map("dataset/wikitq/valid/4096/4096_daglabeled.jsonl")
    question_type = question_type_map.get(question, "hybrid")  # default to hybrid fallback

    try:
        # Generate solution plan (DAG)
        dag = get_dag(cleaned_table, question, question_type, dag_prompt)
        record_data = {
            "question": question,
            "question_type": question_type,
            "DAG": dag,
            "table": cleaned_table,
        }
    except Exception as e:
        print(f"Error encountered at index {index}: {e}. Skipping this iteration.")
        record_data = {
            "question": question,
            "DAG": "null",
            "table": cleaned_table,
        }

    return record_data


def main():
    with open("dataset/wikitq/valid/4096/4096_small.jsonl", 'r') as f:
        data = f.readlines()

    with open("prompt/get_dag_notype.md", "r") as f:
        dag_prompt = f.read()

    # Thread pool: process each table concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(process_single_table, index, d, dag_prompt)
            for index, d in enumerate(data)
        ]

        with open("new_result/dag/generated_dag.jsonl", 'a', encoding='utf-8') as f:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing data"):
                result = future.result()

                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

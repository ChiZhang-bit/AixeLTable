import sys
import json
from utils.request_gpt import request_gpt_chat


def rewrite_question(question, headers):
    """
    Rewrite a question by incorporating schema information (column headers).
    Uses a schema-linking prompt to make the question more explicit.
    """
    with open("prompt/prompt_schema_linking.md", "r") as f:
        prompt = f.read()
    prompt = prompt.format(question=question, headers=str(headers))
    rewritten_question = request_gpt_chat(prompt=prompt)

    return rewritten_question


if __name__ == "__main__":
    with open("dataset/4096.jsonl", 'r') as f:
        data = f.readlines()

    for d in data:
        item = json.loads(d)
        table = item["table_text"]
        question = item["statement"]
        headers = table[0]

        rewritten_question = rewrite_question(question, headers)
        print(rewritten_question)

import random
import json
import sys
import re
from utils.request_gpt import request_gpt_chat, request_gpt_embedding
from utils.processing import sample_table_rows


def get_row_template(table, prompt):
    # Randomly sample rows from the table
    header, sampled_rows = sample_table_rows(table)
    markdown_header = "| " + " | ".join(header) + " |\n"
    markdown_rows = ""
    for row in sampled_rows:
        markdown_rows += "| " + " | ".join(row) + " |\n"

    # Generate the prompt for the row template
    prompt = prompt.format(header=markdown_header, sampled_rows=markdown_rows)
    # print(prompt)

    max_attempts = 10  # Limit retry attempts
    for attempt in range(max_attempts):
        row_template = request_gpt_chat(prompt=prompt)

        # Validate the generated output
        if validate_row_template(row_template, header):
            return row_template
        else:
            print(f"Attempt {attempt + 1}: Generated row template does not match the expected format, retrying...")
            # print("Wrong row template:", row_template)
    raise ValueError("Failed to generate row template in the expected format after multiple attempts.")


def get_col_template(table, prompt):
    # Randomly sample rows from the table
    header, sampled_rows = sample_table_rows(table)
    markdown_header = "| " + " | ".join(header) + " |\n"
    markdown_rows = ""
    for row in sampled_rows:
        markdown_rows += "| " + " | ".join(row) + " |\n"

    # Generate the prompt for the column template
    prompt = prompt.format(header=markdown_header, sampled_rows=markdown_rows)
    # print(prompt)

    max_attempts = 10  # Limit retry attempts
    for attempt in range(max_attempts):

        col_template = request_gpt_chat(prompt=prompt)
        # Validate the generated output
        if validate_col_template(col_template, header):
            return col_template
        else:
            print(f"Attempt {attempt + 1}: Generated template does not match the expected format, retrying...")
            # print("Wrong col template:", col_template)
    raise ValueError("Failed to generate column template in the expected format after multiple attempts.")


def validate_col_template(col_template, header):
    """Validate that each column description is on one line and follows the required format."""
    # Define a regex pattern for format matching
    pattern = r"^Col\d+ ## .+: .+(\n|$)"

    # Split the generated template and validate each line
    col_template_lines = col_template.strip().splitlines()

    # Check whether each column has one valid description line
    if len(col_template_lines) == len(header) and all(re.match(pattern, line) for line in col_template_lines):
        return True
    return False


def validate_row_template(row_template, header):
    """Validate that the generated row template follows the expected format 
    and all placeholders match table column names."""
    # Extract placeholders and compare with the header
    placeholders = re.findall(r"\{(.*?)\}", row_template)
    return all(placeholder in header for placeholder in placeholders)


def get_row_description(table, row_prompt):
    """
    Generate a natural language description for each row in the table.
    """
    row_template = get_row_template(table, row_prompt)
    # print("True Template:", row_template)

    header, *rows = table

    descriptions = []
    for row in rows:
        row_data = dict(zip(header, row))
        description = row_template.format(**row_data)
        descriptions.append(description)

    # for desc in descriptions:
    #     print(desc)
    return descriptions


def get_col_description(table, col_prompt):
    """
    Generate a natural language description for each column in the table.
    """
    col_template = get_col_template(table, col_prompt)
    column_descriptions = col_template.split('\n')
    # print("True column_descriptions:", column_descriptions)

    header, *rows = table

    column_texts = []

    for i, col_name in enumerate(header):
        description = column_descriptions[i] if i < len(column_descriptions) else "No description available."

        # column_values = "|".join(row[i] for row in rows)
        # column_text = f"{description} The values in this column are: {column_values}"

        column_text = description
        column_texts.append(column_text)

    # for column_text in column_texts:
    #     print(column_text)

    return column_texts


def get_row_flattened(table):
    """
    Flatten each row in the table into a single string.
    """
    # Initialize a list to store flattened rows
    flattened_rows = []

    # Skip the first row (header) and process the rest
    for row in table[1:]:
        # Join all elements in the row into one string
        flattened_row = ''.join(row)
        # Add to the flattened list
        flattened_rows.append(flattened_row)

    return flattened_rows


# if __name__ == "__main__":
#     with open("dataset/4096_test.jsonl", 'r') as f:
#         data = f.readlines()
#     for d in data:
#         item = json.loads(d)
#         table = item["table_text"]
#
#     with open("prompt/get_row_template.md", "r") as f:
#         row_prompt = f.read()
#
#     with open("prompt/get_col_template.md", "r") as f:
#         col_prompt = f.read()
#
#     row_descriptions = get_row_description(table, row_prompt)
#     col_descriptions = get_col_description(table, col_prompt)
#
#     print(row_descriptions)
#     print(col_descriptions)
#
#     embedding = request_gpt_embedding(row_descriptions[0])
#     print(embedding)
#     print(len(embedding))

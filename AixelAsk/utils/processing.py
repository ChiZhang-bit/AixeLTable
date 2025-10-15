import random
import re
import copy

def sample_table_rows(table, num_samples=5):
    """
    Randomly sample a specified number (num_samples) of rows 
    from a 2D list representing a table.
    """
    # Get the table header
    header = table[0]
    
    # Randomly sample rows excluding the header
    rows = random.sample(table[1:], num_samples)
    
    return header, rows


def list_to_markdown(header, rows):
    """Convert a table (header + rows) into Markdown table format."""
    markdown_table = "| " + " | ".join(header) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in rows:
        markdown_table += "| " + " | ".join(row) + " |\n"
    return markdown_table


def clean_header(header):
    """
    Clean column names so each contains only letters, digits, or underscores.
    If a column name is empty, replace it with 'null'.
    """
    cleaned_header = []
    for column_name in header:
        if not column_name.strip():  # Check if the name is empty or whitespace
            cleaned_name = 'null'
        else:
            # Replace non-alphanumeric/underscore characters
            cleaned_name = re.sub(r'\W+', '_', column_name)
            # Remove duplicate underscores and trim leading/trailing ones
            cleaned_name = re.sub(r'_+', '_', cleaned_name).strip('_')
        
        cleaned_header.append(cleaned_name)
    return cleaned_header


def clean_table(table):
    """
    Return a cleaned copy of the table where the header is sanitized.
    """
    # Deep copy to avoid modifying the original table
    table_copy = copy.deepcopy(table)
    header = table_copy[0]
    cleaned_header = clean_header(header)
    cleaned_table = [cleaned_header] + table_copy[1:]
    
    return cleaned_table


def index_table(table):
    """
    Add a 'row index' column to the table, labeling rows as 'row 1', 'row 2', etc.
    """
    # Deep copy to avoid modifying the original table
    table_copy = copy.deepcopy(table)
    
    # Insert "row index" header
    table_copy[0].insert(0, "row index")
    
    # Insert row indices starting from 1
    for i in range(1, len(table_copy)):
        table_copy[i].insert(0, f"row {i}")
    
    return table_copy

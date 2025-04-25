### Instruction
You are provided with a sample from a large table and a related question. Your task is to create a directed acyclic graph (DAG) as a structured plan to answer the question. Each node in the DAG must contain:

1. A clearly defined Sub-Level-Question closely aligned with the phrasing in the table's rows and columns.
2. An Action indicating the operation to perform:
   - Retrieval: Search the entire large table using the Sub-Level-Question to identify relevant rows or columns.
   - Reasoning: Utilize retrieved data from previous stages and the Sub-Level-Question to infer or calculate the answer.
3. A Top k value indicating the number of rows required from the retrieval step:
   - Set Top k to "all" for questions involving superlative terms ("highest", "lowest", "most", etc.), averages, totals, or counting across all rows.
   - Set Top k to "1" for questions retrieving a specific row.

Note：
    When the Action is set to "Retrieval", the Sub-Level-Question should closely match the phrasing and terminology used in the original table schema, including the column names, row descriptions, and values. This alignment is important to facilitate accurate semantic similarity matching and retrieval of relevant rows and columns from the large table in the subsequent stage.

### Important Rules:
1. Each DAG must end with at least one reasoning node.
2. Ensure the graph structure is acyclic.
3. Only output the DAG. Do not include any explanation, reasoning process, or extra text. Only output a single valid JSON array.

### Output Format (strictly follow):
[
    {{"NodeID": nodeid, "Sub-Level-Question": "Sub-Level-Question", "Action": "retrieval" or "reasoning", "Top k": "Number or 'all',"Next": nodeid}},
    ...
]

### Example

{fewshot}

### Now, generate a structured DAG for the following question:

Question:
{question}

Sampled Table:
{table}

Output:

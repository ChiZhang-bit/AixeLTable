### Instructions
You are given a Question, a Relevant table, and a Directed Acyclic Graph (DAG) that outlines the reasoning process. Follow the "DAG" step-by-step to answer the "Question" using the "Relevant table" provided below. Only output the final answer. Do not include any explanation, reasoning process, or extra text.

### Example 1 
Question: What is the difference between the highest and lowest revenue companies in the automotive industry?

DAG:
Node 1:
Sub-Level-Question: What is the revenue of the highest revenue company in the automotive industry?
Next Node: 3

Node 2:
Sub-Level-Question: What is the revenue of the lowest revenue company in the automotive industry?
Next Node: 3

Node 3:
Sub-Level-Question: What is the difference between the highest and lowest revenue?
Next Node: null


Relevant table:

| Company Name | Industry    | Revenue | Profit |
|--------------|-------------|---------|--------|
| Toyota       | Automotive  | 256,722 | 21,180 |
| Volkswagen   | Automotive  | 253,965 | 10,104 |
| Ford         | Automotive  | 127,144 | 5,080  |


Question: What is the difference between the highest and lowest revenue companies in the automotive industry?

Answer:
129578

### Attention
1. Ensure your thought process strictly follows each stage in the "Plan."
2. Your output should contain only the answer of the question, in the shortest possible form, with no additional explanation or extra words.

Question:
{question}

Relevant table:
{table}

DAG:
{dag}

Question:
{question}

Answer:
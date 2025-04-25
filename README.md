# TableGuide

**TableGuide: A Stepwise-Guided Retrieval and Reasoning Framework for Large Table QA**

This repository contains the implementation of **TableGuide**, a novel framework specifically designed for question answering tasks over large-scale tables.

## Overview

This repository provides the code implementation for **TableGuide**, a stepwise-guided retrieval and reasoning framework designed for large-scale Table-based Question Answering (QA). 

TableGuide aims to improve table-based QA tasks by combining table retrieval and reasoning steps in a structured manner. The framework leverages state-of-the-art models and methods to efficiently process tabular data and answer complex questions that require deep reasoning across tables.

## Project Structure

This is the structure of the TableGuide project:

```latex
├── dataset/                    # Datasets
│   ├── TabFact+/               # TabFact+ dataset
│   ├── WikiTQ+/                # WikiTQ+ dataset
│   ├── WikiTQ-4k/              # WikiTQ-4k dataset
│   └── Scalability/            # Datasets for scalability experiments
├── open_source_LLM/            # open-source LLM deployment script
├── result/                     # Experimental results
│   ├── dag/                    # Results of DAG Solution Plan
│   ├── scalability/            # Scalability experiment results
│   ├── TabFact+/               # Results on TabFact+ dataset
│   ├── WikiTQ-4k/              # Results on WikiTQ-4k dataset
│   └── WikiTQ+/                # Results on WikiTQ+ dataset
├── TableGuide/                 # Core implementation of TableGuide
│   ├── prompt/                 # Prompt templates
│   ├── scripts/                # Script files and constants
│   └── utils/                  
├── README.md                   
├── requirements.txt            
└── run.sh                      # Script to execute experiments
```

## Data Format

All data is stored in `jsonl` format, where each entry contains the following fields:

```json
{
    "statement": "", 
    "table_text": [[]], 
    "answer": [], 
    "ids": ""
}
```

### Field Descriptions:

- **statement**: The question to be answered based on the provided table.
- **table_text**: The tabular data (in text format) associated with the question.
- **answer**: A list containing the answer(s) to the question.
- **ids**: A unique identifier for the data entry.

## Benchmark Dataset

+ Our dataset is publicly available in our repository.

  ### Datasets:
  
1. **WikiTQ-4k**:
     - **WikiTableQuestions (WikiTQ)** is a well-known benchmark dataset for question answering over structured tabular data. For our experiments, we filter the dataset to include only tables with more than 4k tokens, resulting in a subset of 488 entries.
2. **WikiTQ+**:
     - To augment the WikiTQ dataset, we extend medium-sized tables (containing 2k to 4k tokens) by adding additional rows generated using GPT-4o with prompt-based table generation techniques. These expanded tables now contain at least 4k tokens. The WikiTQ+ dataset includes 1023 entries.
3. **TabFact+**:
     + We manually expand medium-sized tables in TabFact dataset by adding rows while preserving factual consistency, ensuring each table contains at least 4k tokens. The TabFact+ dataset includes 260 entries, which are split into training and testing sets with a 6:4 ratio.

## Baseline Method

| Method         | Reference                                                    |
| -------------- | ------------------------------------------------------------ |
| Tapas          | [TAPAS](https://huggingface.co/docs/transformers/model_doc/tapas) |
| DIN-SQL        | [DIN-SQL](https://github.com/madhup/DIN-SQL)                 |
| Binder         | [Binder](https://github.com/zsong96wisc/Binder-TableQA)      |
| ReAcTable      | [ReAcTable](https://github.com/yunjiazhang/reactable)        |
| Few-Shot QA    | N/A                                                          |
| Dater          | [Dater](https://arxiv.org/pdf/2301.13808)                    |
| Chain-of-Table | [Chain-of-Table](https://github.com/google-research/chain-of-table) |

## Requirements

+ `openai == 1.52.2`
+ `tiktoken == 0.8.0`
+ `scikit-learn == 1.5.2`
+ `scipy == 1.13.1`

## Running the Code

You can run the code by executing the following command in your terminal:

```bash
bash run.sh
```

### Prompt

The prompt directory contains templates for interacting with LLMs. The key prompts utilized by TableGuide include:

- **`final_reasoning.md`**: Performs the final reasoning step, synthesizing retrieved data to answer the question.
- **`final_reasoning_DAG.md`**: Final reasoning using the DAG-structured solution plan.
- **`final_reasoning_tabfact.md`**: Final reasoning template specifically for TabFact+ dataset.
- **`get_col_template.md`**: Generates templates summarizing table columns based on schema.
- **`get_row_template.md`**: Generates templates summarizing table rows based on schema.
- **`get_dag.md`**: Generates the DAG-based solution plan from questions and table schema.
- **`get_dag_tabfact.md`**: DAG-based solution plan specifically adapted for TabFact+ dataset.
- **`noplan_reasoning.md`**: Template for reasoning without a solution plan (ablation study).
- **`noplan_reasoning_tabfact.md`**: Template for reasoning without a solution plan, adapted for TabFact+.
- **`prompt_schema_linking.md`**: Aligns the table schema with the query to enhance retrieval accuracy.

Prompts in TableGuide are dynamically selected based on the dataset, question complexity, and specific task requirements, ensuring structured guidance for effective retrieval and reasoning across large-scale tables.

## Setting Your OpenAI Key:

Before running the code, you need to set your `api_key` and `base_url` in `utils/request_gpt.py`.

```python
client = OpenAI(
    api_key="",  # Set your API key here
    base_url=""  # Set the base URL for the API here
)
```

## Deploy Open-source LLM

To deploy the open-source LLM server, configure and run the script `open_source_LLM\launch_llm_server.py`. Update the following parameters:

```python
model_path = '<path_to_your_model>'  # specify your model path here
port_map = {
    'QWen-7B': 6671
}  # port configuration for different models
```
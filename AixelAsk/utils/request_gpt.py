import os
import time
from openai import OpenAI
import requests
import json
import tiktoken

port_map = {
    'llama': 6666,
    'mistral': 6667,
    'phi': 6668,
    'QWen-7B': 6672
}


client = OpenAI(
    api_key="",
    base_url=""
)

def count_tokens(text, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def request_gpt_chat_1(prompt, model="gpt-4o-mini", retries=30):

    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
    e = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                # max_tokens=max_tokens,
                temperature=0.7
            )
            if not response.choices:
                raise ValueError("No choices returned from the GPT API.")
            answer = response.choices[0].message.content

            #   计算 token 数量
            input_tokens = count_tokens(prompt, model)
            output_tokens = count_tokens(answer, model)

            # 写入日志文件
            with open("new_result/tabfact/gpt4o/write_prompt.jsonl", "a", encoding="utf-8") as f:
                json.dump({
                    "prompt": prompt,
                    "answer": answer,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }, f, ensure_ascii=False)
                f.write("\n")
            return answer

        except Exception as error:
            e = error
            print(f"Error calling GPT API: {e}, sleeping for 1 second before retrying...")
            time.sleep(1)
            if "This model's maximum context length is 16385 tokens." in str(e):
                break

    print("Max retries exceeded.")
    return f"Error calling GPT API: {e}"


def request_gpt_chat(prompt, model='mistral', max_length=40000, temperature=0, retries=30):
    for attempt in range(retries):
        try:
            port = port_map[model]
            url = f"http://localhost:{port}/generate"
            headers = {
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
            }

            response = requests.post(url, headers=headers, data=json.dumps(payload))
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response key in JSON')
            else:
                return f"Error: {response.status_code}, {response.text}"
            
        except Exception as error:
            e = error
            print(f"Error calling Local LLM: {e}, sleeping for 1 second before retrying...")
            time.sleep(1)
            if "This model's maximum context length is" in str(e):
                break

    print("Max retries exceeded.")
    return f"Error calling Local LLM: {e}"
        


def request_gpt_embedding(input, retries=5):

    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     raise ValueError("API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                input=input,
                model="text-embedding-3-small"  
            )
            # 将嵌入向量存储在列表中
            embedding = response.data[0].embedding
            time.sleep(0.1)

            return embedding

        except Exception as e:
            if "Error code: 429" in str(e):
                print(f"Received 429 error, {e}, sleeping for 1 second before retrying...")
                time.sleep(1)
            else:
                print(f"Error calling GPT API: {e}")
                return None

    print("Max retries exceeded.")
    return None

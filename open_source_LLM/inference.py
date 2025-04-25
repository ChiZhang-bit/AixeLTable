import requests
import json


# port = 6660
# port = 6661
# port = 6665
# port = 6667

port_map = {
    'llama': 6666,
    'mistral': 6667,
    'phi': 6668,
    'QWen-7B': 6672
}


def get_llm_response(model, prompt, max_length=40000, temperature=0):
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


def get_llm_json_response(model, prompt, schema, max_tokens=None, temperature=0):
    port = port_map[model]
    url = f"http://localhost:{port}/generate"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "schema": schema,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response key in JSON')
    else:
        return f"Error: {response.status_code}, {response.text}"



# if __name__ == '__main__':
#     prompt = "the answer of 111 + 222 is?"
#     print(get_llm_response('QWen-7B',prompt))
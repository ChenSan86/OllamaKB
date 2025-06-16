import requests
import json

def test_openai_api(messages, stream=False):
    url = "http://127.0.0.1:8000/v1/chat/completions"
    payload = {
        "model": "deepseek-r1:7b",
        "messages": messages,
        "stream": stream
    }
    headers = {"Content-Type": "application/json"}
    if stream:
        with requests.post(url, data=json.dumps(payload), headers=headers, stream=True) as resp:
            print("流式响应：")
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    print(line)
    else:
        resp = requests.post(url, data=json.dumps(payload), headers=headers)
        print("非流式响应：")
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是芙宁娜知识库的专家助手。"},
        {"role": "user", "content": "芙宁娜的生日是哪天？"}
    ]
    test_openai_api(messages, stream=False)
    print("\n" + "="*40 + "\n")
    test_openai_api(messages, stream=True)

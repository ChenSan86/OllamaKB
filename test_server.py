import requests

def chat_stream():
    url = "http://127.0.0.1:8000/chat"
    print("===== 芙宁娜知识库对话模式（流式） =====")
    print("输入你的问题，输入 exit 或 quit 退出。\n")
    while True:
        question = input("你：").strip()
        if question.lower() in ("exit", "quit"):
            print("对话结束。")
            break
        if not question:
            continue
        print("助手：", end="", flush=True)
        with requests.post(url, json={"question": question}, stream=True) as resp:
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                print(chunk, end="", flush=True)
        print("\n" + "-"*40)

if __name__ == "__main__":
    chat_stream()

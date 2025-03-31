#from pathlib import Path
from typing import List
from llama_cpp import Llama  # for local LLM
# from openai import OpenAI  # for OpenAI use

# === CONFIG ===
llm_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
max_examples = 5
max_tokens = 100


def format_prompt(cluster_texts: List[str]) -> str:
    examples = "\n".join([f"{i+1}. {t.strip()}" for i, t in enumerate(cluster_texts[:max_examples])])
    return (
        "Här är några exempel på svenska lagtexter. "
        "Kan du föreslå en rubrik som beskriver vad de handlar om?\n\n"
        f"{examples}\n\nRubrik:"
    )


def label_cluster_local(prompt: str) -> str:
    llm = Llama(
        model_path=llm_path,
        n_ctx=2048,
        n_threads=4,
        temperature=0.7,
        stop=["\n"]
    )
    output = llm(prompt, max_tokens=max_tokens, echo=False)
    return output["choices"][0]["text"].strip()


# def label_cluster_remote(prompt: str) -> str:
#     import openai
#     openai.api_key = "sk-..."  # if using OpenAI
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Du är en juridisk expert."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#
#     return response["choices"][0]["message"]["content"].strip()


def generate_labels_for_clusters(topic_dict: dict):
    labels = {}
    for cluster_id, texts in topic_dict.items():
        if cluster_id == -1:
            labels[cluster_id] = "Brus / Övrigt"
            continue

        prompt = format_prompt(texts)

        label = label_cluster_local(prompt)
        print(f"Kluster {cluster_id}: {label}")
        labels[cluster_id] = label

    return labels

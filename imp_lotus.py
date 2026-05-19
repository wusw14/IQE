import argparse
import pandas as pd
import json
import os
from load_data import load_data
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from typing import List
from transformers import AutoTokenizer


def count_tokens_hf(text, model_name="gpt2"):
    """use Hugging Face tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)


def prepare_prompts(query, corpus):
    prompts = []
    for value in corpus:
        prompt = f"The {value} is the same as or a type of {query}.\nYour job is to determine whether the claim is true for the given context.\nDirectly answer with 'True' or 'False' without any other text."
        prompts.append(prompt)
    return prompts


def run_inference(
    prompts: List[str], max_tokens=1024, system_prompts: list = None, temperature=0.0
) -> List[str]:

    def generate_completion(prompt, system_prompt=None):
        messages = [
            {"role": "user", "content": prompt},
        ]
        system_prompt = "The user will provide a claim and some relevant context.\nYour job is to determine whether the claim is true for the given context.\nThe answer should be either True or False."
        # if system_prompt:
        #     messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=42,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content

    with ThreadPoolExecutor(max_workers=min(100, len(prompts))) as executor:
        if system_prompts:
            completions = list(
                executor.map(generate_completion, prompts, system_prompts)
            )
        else:
            completions = list(executor.map(generate_completion, prompts))
    return completions


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--llm", type=str, default="llama")
    parser.add_argument(
        "--index_combine_method",
        type=str,
        choices=["weighted", "merge"],
        default="weighted",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--mode", type=str, default="lotus")
    parser.add_argument("--port", type=int, default=1117)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.llm == "llama":
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        helper_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        api_key = "llama"
        port = 1170
        helper_port = 1108
    elif args.llm == "deepseek":
        model_name = "openai/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        api_key = "deepseek"
        port = 1117
        helper_port = 1107
    elif args.llm == "qwen":
        model_name = "Qwen/Qwen3.5-27B"
        helper_model_name = "Qwen/Qwen3.5-27B"
        api_key = "qwen3.5"
        port = args.port
        helper_port = args.port
    else:
        raise ValueError(f"Invalid LLM: {args.llm}")

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",  # vLLM server address
        api_key=api_key,  # dummy token
    )

    args.output_dir = f"results/{args.exp_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.llm}.json")
    # load results
    if os.path.exists(output_path) and args.exp_name != "debug":
        results = json.load(open(output_path, "r"))
        results = [d for d in results if len(d["pred"]) > 0]
        processed_queries = [d["query"] for d in results]
    else:
        results = []
        processed_queries = []
    # load data
    df, query_answer, query_template, path = load_data(args.dataset)
    args.path = path
    if args.dataset == "paper":
        batch_size = 1024
        attribute = "abstracts"
        reformat_template = "According to the abstract, the paper is about {query}."
        llm_template = (
            "The abstract of the paper is: {value}. Is this paper about {query}?"
        )
    elif args.dataset == "product":
        batch_size = 512
        attribute = "Product_Title"
        reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
        prefix = "The '{" + attribute + "}' " + "is the same as or a type of"
    else:
        cols = df.columns
        batch_size = 128
        attribute = cols[0]
        reformat_template = (
            f"According to the {attribute} name, the {attribute}"
            + " is the same as or a type of '{query}'."
        )
        prefix = "The '{" + attribute + "}' " + "is the same as or a type of"
    corpus = df[attribute].values.tolist()
    print(f"corpus size: {len(corpus)}")
    # solve query
    print(
        f"Total queries: {len(query_answer)}, processed queries: {len(processed_queries)}"
    )
    cnt = 0
    for query, answers in query_answer.items():
        if query in processed_queries:
            continue
        if len(answers) == 0:
            continue
        start_time = time.time()
        prompts = prepare_prompts(query, corpus)
        responses = run_inference(prompts, max_tokens=5)
        preds = []
        for value, response in zip(corpus, responses):
            response = response.lower()
            true_index, false_index = len(response), len(response)
            if "true" in response:
                true_index = response.index("true")
            if "false" in response:
                false_index = response.index("false")
            if true_index < false_index:
                preds.append(value)
        preds = list(set(preds))
        result = {"query": query, "pred": preds}
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        print(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 3:
            break
    # save results
    save_results(results, output_path)

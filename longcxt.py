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
    parts = 1
    while True:
        # split corpus into parts
        num_parts = (len(corpus) + parts - 1) // parts
        corpus_parts = [
            corpus[i * num_parts : (i + 1) * num_parts] for i in range(parts)
        ]
        if_within_limit = True
        prompts = []
        for corpus_part in corpus_parts:
            prompt = f"Given the query: {query}\nThe list of candidate table values: {corpus_part}. Please identify which values are the same as or a type of the query. Your response should be a list of values separated by ' | ' without any other text."
            prompts.append(prompt)
            if count_tokens_hf(prompt, model_name) > 30000:
                if_within_limit = False
                break
        if if_within_limit:
            break
        parts += 1
    print(f"parts: {parts}")
    return prompts


def run_inference(
    prompts: List[str], max_tokens=1024, system_prompts: list = None, temperature=0.0
) -> List[str]:

    def generate_completion(prompt, system_prompt=None):
        messages = [
            {"role": "user", "content": prompt},
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=42,
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
        model_name = "Qwen/Qwen2.5-72B-Instruct"
        helper_model_name = "Qwen/Qwen2.5-7B-Instruct"
        api_key = "qwen"
        port = 1172
        helper_port = 1107
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
        responses = run_inference(prompts, max_tokens=2000)
        preds = []
        for response in responses:
            vals = response.split(" | ")
            vals = [val.strip() for val in vals]
            preds.extend(vals)
        preds = list(set(preds))
        result = {"query": query, "pred": preds}
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        print(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 10:
            break
    # save results
    save_results(results, output_path)

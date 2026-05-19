import argparse
import json
import os
from load_data import load_data
import time
from reformulate import (
    reformulate,
    score_query,
    reformulate_simple,
    identify_semantic_gap,
    score_query,
)
from retrieve import retrieve_corpus
from iterative_check import llm_check_retrieved_objs
from query import Query
import numpy as np
from index import BM25Index, HNSWIndex
from rerank import rerank_retrieved_objs
from constants import CHECK_NUM
import copy
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../TAG-Bench")
    parser.add_argument("--dataset", type=str, default="TAG")
    parser.add_argument("--test_type", type=str, default="dev")
    parser.add_argument("--method", type=str, default="llm_check")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--stop_iter", type=int, default=1)
    parser.add_argument("--rethink", action="store_true")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--iterative_check", action="store_true")
    parser.add_argument(
        "--reform_type",
        type=str,
        default="multi-aspect",
        choices=["multi-aspect", "simple", "none"],
    )
    parser.add_argument(
        "--rerank", type=str, default="hybrid", choices=["max", "equal", "hybrid"]
    )
    parser.add_argument(
        "--index_combine_method",
        type=str,
        choices=["weighted", "merge"],
        default="weighted",
    )
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument(
        "--select_query", type=str, default="uct", choices=["none", "random", "uct"]
    )
    return parser.parse_args()


def get_sample_values(
    checked_obj_dict: dict[str, int], args, attribute: str
) -> list[str]:
    """
    Get the sample values based on the checked objects
    """
    if len(checked_obj_dict) == 0:
        temp_data = json.load(open(f"{args.path}/{args.dataset}_sample_values.json"))
        samples = temp_data[attribute]
        non_linguistic_samples = temp_data.get(f"{attribute}_non-linguistic", None)
    else:
        pos_objs = [k for k, v in checked_obj_dict.items() if v == 1]
        neg_objs = [k for k, v in checked_obj_dict.items() if v == 0]
        if len(neg_objs) > 5:
            neg_objs = np.random.choice(neg_objs, 5, replace=False)
        samples = list(pos_objs) + list(neg_objs)
        non_linguistic_samples = None
    return samples, non_linguistic_samples


def retrieve_and_rerank(
    query: Query,
    bm25_queries: list[str],
    hnsw_queries: list[str],
    corpus: list,
    args: dict,
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
    check_num: int,
):
    retrieved_info = retrieve_corpus(
        bm25_queries, hnsw_queries, corpus, args, bm25_index, hnsw_index
    )
    obj_to_check, stop_flag = rerank_retrieved_objs(
        query, retrieved_info, args, check_num
    )
    return retrieved_info, obj_to_check, stop_flag


def solve_query(
    query: Query,
    attribute: str,
    corpus: list,
    args,
    answers: list,
    bm25_index: BM25Index,
    hnsw_index: HNSWIndex,
) -> dict:
    answers = [a.lower() for a in answers]
    # Step 1: Initialization
    reformulate_time, refine_time, retrieve_time, check_time = 0, 0, 0, 0
    rerank_time = 0
    checked_obj_dict = {}
    print(f"\n\n\n======Processing Query: [{query.org_query}]=======")
    # Initial Retrieval
    start_time = time.time()
    retrieved_info, obj_to_check, stop_flag = retrieve_and_rerank(
        query,
        [query.org_query],
        [query.org_query],
        corpus,
        args,
        bm25_index,
        hnsw_index,
        args.budget,
    )
    retrieve_time += time.time() - start_time
    hnsw_max_scalar = retrieved_info.hnsw_max_scalar
    start_time = time.time()
    obj_scores = llm_check_retrieved_objs(query, obj_to_check, args)
    check_time += time.time() - start_time
    query.update_obj_scores(obj_scores, corpus)
    query.update_obj_features(retrieved_info)
    query_from_table = [q for q, s in obj_scores.items() if s > 0]
    query.update_queries_from_table(query_from_table)
    score_dict = score_query(query.query_condition, query_from_table)
    accumulate_precision, accumulate_rel, accumulate_hit = 0, 0, 0
    precision_list = []
    pos_objs = []
    rank_list = []
    for i, obj in enumerate(obj_to_check):
        rel = 1 if obj_scores.get(obj, 0) + score_dict.get(obj, 0) >= 2 else 0
        accumulate_hit += rel
        accumulate_precision += (accumulate_hit / (i + 1)) * rel
        accumulate_rel += rel
        if rel > 0:
            precision_list.append(f"{accumulate_hit / (i + 1):.4f}")
            pos_objs.append(obj)
            rank_list.append(i + 1)
        # print(i, obj, obj_scores[obj])
    ap = accumulate_precision / accumulate_rel if accumulate_rel > 0 else 0
    print(f"Query: {query.org_query}, AP: {ap:.4f}")
    # print(f"Precision list: {precision_list}")
    # print(f"Rank list: {rank_list}")
    print(f"pos objs: {pos_objs}")
    # print(f"answers: {answers}")
    recall = len(set(obj_to_check) & set(answers)) / len(answers)
    print(f"Recall: {recall * 100:.4f}")
    print(f"Hitted objs: {set(obj_to_check) & set(answers)}")
    print(f"Missed objs: {set(answers) - set(obj_to_check)}")
    # return None

    # Step 2: determine the type of semantic gap
    start_time = time.time()
    gap_list, format_response = identify_semantic_gap(query.org_query, obj_scores)
    print(f"gap_list: {gap_list}")
    print(f"Time for identifying semantic gap: {time.time() - start_time:.4f}s")
    # if (ap <= 0.1 or hnsw_max_scalar <= 0.8) and len(gap_list) == 0:
    #     gap_list = ["concept", "synonym"]
    #     print(f"Force gap_list: {gap_list}")
    if hnsw_max_scalar <= 0.8:
        gap_list = ["concept", "synonym", "format"]
    else:
        gap_list = []

    # Step 2: do the reformulation based on the gap list
    sample_values, _ = get_sample_values({}, args, attribute)
    gap_reformulated_terms = {}
    if len(gap_list) > 0:
        gap_reformulated_terms = reformulate(
            gap_list,
            query.org_query,
            attribute,
            obj_scores,
            format_response,
            sample_values,
        )
        print(gap_reformulated_terms)
    reformulate_time += time.time() - start_time
    print(f"Time for reformulating: {time.time() - start_time:.4f}s")

    cur_generated_query_list = gap_reformulated_terms.get(
        "concept", []
    ) + gap_reformulated_terms.get("synonym", [])
    if "format" in gap_reformulated_terms:
        for pattern in gap_reformulated_terms["format"]:
            new_query_list = [
                v
                for v in corpus
                if re.search(pattern, v) or re.search(pattern, v.replace(" ", ""))
            ]
            if len(new_query_list) > 10:
                # only keep the values that are identified as the same as the query term
                temp_obj_scores = llm_check_retrieved_objs(query, new_query_list, args)
                new_query_list = [v for v, s in temp_obj_scores.items() if s > 0]
            cur_generated_query_list += new_query_list
    cur_generated_query_list = list(set(cur_generated_query_list))
    print(
        f"Generated query list ({len(cur_generated_query_list)}): {cur_generated_query_list}"
    )

    cur_generated_query_list = [
        q for q in cur_generated_query_list if q != query.org_query
    ]
    query.update_queries_from_generated(cur_generated_query_list)
    query_scores_new = {q: 2 for q in cur_generated_query_list}
    query.update_query_scores(query_scores_new)

    # Step 3: Iteratively refine the query and retrieve the cell values
    early_stop = 0
    step = 0
    if args.iterative_check:
        check_num = max(args.budget // 5, 50)
    else:
        check_num = args.budget
    retrieved_info = None
    while len(query.obj_scores) < args.budget:
        step += 1
        print(f"\n======Step {step}=======")
        check_num = min(check_num, args.budget - len(query.obj_scores))

        # step 3.1: score and select the diversified query words from the query list
        start_time = time.time()
        new_query_objs = query.new_queries_from_generated + query.new_queries_from_table
        # TODO: simplify query term selection process
        # hnsw: diversified query terms
        # bm25: only if when the format gap is detected; otherwise, only use hnsw
        hnsw_queries = query.select_hnsw_queries_old(
            retrieved_info, args.select_query, new_query_objs
        )
        bm25_queries = query.select_bm25_queries_old(
            retrieved_info, args.select_query, new_query_objs
        )
        print(f"Step {step} HNSW query list ({len(hnsw_queries)}): {hnsw_queries}")
        print(f"Step {step} BM25 query list ({len(bm25_queries)}): {bm25_queries}")
        # print(f"Time for refining: {time.time() - start_time:.4f}s")

        # step 3.2: retrieve the corpus based on the query list
        start_time = time.time()
        retrieved_info, obj_to_check, stop_flag = retrieve_and_rerank(
            query,
            bm25_queries,
            hnsw_queries,
            corpus,
            args,
            bm25_index,
            hnsw_index,
            check_num,
        )
        retrieve_time += time.time() - start_time
        # print(f"obj_to_check ({len(obj_to_check)}): {obj_to_check}")

        # step 3.4: llm check the retrieved objs
        start_time = time.time()
        obj_scores = llm_check_retrieved_objs(query, obj_to_check, args)
        # query.update_query_scores(query_scores)
        query_from_table = [q for q, s in obj_scores.items() if s > 0]
        query.update_queries_from_table(query_from_table)
        query.update_obj_scores(obj_scores, corpus)
        query.update_obj_features(retrieved_info)
        query.update_query_scores(obj_scores)
        check_time += time.time() - start_time
        print(f"check {len(obj_scores)} objs")
        # print(f"query_scores: {query_scores}")
        print(f"Time for checking: {time.time() - start_time:.4f}s")

        # if no new positive objs are found, stop
        if (
            args.early_stop
            and sum([query.obj_scores.get(obj, 0) for obj in obj_scores]) == 0
            and (
                sum([query.obj_scores.get(obj, 0) for obj in query.obj_scores]) > 0
                or len(obj_to_check) == 0
            )
        ):
            early_stop += 1
            if early_stop >= 2:
                score_dict = score_query(
                    query.query_condition, query.queries_from_table
                )
                pred = [q for q, s in score_dict.items() if s > 0]
                query.update_query_scores(score_dict)
                query.update_obj_scores(score_dict, corpus)
                if len(pred) > 0:
                    print(f"Early stop at step {step}")
                    break
                else:
                    early_stop = 0
        else:
            early_stop = 0

    retrieved = list(query.obj_scores.keys())
    retrieval_recall = len(set(retrieved) & set(answers)) / len(answers)
    # recheck
    start_time = time.time()
    score_dict = score_query(query.query_condition, query.queries_from_table)
    pred = [q for q, s in score_dict.items() if s > 0]
    if len(pred) == 0:
        pred = query.queries_from_table
    recall = len(set(pred) & set(answers)) / len(answers)
    precision = len(set(pred) & set(answers)) / len(pred) if len(pred) > 0 else 0
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    check_time += time.time() - start_time
    print(f"Retrieval Recall: {retrieval_recall:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {
        "query": query.org_query,
        "pred": pred,
        "retrieval_recall": retrieval_recall,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "retrieved": retrieved,
        "retrieved_num": len(query.obj_scores),
        "score_dict": score_dict,
        "reformulate_time": reformulate_time,
        "refine_time": refine_time,
        "retrieve_time": retrieve_time,
        "check_time": check_time,
        # "iteration_num": step + 1,
    }


def save_results(results, output_path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.output_dir = f"results/{args.exp_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.budget}.json")
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
    # TODO: rename the variables
    if args.dataset == "product":
        batch_size = 512
        attribute = "Product_Title"
        # reformat_template = "According to the product title, the product is the same as or a type of '{query}'."
    else:
        cols = df.columns
        batch_size = 128
        attribute = cols[0]
        # reformat_template = (
        #     f"According to the {attribute} name, the {attribute}"
        #     + " is the same as or a type of '{query}'."
        # )
        # reformat_template = "The value is the same as or a type of '{query}'."
        # llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes' or 'No'."
    reformat_template = "The value is the same as or a type of '{query}'."
    llm_template = "Is '{value}' the same as or a type of '{query}'? Directly answer with 'Yes', 'No', or 'Unsure'."
    corpus = df[attribute].values.tolist()

    args.llm_template = llm_template

    start_time = time.time()
    bm25_index = BM25Index(corpus, args.dataset)
    print(f"Time for loading BM25 index: {time.time() - start_time:.4f}s")
    load_index_time = time.time() - start_time
    start_time = time.time()
    hnsw_index = HNSWIndex(corpus, args.dataset)
    print(f"Time for loading HNSW index: {time.time() - start_time:.4f}s")
    load_index_time += time.time() - start_time

    # solve query
    print(
        f"Total queries: {len(query_answer)}, processed queries: {len(processed_queries)}"
    )
    cnt = 0
    for query, answers in query_answer.items():
        # if query not in filtered_queries:
        #     continue
        if query in processed_queries:
            continue
        if len(answers) == 0:
            continue
        start_time = time.time()
        query = Query(query, reformat_template)
        result = solve_query(
            query, attribute, corpus, args, answers, bm25_index, hnsw_index
        )
        result["answers"] = answers
        result["time"] = time.time() - start_time
        results.append(result)
        save_results(results, output_path)
        cnt += 1
        if args.exp_name == "debug" and cnt >= 10:
            break
    # save results
    save_results(results, output_path)
    retrieval_recall_list = []
    pre_list = []
    rec_list = []
    time_list = []
    reformulate_time_list = []
    check_time_list = []
    for result in results:
        retrieval_recall_list.append(result["retrieval_recall"])
        pre_list.append(result["precision"])
        rec_list.append(result["recall"])
        time_list.append(result["time"])
        reformulate_time_list.append(result["reformulate_time"])
        check_time_list.append(result["check_time"])

    print(f"\n\nAverage retrieval recall: {np.mean(retrieval_recall_list):.4f}")
    avg_pre = np.mean(pre_list)
    avg_rec = np.mean(rec_list)
    avg_f1 = 2 * avg_pre * avg_rec / max(avg_pre + avg_rec, 1e-6)
    print(f"Average Precision: {avg_pre:.4f}")
    print(f"Average Recall: {avg_rec:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    time_list = sorted(time_list)[1:-1]
    print(f"Average Time: {np.median(time_list):.4f}s")
    print(f"Average Reformulate Time: {np.mean(reformulate_time_list):.4f}s")
    print(f"Average Check Time: {np.mean(check_time_list):.4f}s")

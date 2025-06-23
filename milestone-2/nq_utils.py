import json
import random
from collections import defaultdict

def show_sample_nq(
    num_examples:int=3,
    corpus_path:str="./nq/corpus.jsonl",
    queries_path:str="./nq/queries.jsonl",
    qrels_path:str="./nq/qrels/test.tsv"
):
    """
    Load NQ data and print a given number of random queries with their relevant documents.

    Parameters:
    - num_examples: int, number of query examples to display
    - corpus_path: str, path to the NQ corpus JSONL file
    - queries_path: str, path to the NQ queries JSONL file
    - qrels_path: str, path to the NQ relevance judgments TSV file
    """
    # Load corpus into a dict
    corpus = {}
    with open(corpus_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            corpus[entry["_id"]] = entry

    # Load queries
    with open(queries_path, "r") as f:
        queries = [json.loads(line) for line in f]

    # Load qrels (keep only score == 1)
    qrels = defaultdict(list)
    with open(qrels_path, "r") as f:
        next(f)  # skip header
        for line in f:
            qid, docid, score = line.strip().split("\t")
            if int(score) == 1:
                qrels[qid].append(docid)

    # Filter queries that have relevant docs
    valid_queries = [q for q in queries if q["_id"] in qrels]
    if not valid_queries:
        print("No queries with relevant documents found.")
        return

    # Sample queries
    count = min(num_examples, len(valid_queries))
    sampled = random.sample(valid_queries, count)

    # Display each sample
    for idx, query in enumerate(sampled, 1):
        print(f"Example {idx} - Query [{query['_id']}]: {query['text']}\n")
        print("Relevant Document(s):")
        for docid in qrels[query["_id"]]:
            doc = corpus.get(docid)
            if doc:
                snippet = doc["text"][:200].replace("\n", " ")
                print(f" - [{docid}] {doc['title']}: {snippet}...")

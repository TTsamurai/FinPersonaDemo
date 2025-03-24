import os
import json

# Load model and tokenizer from HuggingFace
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

# from pyserini.search.lucene import LuceneSearcher
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker

if not pt.started():
    pt.init()
import ipdb


def extract_context(json_data, number, turn_id):
    # Find the correct dictionary with the given number
    data = None
    for item in json_data:
        if item["number"] == number:
            data = item
            break

    # If we couldn't find the data for the given number
    if not data:
        print("No data found for the given number.")
        return "No data found for the given number.", None

    # Extract the utterance and response values
    texts = []
    current_utterance = ""
    for turn in data["turns"]:
        if turn["turn_id"] < turn_id:
            texts.append(turn["utterance"])
            texts.append(turn["response"])
        elif turn["turn_id"] == turn_id:
            current_utterance = turn["utterance"]
            texts.append(current_utterance)

    # Join the texts with "|||" separator
    context = "|||".join(texts)

    return current_utterance, context


def escape_special_characters(query):
    # Escaping special characters
    special_chars = ["?", "&", "|", "!", "{", "}", "[", "]", "^", "~", "*", ":", '"', "+", "-", "(", ")"]
    for char in special_chars:
        query = query.replace(char, "")
    return query


def str_to_df_query(query):
    if isinstance(query, str):
        query = escape_special_characters(query)
        return pd.DataFrame([[1, query]], columns=["qid", "query"])
    elif isinstance(query, list):
        query = [escape_special_characters(q) for q in query]
        return pd.DataFrame([[i + 1, q] for i, q in enumerate(query)], columns=["qid", "query"])
    else:
        raise ValueError("The query must be a string or a list of strings.")


def retrieve_and_rerank(query, pipeline):
    query_df = str_to_df_query(query)
    res = pipeline.transform(query_df)
    candidate_set = []
    for i, row in res.iterrows():
        passage_id = row["docno"]
        rank = row["rank"]
        score = row["score"]
        passage_text = row["text"]
        candidate_set.append({"passage_id": passage_id, "rank": i + 1, "score": score, "passage_text": passage_text})
    return candidate_set


def rerank_passages(query, passages, reranker):
    res = []
    query_passage_pairs = [[query, passage["passage_text"]] for passage in passages]
    scores = reranker.predict(query_passage_pairs)

    for passage, score in zip(passages, scores):
        passage["reranker_score"] = score
        res.append(passage)

    ranked_passages = sorted(passages, key=lambda x: x["reranker_score"], reverse=True)
    return ranked_passages


def rag(rewrite, top_n_passages=3):
    # Set up
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Set Up Index
    index_path = os.path.join("/root/nfs/iKAT/2023/ikat_index/index_pyterrier_with_text", "data.properties")
    index = pt.IndexFactory.of(index_path)
    # Set up Pipeline for retrieval and reranking
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
    monoT5 = MonoT5ReRanker()
    pipeline = (bm25 % 10) >> pt.text.get_text(index, "text") >> (monoT5 % 5) >> pt.text.get_text(index, "text")
    # Passage retrieval and reranking
    reranked_passages = retrieve_and_rerank(rewrite, pipeline)
    passages = [{"passage_id": passage["passage_id"], "passage_text": passage["passage_text"]} for passage in reranked_passages][:top_n_passages]
    return passages


def retrieve_passage(resolved_query, history, RAG, top_n_passages=3):
    # TODO: RAG function
    if RAG:
        if len(history) >= 1:
            rag_context = rag(resolved_query, top_n_passages)
        else:
            rag_context = rag(
                resolved_query,
            )
    else:
        rag_context = "No Context"
    return rag_context


def get_length_without_special_tokens(text, tokenizer):
    # Tokenize the prompt and get input IDs
    inputs = tokenizer(text, return_tensors="pt")
    # Extract the input IDs from the tokenized output
    input_ids = inputs.input_ids[0]
    # Decode the input IDs to a string, skipping special tokens
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    return len(decoded_text)


def response_generation(messages, model, tokenizer, device, terminators, max_tokens=512, temperature=0.0, top_p=0.9):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=False,  # Greedy_decoding to be deterministic
        # temperature=temperature,
        top_p=top_p,
    )

    prompt_length = get_length_without_special_tokens(prompt, tokenizer)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_length:]
    # ipdb.set_trace()
    return response.strip(), messages + [{"role": "assistant", "content": response.strip()}]


if __name__ == "__main__":
    # Set up
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo_path = "/nfs/primary/iKAT/2023/"
    with open(os.path.join(demo_path, "ikat_demo/test.json"), "r") as f:
        topics = json.load(f)

    # Set up Index
    index_path = os.path.join("/root/nfs/iKAT/2023/index_pyterrier_with_text", "data.properties")
    index = pt.IndexFactory.of(index_path)

    # Set up Pipeline for retrieval and reranking
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"])
    monoT5 = MonoT5ReRanker()
    pipeline = (bm25 % 10) >> pt.text.get_text(index, "text") >> (monoT5 % 5) >> pt.text.get_text(index, "text")

    query = "Can you compare mozzarella with plant-based cheese?"

    # Query rewriting
    rewriter = AutoModelForSeq2SeqLM.from_pretrained("castorini/t5-base-canard").to(device).eval()
    rewriter_tokenizer = AutoTokenizer.from_pretrained("castorini/t5-base-canard")
    number_to_search = "10-1"
    turn_id_to_search = 6
    utterance, context = extract_context(topics, number_to_search, turn_id_to_search)
    rewrite = rewrite_query(context, rewriter, rewriter_tokenizer, device)

    # Passage Retrieval and Reranking
    reranked_passages = retrieve_and_rerank(rewrite, pipeline)

    # Response generation
    summarizer = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    summarizer_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    # We use the top-3 reranked passages to generate a response
    passages = [passage["passage_text"] for passage in reranked_passages][:3]
    print(json.dumps(passages, indent=4))
    responses = generate_response(passages, summarizer, summarizer_tokenizer)
    print("Done")

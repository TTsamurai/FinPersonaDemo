from components.rag_components import get_length_without_special_tokens

REWRITE_PASSAGE_PROMPT = """
A passage has been retrieved from the web based on the query: {}. Please extract only the information that is essential for answering this query with at most two or three sentences. If the passage contains no relevant information, do not extract anything. Provide the extracted information directly without any introductory phrases or additional context.
Query: {}\n
Passage: {}\n
"""


def rewrite_rag_context(resolved_query, rag_context, model, tokenizer, terminator):
    """
    Rewrites the passages in the RAG context based on the resolved query.

    Args:
        resolved_query (str): The resolved user query.
        rag_context (list): A list of dictionaries, each containing 'passage_id' and 'passage_text'.
        model: The model used for generating rewritten passages.
        tokenizer: The tokenizer used for processing text.
        terminator: The terminator token for the model.

    Returns:
        list: A list of dictionaries with rewritten passages.
    """
    retrieved_passages = []
    for passage in rag_context:
        rewrite = rewrite_passage(resolved_query, passage["passage_text"], model, tokenizer, terminator)
        retrieved_passages.append({"passage_id": passage["passage_id"], "passage_text": rewrite})
    return retrieved_passages


def rewrite_passage(resolved_query, passage, model, tokenizer, terminator, max_tokens=256, temperature=0.0, top_p=0.9):
    """
    Rewrites a single passage based on the resolved query.

    Args:
        resolved_query (str): The resolved user query.
        passage (str): The passage text to be rewritten.
        model: The model used for generating rewritten passages.
        tokenizer: The tokenizer used for processing text.
        terminator: The terminator token for the model.
        max_tokens (int): The maximum number of tokens to generate. Default is 256.
        temperature (float): The temperature for sampling. Default is 0.6.
        top_p (float): The nucleus sampling probability. Default is 0.9.

    Returns:
        str: The rewritten passage.
    """
    chatbot = []
    user_prompt = REWRITE_PASSAGE_PROMPT.format(resolved_query, passage, passage)
    chatbot.append({"role": "user", "content": user_prompt})
    prompt = tokenizer.apply_chat_template(chatbot, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=terminator,
        do_sample=False,  # Greedy decoding to be deterministic
        # temperature=temperature
        top_p=top_p,
    )

    prompt_length = get_length_without_special_tokens(prompt, tokenizer)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_length:]
    return response.strip()


# def rewrite_rag_context(resoloved_query, rag_context, model, tokenizer, terminator):
#     """
#     rag_context: [{"passage_id": passage["passage_id"], "passage_text": passage['passage_text']} for passage in reranked_passages]
#     """
#     retrieved_passages = []
#     for passage in rag_context:
#         rewrite = rewrite_passage(resoloved_query, passage["passage_text"], model, tokenizer, terminator)
#         retrieved_passages.append([{"passage_id": passage["passage_id"], "passage_text":rewrite}])
#     return retrieved_passages

# def rewrite_passage(resoloved_query, passage, model, tokenizer, terminator, max_tokens=256, temperature=0.6, top_p=0.9):
#     chatbot = []
#     user_prompt = REWRITE_PASSAGE_PROMPT.format(resoloved_query, passage, passage)
#     chatbot.append({"role": "user", "content": message})
#     prompt = tokenizer.apply_chat_template(chatbot, tokenize=False, add_generation_prompt=True)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=temperature,
#         top_p=top_p,
#     )

#     prompt_length = get_length_without_special_tokens(prompt, tokenizer)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_length:]
#     return response.strip()

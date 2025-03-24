from components.rag_components import get_length_without_special_tokens
import ipdb

QUERY_REWRITING = """Given a user query and its context (conversational history), decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context. JUST ANSWER THE RESOLVED QUERY WITHOUT ANY OTHER SENTENCES.\nContext: {}\n"""
REMINDER = """\nRemember you are a query rewriter. JUST ANSWER THE RESOLVED QUERY WITHOUT ANY OTHER SENTENCES."""


def get_context_from_message_history(message_history):
    context = ""
    for message in message_history:
        if message["role"] not in ["system"]:
            context += f'{message["role"]}: {message["content"]}\n'
    return context if context else "No context available."


def rewrite_query(query: str, history: str, rewriter, rewriter_tokenizer, rewriter_terminator, device="cuda", max_tokens=256, temperature=0.0, top_p=0.9) -> str:
    # ipdb.set_trace()
    # DELETE LAST \n<assistant>\n
    history = "\n".join(history.split("\n")[:-1])
    system_prompt = QUERY_REWRITING.format(history)

    query += REMINDER
    user_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"user query: {query}"}]
    prompt = rewriter_tokenizer.apply_chat_template(user_prompt, tokenize=False, add_generation_prompt=True)
    print("user_prompt:", user_prompt)
    print("PROMPT:", prompt)
    # ipdb.set_trace()
    print("System Prompt:", system_prompt)
    print("Prompt:", prompt)

    inputs = rewriter_tokenizer(prompt, return_tensors="pt").to(rewriter.device)
    outputs = rewriter.generate(
        **inputs,
        max_new_tokens=max_tokens,
        eos_token_id=rewriter_terminator,
        do_sample=False,  # Greedy decoding to be deterministic
        # temperature=temperature,
        top_p=top_p,
    )
    prompt_length = get_length_without_special_tokens(prompt, rewriter_tokenizer)
    response = rewriter_tokenizer.decode(outputs[0], skip_special_tokens=True)[prompt_length:]
    return response.strip()


# def rewrite_query(query: str, history: str, rewriter, rewriter_tokenizer, device="cuda") -> str:
#     context = "|||".join([history, query])
#     # rewriter = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
#     # rewriter_tokenizer = AutoTokenizer.from_pretrained(model_name)
#     tokenized_context = rewriter_tokenizer.encode(context, return_tensors="pt").to(device)
#     output_ids = rewriter.generate(
#       tokenized_context,
#       max_length=200,
#       num_beams=4,
#       repetition_penalty=2.5,
#       length_penalty=1.0,
#       early_stopping=True
#     ).to(device)

#     rewrite = rewriter_tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     return rewrite

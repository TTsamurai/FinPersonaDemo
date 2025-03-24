import os
import ipdb
import itertools
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

from utils import login_to_huggingface, ACCESS
from components.rag_components import rag, retrieve_passage, response_generation
from components.rewrite_passages import rewrite_rag_context
from components.query_rewriting import rewrite_query
from components.chat_conversation import format_message_history, format_user_message, format_context, gradio_to_huggingface_message, huggingface_to_gradio_message, get_system_instruction, prepare_tokenizer, format_rag_context
from components.constant import ACCESS, QUERY_REWRITING, RAG, DEVICE, RESPONSE_GENERATOR, NUM_PASSAGES
from components.prompt import SYSTEM_INSTRUCTION, RAG_INSTRUCTION, PERSONALITY_INSTRUCTION
from components.induce_personality import construct_big_five_words


def get_conversation_hitory(persona_type, user_predefined_message, tokenizer, model, terminator):
    # Output: conversation history {"role": "user", "content": "message"}
    assert len(user_predefined_message) >= 1, "User message should be at least one"
    system_instruction = get_system_instruction(rag=RAG, personality_list=persona_type)
    messages = [{"role": "system", "content": system_instruction}]
    for user_message in user_predefined_message:
        if QUERY_REWRITING:
            str_history = format_message_history(user_message, messages)
            resolved_query = rewrite_query(user_message, str_history, model, tokenizer, terminator, device=DEVICE)
        else:
            resolved_query = user_message
        messages = format_user_message(resolved_query, messages)
        # TODO implement rag function as this will be important later
        _, messages = response_generation(messages, model, tokenizer, device=DEVICE, terminators=terminator)
    return messages


def store_conversation_to_text(filename, conversation):
    with open(filename, "w") as file:
        for turn in conversation:
            file.write(f"{turn['role']}: {turn['content']}\n")
        file.write("\n")  # Add a newline at the end of the conversation


if __name__ == "__main__":
    output_par_dir = "./output/personality_output"
    personality_types = [["extroverted", "introverted"], ["agreeable", "antagonistic"], ["conscientious", "unconscientious"], ["neurotic", "emotionally stable"], ["open to experience", "closed to experience"]]
    # load case
    with open("user_predefined_queries.json", "r") as file:
        user_q = json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(RESPONSE_GENERATOR)
    tokenizer, terminator = prepare_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(RESPONSE_GENERATOR, torch_dtype=torch.float16, pad_token_id=tokenizer.eos_token_id).to(DEVICE)
    for case_name, user_predefined_message in user_q.items():
        for persona_type in tqdm(itertools.product(*personality_types)):
            conv_hist = get_conversation_hitory(persona_type, user_predefined_message, tokenizer, model, terminator)
            save_file_name = "_".join(persona_type) + ".txt"
            output_dir = os.path.join(output_par_dir, case_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            save_file_path = os.path.join(output_dir, save_file_name)
            store_conversation_to_text(save_file_path, conv_hist)

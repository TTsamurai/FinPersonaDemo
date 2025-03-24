from components.induce_personality import construct_big_five_words
from components.constant import (
    ACCESS,
    QUERY_REWRITING,
    RAG,
    PERSONALITY,
    PERSONALITY_LIST,
    REWRITE_PASSAGES,
    NUM_PASSAGES,
    DEVICE,
    RESPONSE_GENERATOR,
    TEMPLATE_PAYLOAD,
)
from components.prompt import SYSTEM_INSTRUCTION, RAG_INSTRUCTION, PERSONALITY_INSTRUCTION
import requests
import together


def generate_response_debugging(history):
    # outputs_text = "This is a test response"
    outputs_text = " ".join([item["content"] for item in history])
    history = history + [{"role": "assistant", "content": outputs_text}]
    return outputs_text, history


# REWRITER = "castorini/t5-base-canard"
def generate_response_together_api(history, max_tokens, client, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"):
    together_request = {
        "model": model,
        "messages": history,
        "stream": False,
        "logprobs": False,
        "stop": ["<eos>", "<unk>", "<sep>", "<pad>", "<cls>", "<mask>"],
        "max_tokens": max_tokens,
    }
    response = client.chat.completions.create(**together_request)
    outputs_text = response.choices[0].message.content
    history = history + [{"role": "assistant", "content": outputs_text}]
    return outputs_text, history


def make_local_api_call(payload, api_url):
    try:
        # Send the POST request to the API
        response = requests.post(api_url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            # Print the generated text
            return result.get("text", [""])[0]
            # if "logits" in result:
            #     print(f"Logits: {result['logits']}")
        else:
            # If there was an error, print the status code and message
            print(f"Error: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


def generate_response_local_api(history, terminator, max_tokens, api_url):
    payload = TEMPLATE_PAYLOAD.copy()
    payload.update(
        {
            "prompt": history,
            "max_tokens": max_tokens,
            "stop_token_ids": terminator,
        }
    )
    # Call the API to generate the response
    outputs_text = make_local_api_call(payload, api_url)

    if outputs_text:
        # Update history with the assistant's response
        history = history + [{"role": "assistant", "content": outputs_text}]
        return outputs_text, history
    else:
        print("Failed to generate a response.")
        return "Generation failed", history  # Return the original history in case of failure


def conversation_window(history, N=100):
    if len(history) > N:
        return history[2:]
    return history


def format_message_history(message, history):
    if not history:
        str_history = f"\n<user>: {message}\n<assistant>"
    else:
        # Query written
        str_history = (
            "".join(["".join(["\n<user>:" + item[0], "\n<assistant>:" + item[1]]) for item in history])
            + f"\n<user>: {message}\n<assistant>"
        )
    return str_history


def format_user_message(message, history):
    return history + [{"role": "user", "content": message}]


def format_context(message, history):
    return [{"role": "system", "content": message}] + history


def prepare_tokenizer(tokenizer):
    special_tokens = ["<eos>", "<unk>", "<sep>", "<pad>", "<cls>", "<mask>"]
    for token in special_tokens:
        if tokenizer.convert_tokens_to_ids(token) is None:
            tokenizer.add_tokens([token])

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")
    terminators = [
        tokenizer.eos_token_id,
        # self.pipeline.tokenizer.convert_tokens_to_ids(""),
    ]
    return tokenizer, terminators


def gradio_to_huggingface_message(gradio_message):
    huggingface_message = []
    for user, bot in gradio_message:
        huggingface_message.append({"role": "user", "content": user})
        huggingface_message.append({"role": "assistant", "content": bot})
    return huggingface_message


def huggingface_to_gradio_message(huggingface_message):
    gradio_message = []
    store = []
    for utter in huggingface_message:
        if utter["role"] in ["user", "assistant"]:
            if utter["role"] == "assistant":
                store.append(utter["content"])
                gradio_message.append(store)
                store = []
            else:
                store.append(utter["content"])
    return gradio_message


def get_personality_instruction(personality):
    return PERSONALITY_INSTRUCTION.format(personality)


def get_system_instruction(rag=RAG, personality_list=None):
    if rag and personality_list:
        return (
            SYSTEM_INSTRUCTION
            + RAG_INSTRUCTION
            + get_personality_instruction(construct_big_five_words(personality_list))
        )
    elif personality_list:
        return SYSTEM_INSTRUCTION + get_personality_instruction(construct_big_five_words(personality_list))
    elif rag:
        return SYSTEM_INSTRUCTION + RAG_INSTRUCTION
    else:
        return SYSTEM_INSTRUCTION


def format_rag_context(rag_context):
    """
    rag_context [{"passage_id": clue_web, "passage_text": "abc"}, ...]
    """
    passage_context = "Context: \n"
    for passage_rank, info in enumerate(rag_context):
        passage_context += f"Passage ID: {info['passage_id']}, Text: {info['passage_text']}\n\n"
    return passage_context

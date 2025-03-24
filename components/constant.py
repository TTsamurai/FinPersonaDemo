# Configuration Constants
import os

ACCESS = os.getenv("HF_ACCESS_TOKEN")
QUERY_REWRITING = False
RAG = False
PERSONALITY = True
PERSONALITY_LIST = ["introverted", "antagonistic", "conscientious", "emotionally stable", "open to experience"]
REWRITE_PASSAGES = False
NUM_PASSAGES = 3
DEVICE = "cuda"
RESPONSE_GENERATOR = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CONV_WINDOW = 100
API_URL = "http://10.249.1.2:8888/generate"
TEMPLATE_PAYLOAD = {
    "stream": False,  # Set to True if you want to stream the results
    "logprobs": False,  # Set to True if you want the log probabilities of the tokens
    "include_prompt": False,  # Whether to include the original prompt in the response}
}
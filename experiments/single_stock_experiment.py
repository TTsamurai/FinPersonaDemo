import subprocess
import sys
import os
from components.induce_personality import (
    construct_big_five_words,
)


# need to import: gradio
def install(package, upgrade=False):
    if upgrade:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                package,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                package,
            ],
            check=True,
        )


# install("ipdb")
# install("gradio")
# install("sentence-transformers")
# install("git+https://github.com/terrierteam/pyterrier_t5.git")
# install("protobuf")
# install("transformers", upgrade=True)
import random
import json
import gradio as gr
import random
import time
import ipdb
import markdown
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import login_to_huggingface, ACCESS
from components.rag_components import (
    rag,
    retrieve_passage,
    response_generation,
)
from components.rewrite_passages import rewrite_rag_context
from components.query_rewriting import rewrite_query
from components.chat_conversation import (
    format_message_history,
    format_user_message,
    format_context,
    gradio_to_huggingface_message,
    huggingface_to_gradio_message,
    get_system_instruction,
    prepare_tokenizer,
    format_rag_context,
)
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
)
from components.induce_personality import (
    build_personality_prompt,
)

# LOG_FILE = "log_file_bingzhi_information_seeking.txt"
ROOT_FILE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_FILE, "log/single_stock_experiment/othres/")
if os.path.exists(LOG_DIR) is False:
    os.makedirs(LOG_DIR)
STATIC_FILE = os.path.join("_static")

with open(os.path.join(STATIC_FILE, "html/instruction_page.html"), "r") as f:
    INSTRUCTION_PAGE = f.read()
with open(os.path.join(STATIC_FILE, "html/evaluation_instruction.html"), "r") as f:
    EVALUATION_INSTRUCTION = f.read()
with open(os.path.join(STATIC_FILE, "txt/general_instruction_task.txt"), "r") as f:
    GENERAL_INSTRUCTION_TASK = f.read()
with open(os.path.join(STATIC_FILE, "txt/general_instruction_button.txt"), "r") as f:
    GENERAL_INSTRUCTION_BUTTON = f.read()
with open(os.path.join(STATIC_FILE, "txt/system_instruction.txt"), "r") as f:
    SYSTEM_INSTRUCTION = f.read()
FIRST_MESSAGE = "Hey"
INFORMATION_SEEKING = True
DEBUG = False


def get_context(synthetic_data_path):
    # Load data from the synthetic data file
    with open(synthetic_data_path, "r") as f:
        data = [json.loads(line) for line in f]

    return data


def add_ticker_prefix(ticker_list, context_list):
    res = []
    for ticker, context in zip(ticker_list, context_list):
        res.append(f"{ticker}: {context}")
    return res


def build_raw_context_list(context_dict):
    return context_dict["data"]


def build_context(context_dict):
    return [build_context_element(context) for context in context_dict["data"]]


def build_context_element(context):
    # [{topic: ex, data: {}}, {..}, ..]
    # Extract information from the context
    ticker = context["ticker"]
    sector = context["sector"]
    business_summary = context["business_summary"]
    name = context["short_name"]
    stock_price = context["price_data"]
    earning = context["earning_summary"]
    beta = context["beta"]

    # Build the context string
    stock_candidate = f"Stock Candidate: {name}"
    stock_info = f"Stock Information: \nIndustry - {sector}, \nBeta (risk indicator) - {beta}, \nEarning Summary - {earning}\n, 2023 Monthly Stock Price - {stock_price}\n, Business Summary - {business_summary}"

    context_list = [stock_candidate, stock_info]

    # Combine all parts into a single string
    return "\n".join(context_list)


def get_task_instruction_for_user(context):
    ticker_name = context["short_name"]
    user_narrative = context["user_narrative"]
    user_narrative = user_narrative.replace("\n", "<br>")
    html_user_narrative = markdown.markdown(user_narrative)

    general_instruction = f"""<!-- Grouped Container for Task Instruction and Stock Information -->
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px; max-height: 780px; overflow-y: auto; overflow-x: hidden;">
    <!-- Heading -->
    <h2 style="color: #2c3e50; text-align: center; margin-bottom: 20px; font-size: 20px; font-weight: 600;">
        General Instruction
    </h2>
    
    <!-- User Instruction -->
    <p style="text-align: left; font-size: 16px; color: #34495e; margin-bottom: 20px;">
        {GENERAL_INSTRUCTION_TASK}
        {GENERAL_INSTRUCTION_BUTTON}
    </p>
</div>"""
    round_instruction = f"""
<div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px; max-height: 780px; overflow-y: auto; overflow-x: hidden;">
    <!-- Stock Information (Bold label, Normal ticker name) -->
    <h2 style="color: #2c3e50; text-align: center; margin-bottom: 20px; font-size: 20px; font-weight: 600;">
        Round Info
    </h2>
    <div style="text-align: left; font-size: 20px; font-weight: bold; margin-bottom: 20px;">
        Stock
    </div>
    <div style="text-align: left; font-weight: normal; font-size: 16px; margin-bottom: 20px;">
        <span style="font-weight: bold;">
            This Round's Stock:
        </span>
        {ticker_name}
    </div>

    <!-- User Narrative (Bold label, Normal narrative) -->
    <div style="text-align: left; font-size: 20px; font-weight: bold; margin-bottom: 20px;">
        User Narrative
    </div>
    <div style="text-align: left; font-weight: normal; font-size: 16px; margin-bottom: 20px;">
        {html_user_narrative}
    </div>
</div>"""

    return general_instruction, round_instruction


def display_system_instruction_with_html(
    system_instruction,
):
    html_system_instruction = f"""
        <p style="text-align: left; margin-bottom: 10px;">
            {system_instruction}
        </p>
    """
    return html_system_instruction


def create_demo(
    model,
    tokenizer,
    terminator,
    system_description_without_context,
    stock_context_list,
    raw_context_list,
):
    # Store the history here and use this as an input to each tab.
    tab_data = {}
    tab_gradio = {}

    def tab_creation(order):
        comp, contex, general_instruction, round_instruction = get_context(order)
        system_instruction = system_description_without_context + "\n" + contex
        tab_data[comp] = {"history": [], "selection": "", "reason": ""}
        english_order = ["First", "Second", "Third", "Fourth", "Fifth"]
        with gr.Tab(f"{english_order[order]}: {comp}") as tab:
            with gr.Tab("Interaction with a Financial Advisor"):
                gr.HTML(value=general_instruction, label="General Instruction")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            gr.HTML(
                                value=round_instruction,
                                label="Round Instruction",
                            )
                    with gr.Column():
                        with gr.Row():
                            chatbot = gr.Chatbot(height=600)
                        with gr.Row():
                            start_conversation = gr.Button(value="Start Conversation")
                        with gr.Row():
                            msg = gr.Textbox(scale=1, label="Input: User Input")
                        with gr.Row():
                            msg_button = gr.Button(value="Send: User Input", interactive=False)
                            continue_button = gr.Button(value="Continue", interactive=False)
                with gr.Row():
                    clear = gr.ClearButton([msg, chatbot])
                # if DEBUG:
                with gr.Row():
                    display_prompt = gr.HTML(
                        value=display_system_instruction_with_html(system_instruction),
                        label="System Instruction",
                    )
            with gr.Tab("Evaluation"):
                with gr.Row():
                    gr.HTML(value=EVALUATION_INSTRUCTION)
                with gr.Row():
                    dropdown = gr.Dropdown(
                        label="Decision Making",
                        choices=["Purchase", "Not Purchase"],
                        show_label=True,
                    )
                    reason = gr.Textbox(scale=1, label="The reason of your choice")
                with gr.Row():
                    trust = gr.Slider(
                        label="Trust",
                        minimum=1,
                        maximum=100,
                        value=50,
                        info="How much do you trust the financial advisor? Answer from 1 to 100.",
                        step=1,
                    )
                    satisfaction = gr.Slider(
                        label="Satisfaction",
                        minimum=1,
                        maximum=100,
                        value=50,
                        info="How satisfied are you with the financial advisor? Answer from 1 to 100.",
                        step=1,
                    )
                with gr.Row():
                    knowledgeable = gr.Slider(
                        label="Knowledgeable",
                        minimum=1,
                        maximum=100,
                        value=50,
                        info="How knowledgeable do you feel after interacting with the financial advisor? Answer from 1 to 100.",
                        step=1,
                    )
                    helpful = gr.Slider(
                        label="Helpful",
                        minimum=1,
                        maximum=100,
                        value=50,
                        info="How helpful do you find the financial advisor? Answer from 1 to 100.",
                        step=1,
                    )
                evaluation_send_button = gr.Button(value="Send: Evaluation")
            return {
                "comp": comp,
                "system_instruction": system_instruction,
                "start_conversation": start_conversation,
                "msg_button": msg_button,
                "continue_button": continue_button,
                "chatbot": chatbot,
                "msg": msg,
                "dropdown": dropdown,
                "reason": reason,
                "trust": trust,
                "satisfaction": satisfaction,
                "knowledgeable": knowledgeable,
                "helpful": helpful,
                "evaluation_send_button": evaluation_send_button,
            }

    def click_control(tabs):
        (
            comp,
            system_instruction,
            start_conversation,
            msg_button,
            continue_button,
            chatbot,
            msg,
            dropdown,
            reason,
            trust,
            satisfaction,
            knowledgeable,
            helpful,
            evaluation_send_button,
        ) = (
            tabs["comp"],
            tabs["system_instruction"],
            tabs["start_conversation"],
            tabs["msg_button"],
            tabs["continue_button"],
            tabs["chatbot"],
            tabs["msg"],
            tabs["dropdown"],
            tabs["reason"],
            tabs["trust"],
            tabs["satisfaction"],
            tabs["knowledgeable"],
            tabs["helpful"],
            tabs["evaluation_send_button"],
        )
        start_conversation.click(
            lambda history: respond_start_conversation(history, system_instruction, comp),
            [chatbot],
            [chatbot, start_conversation, msg_button, continue_button],
        )
        msg_button.click(
            lambda message, history: respond(message, tab_data[comp]["history"], system_instruction, comp),
            [msg, chatbot],
            [msg, chatbot],
        )
        continue_button.click(
            lambda history: respond_continue(tab_data[comp]["history"], system_instruction, comp),
            [chatbot],
            [chatbot],
        )
        evaluation_send_button.click(
            lambda dropdown, reason, trust, satisfaction, knowledgeable, helpful: respond_evaluation(
                {
                    "selection": dropdown,
                    "reason": reason,
                    "trust": trust,
                    "satisfaction": satisfaction,
                    "knowledgeable": knowledgeable,
                    "helpful": helpful,
                },
                comp,
            ),
            [dropdown, reason, trust, satisfaction, knowledgeable, helpful],
            [dropdown, reason, trust, satisfaction, knowledgeable, helpful],
        )

    def log_action(tab_name, action, details):
        """
        Log actions for each tab (stock).
        """
        log_file = os.path.join(LOG_DIR, f"{tab_name}.txt")
        with open(log_file, "a") as f:
            f.write(f"Action: {action} | Details: {details}\n")

    def respond(message, history, system_instruction, tab_name):
        """
        Return:
        msg
        chat_history
        retrieved_passage
        rewritten_query

        """
        # Formatting Input
        print(f"User Message: {message} in Tab: {tab_name}")
        history = gradio_to_huggingface_message(history)
        history = format_context(system_instruction, history)
        history_with_user_utterance = format_user_message(message, history)

        outputs_text, history = response_generation(
            history_with_user_utterance,
            model,
            tokenizer,
            max_tokens=128,
            device=DEVICE,
            terminators=terminator,
        )
        # Format
        history = huggingface_to_gradio_message(history)
        print(f"Tab: {tab_name}\nHistory: {history}")

        # Log the user message and response
        log_action(tab_name, "User Message", message)
        log_action(tab_name, "Response", outputs_text)
        # Store the updated history for this tab
        tab_data[tab_name]["history"] = history

        return "", history

    def respond_start_conversation(history, system_instruction, tab_name):
        history = gradio_to_huggingface_message(history)
        history = format_context(system_instruction, history)
        first_message = FIRST_MESSAGE
        history_with_user_utterance = format_user_message(first_message, history)

        outputs_text, history = response_generation(
            history_with_user_utterance,
            model,
            tokenizer,
            max_tokens=128,
            device=DEVICE,
            terminators=terminator,
        )
        # Format
        history = huggingface_to_gradio_message(history)
        print(f"Tab: {tab_name}\nHistory: {history}")

        # Log the user message and response
        log_action(tab_name, "User Message", first_message)
        log_action(tab_name, "Response", outputs_text)
        # Store the updated history for this tab
        tab_data[tab_name]["history"] = history

        return (
            history,
            gr.Button(value="Start Conversation", interactive=False),
            gr.Button(value="Send: User Input", interactive=True),
            gr.Button(value="Continue", interactive=True),
        )

    def respond_continue(history, system_instruction, tab_name):
        message = "continue"
        history = gradio_to_huggingface_message(history)
        history = format_context(system_instruction, history)
        history_with_user_utterance = format_user_message(message, history)

        outputs_text, history = response_generation(
            history_with_user_utterance,
            model,
            tokenizer,
            max_tokens=128,
            device=DEVICE,
            terminators=terminator,
        )
        history = huggingface_to_gradio_message(history)
        log_action(tab_name, "Continue", "User continued the conversation")
        log_action(tab_name, "Response", outputs_text)

        # Update history for this tab
        tab_data[tab_name]["history"] = history

        return history

    def respond_evaluation(evals, tab_name):

        # dropdown, readon_button, multi-evaluator
        log_action(tab_name, "Round Evaluation", "Following")
        for key, value in evals.items():
            log_action(tab_name, key, value)
        # Store the reason for this tab
        tab_data[tab_name]["multi_evaluator"] = evals
        return (
            evals["selection"],
            evals["reason"],
            evals["trust"],
            evals["satisfaction"],
            evals["knowledgeable"],
            evals["helpful"],
        )

    def get_context(index):
        comp = raw_context_list[index]["short_name"]
        context = stock_context_list[index]
        general_instruction, round_instruction = get_task_instruction_for_user(raw_context_list[index])
        return comp, context, general_instruction, round_instruction

    with gr.Blocks(title="RAG Chatbot Q&A", theme="Soft") as demo:
        first_comp, first_context, first_general_instruction, first_round_instruction = get_context(0)
        second_comp, second_context, second_general_instruction, second_round_instruction = get_context(1)
        third_comp, third_context, third_general_instruction, third_round_instruction = get_context(2)
        fourth_comp, fourth_context, forth_general_instruction, forth_round_instruction = get_context(3)
        fifth_comp, fifth_context, fifth_general_instruction, fifth_round_instruction = get_context(4)
        first_system_instruction = system_description_without_context + "\n" + first_context
        second_system_instruction = system_description_without_context + "\n" + second_context
        third_system_instruction = system_description_without_context + "\n" + third_context
        fourth_system_instruction = system_description_without_context + "\n" + fourth_context
        fifth_system_instruction = system_description_without_context + "\n" + fifth_context
        # # initialize tab data
        for comp in [first_comp, second_comp, third_comp, fourth_comp, fifth_comp]:
            tab_data[comp] = {"history": [], "selection": "", "reason": ""}

        # EXperiment Instruction
        with gr.Tab("Experiment Instruction") as instruction_tab:
            gr.HTML(value=INSTRUCTION_PAGE, label="Experiment Instruction")
        # Experiment Tag
        first_tab = tab_creation(0)
        click_control(first_tab)
        second_tab = tab_creation(1)
        click_control(second_tab)
        third_tab = tab_creation(2)
        click_control(third_tab)
        fourth_tab = tab_creation(3)
        click_control(fourth_tab)
        fifth_tab = tab_creation(4)
        click_control(fifth_tab)
    return demo


if __name__ == "__main__":
    login_to_huggingface(ACCESS)

    file_path = os.path.join(ROOT_FILE, "./data/single_stock_data/single_stock_demo.jsonl")
    context_info = get_context(file_path)  # str to List of Dict
    # For Demo Usage, just use the first dict
    context_info = context_info[0]
    stock_context_list = build_context(context_info)  # List of str
    raw_context_list = build_raw_context_list(context_info)  # List of str
    # system instruction consist of Task, Personality, and Context
    """
    Personality
    ["extroverted", "introverted"]
    ["agreeable", "antagonistic"]
    ["conscientious", "unconscientious"]
    ["neurotic", "emotionally stable"]
    ["open to experience", "closed to experience"]]
    """

    personality = [
        "extroverted",
        "agreeable",
        "conscientious",
        "emotionally stable",
        "open to experience",
    ]

    personality_prompt = build_personality_prompt(personality)
    system_instruction_without_context = SYSTEM_INSTRUCTION + "\n" + personality_prompt + "\n"
    if DEBUG:
        tokenizer, terminator, model = "", "", ""
    else:
        tokenizer = AutoTokenizer.from_pretrained(RESPONSE_GENERATOR)
        tokenizer, terminator = prepare_tokenizer(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            RESPONSE_GENERATOR,
            torch_dtype=torch.float16,
            pad_token_id=tokenizer.eos_token_id,
        ).to(DEVICE)
    demo = create_demo(
        model, tokenizer, terminator, system_instruction_without_context, stock_context_list, raw_context_list
    )
    demo.launch(share=True)

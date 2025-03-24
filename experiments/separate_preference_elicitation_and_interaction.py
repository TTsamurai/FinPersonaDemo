import subprocess
import sys
import os

ROOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(ROOT_FILE)
from components.induce_personality import construct_big_five_words


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
    conversation_window,
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
    CONV_WINDOW,
)
from components.induce_personality import (
    build_personality_prompt,
)

# LOG_FILE = "log_file_bingzhi_information_seeking.txt"
LOG_DIR = os.path.join(ROOT_FILE, "log/seperate_preference_elicitation/others/")
if os.path.exists(LOG_DIR) is False:
    os.makedirs(LOG_DIR)
STATIC_FILE = os.path.join(ROOT_FILE, "_static")

with open(os.path.join(STATIC_FILE, "html/instruction_page.html"), "r") as f:
    INSTRUCTION_PAGE = f.read()
with open(os.path.join(STATIC_FILE, "html/evaluation_instruction.html"), "r") as f:
    EVALUATION_INSTRUCTION = f.read()
with open(os.path.join(STATIC_FILE, "html/general_instruction.html"), "r") as f:
    GENERAL_INSTRUCTION = f.read()
with open(os.path.join(STATIC_FILE, "html/user_narrative.html"), "r") as f:
    USER_NARRATIVE = f.read()
with open(os.path.join(STATIC_FILE, "html/system_instruction_preference_elicitation.html"), "r") as f:
    PREFERENCE_ELICITATION_TASK = f.read()
with open(os.path.join(STATIC_FILE, "html/final_evaluation.html"), "r") as f:
    FINAL_EVALUATION = f.read()
with open(os.path.join(STATIC_FILE, "txt/system_instruction_with_user_persona.txt"), "r") as f:
    SYSTEM_INSTRUCTION = f.read()
with open(os.path.join(STATIC_FILE, "txt/system_instruction_preference_elicitation.txt"), "r") as f:
    SYSTEM_INSTRUECTION_PREFERENCE_ELICITATION = f.read()
with open(os.path.join(STATIC_FILE, "txt/system_summarization_user_preference_elicitation.txt"), "r") as f:
    SUMMARIZATION_PROMPT = f.read()
FIRST_MESSAGE = "Hey"
INFORMATION_SEEKING = True
USER_PREFERENCE_SUMMARY = True
DEBUG = True
# if DEBUG:
#     CONV_WINDOW = 3


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


def get_user_narrative_html(user_narrative):
    return USER_NARRATIVE.replace("{user_narrative}", user_narrative).replace("\n", "<br>")


def get_task_instruction_for_user(context):
    ticker_name = context["short_name"]
    user_narrative = context["user_narrative"]
    user_narrative = user_narrative.replace("\n", "<br>")
    html_user_narrative = markdown.markdown(user_narrative)
    general_instruction = GENERAL_INSTRUCTION
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


def log_action(tab_name, action, details):
    """
    Log actions for each tab (stock).
    """
    log_file = os.path.join(LOG_DIR, f"{tab_name}.txt")
    with open(log_file, "a") as f:
        f.write(f"Action: {action} | Details: {details}\n")


def add_user_profile_to_system_instruction(
    system_instruction, user_preference_elicitation_data, summary, model, terminator
):
    if summary:
        if user_preference_elicitation_data["summary_history"] == "":
            # Format prompt
            summarization_prompt = SUMMARIZATION_PROMPT + "\nPrevious Conversations: {}".format(
                user_preference_elicitation_data["history"]
            )
            summarization_instruction = [{"role": "system", "content": summarization_prompt}]
            summ, _ = response_generation(
                summarization_instruction,
                model,
                tokenizer,
                max_tokens=512,
                device=DEVICE,
                terminators=terminator,
            )
            user_preference_elicitation_data["summary_history"] = summ
            log_action("Prompt", "Preference Elicitation Summarization", summ)
            print(f"Preference Summary:{summ}")
        system_instruction += f"\nPrevious Conversations with the Customer about the User Profile: {user_preference_elicitation_data['summary_history']}\n"
    else:
        system_instruction += f"\nPrevious Conversations with the Customer about the User Profile: {user_preference_elicitation_data['history']}\n"
    return system_instruction


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
    user_preference_elicitation_data = {"history": "", "summary_history": ""}

    if DEBUG:
        user_preference_elicitation_data[
            "summary_history"
        ] = """Previous Conversations with the Customer about the User Profile: Based on our previous conversation, here's a summary of your investment preferences:

        1. **Preferred Industries:** You're interested in investing in the healthcare sector, without a specific preference for sub-industries such as pharmaceuticals, medical devices, biotechnology, or healthcare services.
        2. **Value vs. Growth Stocks:** You prefer growth stocks, which have the potential for high returns but may be riskier.
        3. **Dividend vs. Non-Dividend Stocks:** You're open to both dividend and non-dividend growth stocks, focusing on reinvesting profits for future growth.
        4. **Cyclical vs. Non-Cyclical Stocks:** You're interested in cyclical stocks, which are sensitive to economic fluctuations and tend to perform well during economic expansions."""

    def tab_creation_exploration_stage(order):
        comp, context, general_instruction, round_instruction = get_context(order)
        system_instruction = system_description_without_context + "\n" + context
        tab_data[comp] = {"history": [], "selection": "", "reason": ""}
        english_order = ["1", "2", "3", "4", "5"]
        # with gr.Tab(f"{english_order[order]}: {comp}") as tab:
        with gr.Tab(f"{english_order[order]}-1:Discuss"):
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
                        msg = gr.Textbox(scale=1, label="User Input")
                    with gr.Row():
                        msg_button = gr.Button(value="Send This Message to Advisor", interactive=False)
                        continue_button = gr.Button(value="Show More of the Advisor’s Answer", interactive=False)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot])
            if DEBUG:
                with gr.Row():
                    display_prompt = gr.HTML(
                        value=display_system_instruction_with_html(system_instruction),
                        label="System Instruction",
                    )
        with gr.Tab(f"{english_order[order]}-2:Eval"):
            with gr.Row():
                gr.HTML(value=EVALUATION_INSTRUCTION)
            with gr.Row():
                dropdown = gr.Dropdown(
                    label="Would you like to purchase the stock?",
                    choices=["Yes", "No"],
                    show_label=True,
                )
                reason = gr.Textbox(
                    scale=1,
                    label="Reason for Your Choice (Explain Your Reasoning & Highlight Useful Parts of Conversation)",
                    lines=5,
                )
            with gr.Row():
                trust = gr.Slider(
                    label="Trust",
                    minimum=1,
                    maximum=100,
                    value=50,
                    info="How much do you trust the financial advisor? Answer from 1 to 100. A score of 100 means you have complete trust in the financial advisor, while a score of 1 means you have no trust at all.",
                    step=1,
                )
                satisfaction = gr.Slider(
                    label="Satisfaction",
                    minimum=1,
                    maximum=100,
                    value=50,
                    info="How satisfied are you with the financial advisor? Answer from 1 to 100. A score of 100 means you are completely satisfied, while a score of 1 means you are not satisfied at all.",
                    step=1,
                )
            with gr.Row():
                knowledgeable = gr.Slider(
                    label="Knowledgeable",
                    minimum=1,
                    maximum=100,
                    value=50,
                    info="How knowledgeable do you feel after interacting with the financial advisor? Answer from 1 to 100. A score of 100 means you feel very knowledgeable, while a score of 1 means you feel not knowledgeable at all.",
                    step=1,
                )
                helpful = gr.Slider(
                    label="Helpful",
                    minimum=1,
                    maximum=100,
                    value=50,
                    info="How helpful do you find the financial advisor? Answer from 1 to 100. A score of 100 means you find the financial advisor very helpful, while a score of 1 means you find the financial advisor not helpful at all.",
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

    def tab_creation_preference_stage():
        with gr.Row():
            gr.HTML(value=PREFERENCE_ELICITATION_TASK, label="Preference Elicitation Task")
        with gr.Row():
            with gr.Column():
                whole_user_narrative = get_user_narrative_html(user_narrative)
                gr.HTML(value=whole_user_narrative, label="User Narrative")
            with gr.Column():
                with gr.Row():
                    elicitation_chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    start_conversation = gr.Button(value="Start Conversation")
                with gr.Row():
                    msg = gr.Textbox(scale=1, label="User Input")
                with gr.Row():
                    msg_button = gr.Button(value="Send This Message to Advisor", interactive=False)
                    continue_button = gr.Button(value="Show More of the Advisor’s Answer", interactive=False)
        return {
            "start_conversation": start_conversation,
            "msg_button": msg_button,
            "continue_button": continue_button,
            "msg": msg,
            "elicitation_chatbot": elicitation_chatbot,
        }

    def tab_final_evaluation(first_comp, second_comp, third_comp, fourth_comp, fifth_comp):
        with gr.Row():
            gr.HTML(value=FINAL_EVALUATION)
        with gr.Row():
            ranking_first_comp = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{first_comp}")
            ranking_second_comp = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{second_comp}")
            ranking_third_comp = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{third_comp}")
            ranking_fourth_comp = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{fourth_comp}")
            ranking_fifth_comp = gr.Dropdown(choices=[1, 2, 3, 4, 5], label=f"{fifth_comp}")
        with gr.Row():
            textbox = gr.HTML(
                """<div style="background-color: #f8d7da; color: #721c24; padding: 15px; border: 1px solid #f5c6cb; border-radius: 5px; margin-bottom: 20px;">
                    <strong>Please rank the stocks from 1 to 5, where 1 is the most preferred and 5 is the least preferred.</strong> 
                    <br>
                    <strong>Make sure to assign different scores to different stocks.</strong>
                </div>"""
            )
            submit_ranking = gr.Button(value="Submit Ranking")
        return {
            "first": {"comp": first_comp, "ranking_first_comp": ranking_first_comp},
            "second": {"comp": second_comp, "ranking_second_comp": ranking_second_comp},
            "third": {"comp": third_comp, "ranking_third_comp": ranking_third_comp},
            "fourth": {"comp": fourth_comp, "ranking_fourth_comp": ranking_fourth_comp},
            "fifth": {"comp": fifth_comp, "ranking_fifth_comp": ranking_fifth_comp},
            "submit_ranking": submit_ranking,
            "text_box": textbox,
        }

    def click_control_exploration_stage(tabs):
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

    def click_control_preference_stage(tabs):
        (
            start_conversation,
            msg_button,
            continue_button,
            elicitation_chatbot,
            msg,
        ) = (
            tabs["start_conversation"],
            tabs["msg_button"],
            tabs["continue_button"],
            tabs["elicitation_chatbot"],
            tabs["msg"],
        )
        start_conversation.click(
            lambda history: respond_start_conversation(
                history, SYSTEM_INSTRUECTION_PREFERENCE_ELICITATION, user_elicitation=True
            ),
            [elicitation_chatbot],
            [elicitation_chatbot, start_conversation, msg_button, continue_button],
        )
        msg_button.click(
            lambda message, history: respond(
                message,
                user_preference_elicitation_data["history"],
                SYSTEM_INSTRUECTION_PREFERENCE_ELICITATION,
                user_elicitation=True,
            ),
            [msg, elicitation_chatbot],
            [msg, elicitation_chatbot],
        )
        continue_button.click(
            lambda history: respond_continue(
                user_preference_elicitation_data["history"],
                SYSTEM_INSTRUECTION_PREFERENCE_ELICITATION,
                user_elicitation=True,
            ),
            [elicitation_chatbot],
            [elicitation_chatbot],
        )

    def click_control_final_evaluation(tabs):
        first_comp, ranking_first_comp = tabs["first"]["comp"], tabs["first"]["ranking_first_comp"]
        second_comp, ranking_second_comp = tabs["second"]["comp"], tabs["second"]["ranking_second_comp"]
        third_comp, ranking_third_comp = tabs["third"]["comp"], tabs["third"]["ranking_third_comp"]
        fourth_comp, ranking_fourth_comp = tabs["fourth"]["comp"], tabs["fourth"]["ranking_fourth_comp"]
        fifth_comp, ranking_fifth_comp = tabs["fifth"]["comp"], tabs["fifth"]["ranking_fifth_comp"]
        result_textbox = tabs["text_box"]
        submit_ranking = tabs["submit_ranking"]
        submit_ranking.click(
            lambda ranking_first_comp, ranking_second_comp, ranking_third_comp, ranking_fourth_comp, ranking_fifth_comp: respond_final_ranking(
                first_comp,
                ranking_first_comp,
                second_comp,
                ranking_second_comp,
                third_comp,
                ranking_third_comp,
                fourth_comp,
                ranking_fourth_comp,
                fifth_comp,
                ranking_fifth_comp,
            ),
            # Input components (names and rankings)
            [
                ranking_first_comp,
                ranking_second_comp,
                ranking_third_comp,
                ranking_fourth_comp,
                ranking_fifth_comp,
            ],
            # Output component(s) where you want the result to appear, e.g., result_textbox
            [result_textbox],
        )

    def respond(message, history, system_instruction, tab_name=None, user_elicitation=False):
        """
        Return:
        msg
        chat_history
        retrieved_passage
        rewritten_query

        """
        assert (
            tab_name is not None or user_elicitation is True
        ), "Tab name is required for the start of the conversation unless it is not preference elicitation."
        # Add user profile to system instruction
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                model=model,
                terminator=terminator,
            )
            # print(f"Tab: {tab_name}\nSystem Instruction:{system_instruction}")
        # Formatting Input
        print(f"User Message: {message} in Tab: {tab_name}")
        # From string to list [{"role":"user", "content": message}, ...]
        history = gradio_to_huggingface_message(history)
        # We can implement context window here as we need all the system interaction. We can cut some of the early interactions if needed.
        history = conversation_window(history, CONV_WINDOW)
        print(f"History Length: {len(history)}")
        print(f"History: {history}")
        # Add system instruction to the history
        history = format_context(system_instruction, history)
        # Add user message to the history
        history_with_user_utterance = format_user_message(message, history)

        outputs_text, history = response_generation(
            history_with_user_utterance,
            model,
            tokenizer,
            max_tokens=128,
            device=DEVICE,
            terminators=terminator,
        )
        # exclude system interaction and store the others in the history
        history = huggingface_to_gradio_message(history)
        if tab_name is not None:
            print(f"Tab: {tab_name}\nSystem Output: {outputs_text}")

            # Log the user message and response
            log_action(tab_name, "User Message", message)
            log_action(tab_name, "Response", outputs_text)
            # Store the updated history for this tab
            tab_data[tab_name]["history"] = history
        if user_elicitation:
            print(f"User Elicitation\nSystem Output: {outputs_text}")
            log_action("User_Elicitation", "User Message", message)
            log_action("User_Elicitation", "Response", outputs_text)
            user_preference_elicitation_data["history"] = history

        return "", history

    def respond_start_conversation(history, system_instruction, tab_name=None, user_elicitation=False):
        assert (
            tab_name is not None or user_elicitation is True
        ), "Tab name is required for the start of the conversation unless it is not preference elicitation."
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                model=model,
                terminator=terminator,
            )
            print(f"Tab: {tab_name}\nSystem Instruction:{system_instruction}")
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
        if tab_name is not None:
            print(f"Tab: {tab_name}\nHistory: {history}")

            # Log the user message and response
            log_action(tab_name, "User Message", first_message)
            log_action(tab_name, "Response", outputs_text)
            # Store the updated history for this tab
            tab_data[tab_name]["history"] = history
        if user_elicitation:
            print(f"User Elicitation\nHistory: {history}")
            log_action("User_Elicitation", "User Message", first_message)
            log_action("User_Elicitation", "Response", outputs_text)
            user_preference_elicitation_data["history"] = history

        return (
            history,
            gr.Button(value="Start Conversation", interactive=False),
            gr.Button(value="Send This Message to Advisor", interactive=True),
            gr.Button(value="Show More of the Advisor’s Answer", interactive=True),
        )

    def respond_continue(history, system_instruction, tab_name=None, user_elicitation=False):
        assert (
            tab_name is not None or user_elicitation is True
        ), "Tab name is required for the start of the conversation."
        # Add user profile to system instruction
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                model=model,
                terminator=terminator,
            )
            # print(f"Tab: {tab_name}\nSystem Instruction:{system_instruction}")
        message = "continue"
        history = gradio_to_huggingface_message(history)
        history = conversation_window(history, CONV_WINDOW)
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
        if tab_name is not None:
            log_action(tab_name, "Show More of the Advisor’s Answer", "User continued the conversation")
            log_action(tab_name, "Response", outputs_text)

            # Update history for this tab
            tab_data[tab_name]["history"] = history
        if user_elicitation:
            print(f"User Elicitation\nSystem Output: {outputs_text}")
            log_action("User_Elicitation", "Response", outputs_text)
            user_preference_elicitation_data["history"] = history

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

    def respond_final_ranking(
        first_comp,
        ranking_first_comp,
        second_comp,
        ranking_second_comp,
        third_comp,
        ranking_third_comp,
        fourth_comp,
        ranking_fourth_comp,
        fifth_comp,
        ranking_fifth_comp,
    ):
        # make sure that they are not the same
        ranking_list = [
            ranking_first_comp,
            ranking_second_comp,
            ranking_third_comp,
            ranking_fourth_comp,
            ranking_fifth_comp,
        ]
        if len(set(ranking_list)) != len(ranking_list):
            return """<div style="background-color: #fff3cd; color: #856404; padding: 15px; border: 1px solid #ffeeba; border-radius: 5px; margin-bottom: 20px;">
                        <strong>Please make sure that you are not ranking the same stock multiple times.</strong>
                    </div>"""
        else:
            log_action("Final_Ranking", first_comp, ranking_first_comp)
            log_action("Final_Ranking", second_comp, ranking_second_comp)
            log_action("Final_Ranking", third_comp, ranking_third_comp)
            log_action("Final_Ranking", fourth_comp, ranking_fourth_comp)
            log_action("Final_Ranking", fifth_comp, ranking_fifth_comp)
            return """<div style="background-color: #d4edda; color: #155724; padding: 15px; border: 1px solid #c3e6cb; border-radius: 5px; margin-bottom: 20px;">
                        <strong>Thank you for participating in the experiment. This concludes the session. You may now close the tab.</strong>
                    </div>"""

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
        user_narrative = markdown.markdown(raw_context_list[0]["user_narrative"].replace("\n", "<br>"))

        # # initialize tab data
        for comp in [first_comp, second_comp, third_comp, fourth_comp, fifth_comp]:
            tab_data[comp] = {"history": [], "selection": "", "reason": ""}

        # EXperiment Instruction
        with gr.Tab("Experiment Instruction") as instruction_tab:
            gr.HTML(value=INSTRUCTION_PAGE, label="Experiment Instruction")
        # User Preference Elicitation Tab
        with gr.Tab("Preference Elicitation Stage") as preference_elicitation_tab:
            user_preference_elicitation_tab = tab_creation_preference_stage()
            click_control_preference_stage(user_preference_elicitation_tab)
        with gr.Tab("Financial Decision Stage"):
            # Experiment Tag
            first_tab = tab_creation_exploration_stage(0)
            click_control_exploration_stage(first_tab)
            second_tab = tab_creation_exploration_stage(1)
            click_control_exploration_stage(second_tab)
            third_tab = tab_creation_exploration_stage(2)
            click_control_exploration_stage(third_tab)
            fourth_tab = tab_creation_exploration_stage(3)
            click_control_exploration_stage(fourth_tab)
            fifth_tab = tab_creation_exploration_stage(4)
            click_control_exploration_stage(fifth_tab)
        with gr.Tab("Final Evaluation Stage") as final_evaluation:
            final_evaluation_tab = tab_final_evaluation(first_comp, second_comp, third_comp, fourth_comp, fifth_comp)
            click_control_final_evaluation(final_evaluation_tab)

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
    # if DEBUG:
    #     tokenizer, terminator, model = "", "", ""
    # else:
    tokenizer = AutoTokenizer.from_pretrained(RESPONSE_GENERATOR)
    tokenizer, terminator = prepare_tokenizer(tokenizer)
    p
    model = AutoModelForCausalLM.from_pretrained(
        RESPONSE_GENERATOR,
        torch_dtype=torch.float16,
        pad_token_id=tokenizer.eos_token_id,
    ).to(DEVICE)
    demo = create_demo(
        model, tokenizer, terminator, system_instruction_without_context, stock_context_list, raw_context_list
    )
    demo.launch(share=True)

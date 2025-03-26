import sys
import os
from datetime import datetime
import json
import uuid
from pathlib import Path
from huggingface_hub import CommitScheduler, login
from datasets import load_dataset
import gradio as gr
import markdown
from together import Together

ROOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./")
sys.path.append(ROOT_FILE)
from components.induce_personality import construct_big_five_words
from components.chat_conversation import (
    format_user_message,
    format_context,
    gradio_to_huggingface_message,
    huggingface_to_gradio_message,
    conversation_window,
    generate_response_local_api,
    generate_response_together_api,
    generate_response_debugging,
)
from components.constant import (
    CONV_WINDOW,
    API_URL,
)

LOG_DIR = os.path.join(ROOT_FILE, "log/api/")
if os.path.exists(LOG_DIR) is False:
    os.makedirs(LOG_DIR)

# Load Static Files
STATIC_FILE = os.path.join(ROOT_FILE, "_static")
LOG_DIR = os.path.join(ROOT_FILE, "log/test_session/")
INSTRUCTION_PAGE_FILE = os.path.join(STATIC_FILE, "html/instruction_page.html")
USER_NARRATIVE_FILE = os.path.join(STATIC_FILE, "html/user_narrative.html")
PREFERENCE_ELICITATION_TASK_FILE = os.path.join(STATIC_FILE, "html/system_instruction_preference_elicitation.html")
EVALUATION_INSTRUCTION_FILE = os.path.join(STATIC_FILE, "html/evaluation_instruction.html")
GENERAL_INSTRUCTION_FILE = os.path.join(STATIC_FILE, "html/general_instruction.html")
FINAL_EVALUATION_FILE = os.path.join(STATIC_FILE, "html/final_evaluation.html")
SYSTEM_INSTRUCTION_PERSONALIZATION_FILE = os.path.join(STATIC_FILE, "txt/system_instruction_personalization.txt")
SYSTEM_INSTRUCTION_NON_PERSONALIZATION_FILE = os.path.join(
    STATIC_FILE, "txt/system_instruction_non_personalization.txt"
)
SYSTEM_INSTRUCTION_PERSONALITY_FILE = os.path.join(STATIC_FILE, "txt/system_instruction_personality.txt")
SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_FILE = os.path.join(
    STATIC_FILE, "txt/system_instruction_preference_elicitation.txt"
)
SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY_FILE = os.path.join(
    STATIC_FILE, "txt/system_instruction_preference_elicitation_personality.txt"
)
SUMMARIZATION_PROMPT_FILE = os.path.join(STATIC_FILE, "txt/system_summarization_user_preference_elicitation.txt")
PERSONALITY_EXT_FILE = os.path.join(STATIC_FILE, "txt/personality_ext.txt")
PERSONALITY_INT_FILE = os.path.join(STATIC_FILE, "txt/personality_int.txt")
PERSONALITY_MISADVISOR_FILE = os.path.join(STATIC_FILE, "txt/personality_misadvisor.txt")

uuid_this_session = str(uuid.uuid4())
system_order = "first"
feedback_dir = Path("user_feedback_debug/")
feedback_file_interaction = feedback_dir / f"interaction_{uuid_this_session}_{system_order}.json"
feedback_file_summarization = feedback_dir / f"summarization_{uuid_this_session}_{system_order}.json"
feedback_file_round_evaluation = feedback_dir / f"round_evaluation_{uuid_this_session}_{system_order}.json"
feedback_file_final_ranking = feedback_dir / f"final_ranking_{uuid_this_session}_{system_order}.json"
feedback_file_final_survey = feedback_dir / f"final_survey_{uuid_this_session}_{system_order}.json"
feedback_folder = feedback_file_interaction.parent
feedback_folder.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

# scheduler = CommitScheduler(
#     repo_id=os.getenv("LOGGING_FILE"),
#     repo_type="dataset",
#     folder_path=feedback_folder,
#     path_in_repo="data",
#     token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
#     every=1,
# )


# Function to save user feedback
# def save_feedback(user_id: str, uuid: str, type: str, value, feedback_file) -> None:
#     """
#     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
#     """
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     with scheduler.lock:
#         with feedback_file.open("a") as f:
#             f.write(
#                 json.dumps({"user_id": user_id, "uuid": uuid, "timestamp": timestamp, "type": type, "value": value})
#             )
#             f.write("\n")


# Load the required static content from files
def load_static_content(file_path):

    with open(file_path, "r") as f:
        return f.read()


def ensure_directory_exists(directory_path):
    """Ensures the given directory exists; creates it if it does not."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


INSTRUCTION_PAGE = load_static_content(INSTRUCTION_PAGE_FILE)
EVALUATION_INSTRUCTION = load_static_content(EVALUATION_INSTRUCTION_FILE)
GENERAL_INSTRUCTION = load_static_content(GENERAL_INSTRUCTION_FILE)
USER_NARRATIVE = load_static_content(USER_NARRATIVE_FILE)
PREFERENCE_ELICITATION_TASK = load_static_content(PREFERENCE_ELICITATION_TASK_FILE)
FINAL_EVALUATION = load_static_content(FINAL_EVALUATION_FILE)
SYSTEM_INSTRUCTION_PERSONALIZATION = load_static_content(SYSTEM_INSTRUCTION_PERSONALIZATION_FILE)
SYSTEM_INSTRUCTION_NON_PERSONALIZATION = load_static_content(SYSTEM_INSTRUCTION_NON_PERSONALIZATION_FILE)
SYSTEM_INSTRUCTION_PERSONALITY = load_static_content(SYSTEM_INSTRUCTION_PERSONALITY_FILE)
SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION = load_static_content(SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_FILE)
SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY = load_static_content(
    SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY_FILE
)
SUMMARIZATION_PROMPT = load_static_content(SUMMARIZATION_PROMPT_FILE)
PERSONALITY_EXT = load_static_content(PERSONALITY_EXT_FILE)
PERSONALITY_INT = load_static_content(PERSONALITY_INT_FILE)
PERSONALITY_MISADVISOR = load_static_content(PERSONALITY_MISADVISOR_FILE)

# Other constants
FIRST_MESSAGE = "Hey"
USER_PREFERENCE_SUMMARY = True
DEBUG = False
API_TYPE = "together"
assert API_TYPE in ["together", "local", "debug"], "The API should be either 'together' or 'local'"
if API_TYPE == "together":
    TOGETHER_CLIENT = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def generate_username_pwd_list(data):
    user_list = []
    demo_list = []
    for index, row in data.iterrows():
        user_list.append((row["user"], str(row["pwd"])))
        demo_list.append((row["demo"], str(row["pwd"])))
    return user_list, demo_list


def load_username_and_pwd():
    login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
    dataset = load_dataset(os.getenv("USER_PWD_FILE"))
    df = dataset["train"].to_pandas()
    user_list, demo_list = generate_username_pwd_list(df)
    return user_list, demo_list


def get_context_list(synthetic_data_path):
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
    print(name)
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


def get_user_narrative_from_raw(raw_narrative):
    return get_user_narrative_html(markdown.markdown(raw_narrative.replace("\n", "<br>")))


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


def log_action(user_id, tab_name, action, details):
    """
    Log actions for each tab (stock).
    """
    log_file_dir = os.path.join(LOG_DIR, f"{user_id}")
    if os.path.exists(log_file_dir) is False:
        os.makedirs(log_file_dir)
    log_file = os.path.join(log_file_dir, f"{tab_name}.txt")
    with open(log_file, "a") as f:
        f.write(f"Action: {action} | Details: {details}\n")


def add_user_profile_to_system_instruction(
    user_id, system_instruction, user_preference_elicitation_data, summary, terminator
):
    exp_id = int(user_id.split("_")[-3])
    # exp_id = 1 => No personalization
    if exp_id == 1:
        return system_instruction
    if summary:
        if user_preference_elicitation_data["summary_history"] == "":
            # Format prompt
            summarization_prompt = SUMMARIZATION_PROMPT + "\nPrevious Conversations: {}".format(
                user_preference_elicitation_data["history"]
            )
            summarization_instruction = [{"role": "system", "content": summarization_prompt}]
            if API_TYPE == "local":
                summ, _ = generate_response_local_api(summarization_instruction, terminator, 512, API_URL)
            elif API_TYPE == "together":
                summ, _ = generate_response_together_api(summarization_instruction, 512, TOGETHER_CLIENT)
            else:
                summ, _ = generate_response_debugging(summarization_instruction)
            user_preference_elicitation_data["summary_history"] = summ
            # # log_action(user_id, "Prompt", "Preference Elicitation Summarization", summ)
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "preference_elicitation_summarization",
            #     {"summarization": summ},
            #     feedback_file_summarization,
            # )
        system_instruction += f"\nUser Profile collected in the previous conversations: {user_preference_elicitation_data['summary_history']}\n"
    else:
        system_instruction += (
            f"\nUser Profile collected in the previous conversations: {user_preference_elicitation_data['history']}\n"
        )
    return system_instruction


def likert_evaluation(content):
    return gr.Radio(
        [1, 2, 3, 4, 5, 6, 7],
        label=f"{content}",
        show_label=True,
    )


def create_demo():
    global context_info_list, terminator

    def tab_creation_exploration_stage(order, comp, context):
        comp_list = [
            "Eli Lilly and Company",
            "Johnson & Johnson",
            "Philip Morris International Inc",
            "Exxon Mobil Corporation",
            "Amazon.com, Inc.",
            "JP Morgan Chase & Co.",
            "Coca-Cola Company",
            "Walmart Inc.",
            "Apple Inc.",
            "Procter & Gamble Company",
            "Tesla, Inc.",
            "Bank of America Corporation",
        ]
        with gr.Tab(comp_list[order], id=comp):
            general_instruction = gr.HTML(label="General Instruction")
            with gr.Row():
                # with gr.Column():
                #     with gr.Row():
                #         round_instruction = gr.HTML(label="Round Instruction")
                with gr.Column():
                    with gr.Row():
                        chatbot = gr.Chatbot(height=600)
                    with gr.Row():
                        start_conversation = gr.Button(value="Start Conversation")
                    with gr.Row():
                        msg = gr.Textbox(scale=2, lines=3, label="User Input", container=False)
                        msg_button = gr.Button(value="Send This Message to Advisor", interactive=False)
                        continue_button = gr.Button(value="Show More of the Advisor’s Answer", interactive=False)
            with gr.Row():
                clear = gr.ClearButton([msg, chatbot])
        return {
            "comp": comp,
            "system_instruction_context": context,
            "start_conversation": start_conversation,
            "msg_button": msg_button,
            "continue_button": continue_button,
            "chatbot": chatbot,
            "msg": msg,
            "general_instruction": general_instruction,
            # "round_instruction": round_instruction,
        }

    def tab_creation_preference_stage():
        # Custom CSS to reduce padding and margin of the textbox
        with gr.Row():
            gr.HTML(value=PREFERENCE_ELICITATION_TASK, label="Preference Elicitation Task")
        with gr.Row():
            # with gr.Column():
            #     user_narrative = gr.HTML(label="User Narrative")
            with gr.Column():
                with gr.Row():
                    elicitation_chatbot = gr.Chatbot(height=600)
                with gr.Row():
                    start_conversation = gr.Button(value="Start Conversation")
                with gr.Row():
                    msg = gr.Textbox(scale=2, lines=3, label="User Input", container=False)
                    msg_button = gr.Button(value="Send This Message to Advisor", interactive=False)
                    continue_button = gr.Button(value="Show More of the Advisor’s Answer", interactive=False)
        return {
            "start_conversation": start_conversation,
            "msg_button": msg_button,
            "continue_button": continue_button,
            "msg": msg,
            "elicitation_chatbot": elicitation_chatbot,
            # "user_narrative": user_narrative,
        }

    def tab_personality_injection():
        gr.Markdown("## Choose Your Advisor Personality")
        # A radio button to choose between Optimist and Pragmatis
        personality_choice = gr.Radio(
            choices=[
                "Optimist (Friendly, Daring, Confident)",
                "Pragmatist (Disciplined, Cautious, Practical)",
                "Misadvisor (Just for fun! Unreliable, Messy, Sarcastic - don't take it seriously!)",
            ],
            value="Optimist",
            label="Select a Personality",
            interactive=True,
        )
        # An HTML element to display personality details/description
        personality_description = gr.HTML(value="", label="Personality Details")
        # A button that, when clicked, confirms the selection
        personality_submit = gr.Button("Apply Personality")

        # Return a dict with the UI components for later use
        return {
            "personality_choice": personality_choice,
            "personality_description": personality_description,
            "personality_submit": personality_submit,
        }

    def click_control_personality_injection_stage(
        tabs, system_description_without_context, system_description_user_elicitation
    ):
        def respond_personality(personality_choice):
            if personality_choice == "Optimist (Friendly, Daring, Confident)":
                new_sys_desc_no_ctx = SYSTEM_INSTRUCTION_PERSONALITY.format(personality=PERSONALITY_EXT)
                new_sys_desc_user_elicitation = SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(
                    personality=PERSONALITY_EXT
                )
                message = "Optimist advisor personality is successfully applied."
            elif personality_choice == "Pragmatist (Disciplined, Cautious, Practical)":
                new_sys_desc_no_ctx = SYSTEM_INSTRUCTION_PERSONALITY.format(personality=PERSONALITY_INT)
                new_sys_desc_user_elicitation = SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(
                    personality=PERSONALITY_INT
                )
                message = "Pragmatist personality is successfully applied."
            elif personality_choice == "Misadvisor (Unreliable, Messy, Sarcastic)":
                new_sys_desc_no_ctx = SYSTEM_INSTRUCTION_PERSONALITY.format(personality=PERSONALITY_MISADVISOR)
                new_sys_desc_user_elicitation = SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(
                    personality=PERSONALITY_MISADVISOR
                )
                message = "Misadvisor personality is successfully applied."
            else:
                new_sys_desc_no_ctx = SYSTEM_INSTRUCTION_PERSONALITY.format(personality="None")
                new_sys_desc_user_elicitation = SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(
                    personality="None"
                )
                message = "No personality is applied."
            return new_sys_desc_no_ctx, new_sys_desc_user_elicitation, message

        # Only pass the personality_choice as input
        tabs["personality_submit"].click(
            fn=respond_personality,
            inputs=[tabs["personality_choice"]],
            outputs=[
                system_description_without_context,
                system_description_user_elicitation,
                tabs["personality_description"],
            ],
        )

    def click_control_exploration_stage(
        tabs, user_id, tab_session, user_preference_elicitation_session, system_description_without_context
    ):
        (
            comp,
            system_instruction_context,
            start_conversation,
            msg_button,
            continue_button,
            chatbot,
            msg,
        ) = (
            tabs["comp"],
            tabs["system_instruction_context"],
            tabs["start_conversation"],
            tabs["msg_button"],
            tabs["continue_button"],
            tabs["chatbot"],
            tabs["msg"],
        )
        system_instruction = ""
        start_conversation.click(
            lambda user_id, tab_session, history, comp, user_preference_elicitation_session, system_description_without_context, system_instruction_context: respond_start_conversation(
                user_id,
                tab_session,
                history,
                system_instruction,
                comp,
                user_preference_elicitation_data=user_preference_elicitation_session,
                system_description_without_context=system_description_without_context,
                system_instruction_context=system_instruction_context,
            ),
            [
                user_id,
                tab_session,
                chatbot,
                comp,
                user_preference_elicitation_session,
                system_description_without_context,
                system_instruction_context,
            ],
            [tab_session, chatbot, start_conversation, msg_button, continue_button],
        )
        msg_button.click(
            lambda user_id, tab_session, message, history, comp, user_preference_elicitation_session, system_description_without_context, system_instruction_context: respond(
                user_id,
                tab_session,
                message,
                tab_session["history"],
                system_instruction,
                comp,
                user_preference_elicitation_data=user_preference_elicitation_session,
                system_description_without_context=system_description_without_context,
                system_instruction_context=system_instruction_context,
            ),
            [
                user_id,
                tab_session,
                msg,
                chatbot,
                comp,
                user_preference_elicitation_session,
                system_description_without_context,
                system_instruction_context,
            ],
            [tab_session, msg, chatbot],
        )
        continue_button.click(
            lambda user_id, tab_session, history, comp, user_preference_elicitation_session, system_description_without_context, system_instruction_context: respond_continue(
                user_id,
                tab_session,
                tab_session["history"],
                system_instruction,
                comp,
                user_preference_elicitation_data=user_preference_elicitation_session,
                system_description_without_context=system_description_without_context,
                system_instruction_context=system_instruction_context,
            ),
            [
                user_id,
                tab_session,
                chatbot,
                comp,
                user_preference_elicitation_session,
                system_description_without_context,
                system_instruction_context,
            ],
            [tab_session, chatbot],
        )

    def click_control_preference_stage(
        tabs, user_id, user_preference_elicitation_session, system_description_user_elicitation
    ):
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
        # nonlocal user_id
        start_conversation.click(
            lambda user_id, user_preference_elicitation_data, history, system_description_user_elicitation: respond_start_conversation(
                user_id,
                user_preference_elicitation_data,
                history,
                system_description_user_elicitation,
                user_elicitation=True,
            ),
            [user_id, user_preference_elicitation_session, elicitation_chatbot, system_description_user_elicitation],
            [user_preference_elicitation_session, elicitation_chatbot, start_conversation, msg_button, continue_button],
        )
        msg_button.click(
            lambda user_id, tab_data, message, history, system_description_user_elicitation: respond(
                user_id,
                tab_data,
                message,
                tab_data["history"],
                system_description_user_elicitation,
                user_elicitation=True,
            ),
            [
                user_id,
                user_preference_elicitation_session,
                msg,
                elicitation_chatbot,
                system_description_user_elicitation,
            ],
            [user_preference_elicitation_session, msg, elicitation_chatbot],
        )
        continue_button.click(
            lambda user_id, tab_data, history, system_description_user_elicitation: respond_continue(
                user_id,
                tab_data,
                tab_data["history"],
                system_description_user_elicitation,
                user_elicitation=True,
            ),
            [user_id, user_preference_elicitation_session, elicitation_chatbot, system_description_user_elicitation],
            [user_preference_elicitation_session, elicitation_chatbot],
        )

    def respond(
        user_id,
        tab_data,
        message,
        history,
        system_instruction,
        tab_name=None,
        user_elicitation=False,
        user_preference_elicitation_data=None,
        system_description_without_context=None,
        system_instruction_context=None,
    ):
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
        if system_description_without_context is not None and system_instruction_context is not None:
            system_instruction = system_description_without_context + "\n" + system_instruction_context
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                user_id,
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                terminator=terminator,
            )
        # From string to list [{"role":"user", "content": message}, ...]
        history = gradio_to_huggingface_message(history)
        # We can implement context window here as we need all the system interaction. We can cut some of the early interactions if needed.
        history = conversation_window(history, CONV_WINDOW)
        # Add system instruction to the history
        history = format_context(system_instruction, history)
        # Add user message to the history
        history_with_user_utterance = format_user_message(message, history)
        # Call API instead of locally handle it
        if API_TYPE == "local":
            outputs_text, history = generate_response_local_api(history_with_user_utterance, terminator, 128, API_URL)
        elif API_TYPE == "together":
            outputs_text, history = generate_response_together_api(history_with_user_utterance, 128, TOGETHER_CLIENT)
        else:
            outputs_text, history = generate_response_debugging(history_with_user_utterance)
        # exclude system interaction and store the others in the history
        history = huggingface_to_gradio_message(history)
        if tab_name is not None:
            # Log the user message and response
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": tab_name, "role": "user", "content": message},
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": tab_name, "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )
            # log_action(user_id, tab_name, "User Message", message)
            # log_action(user_id, tab_name, "Response", outputs_text)
            # Store the updated history for this tab
            tab_data["history"] = history
        if user_elicitation:
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "Interaction",
            #     {"type": "user_elicitation", "role": "user", "content": message},
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "Interaction",
            #     {"type": "user_elicitation", "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )
            # log_action(user_id, "User_Elicitation", "User Message", message)
            # log_action(user_id, "User_Elicitation", "Response", outputs_text)
            tab_data["history"] = history

        return tab_data, "", history

    def respond_start_conversation(
        user_id,
        tab_data,
        history,
        system_instruction,
        tab_name=None,
        user_elicitation=False,
        user_preference_elicitation_data=None,
        system_description_without_context=None,
        system_instruction_context=None,
    ):
        assert (
            tab_name is not None or user_elicitation is True
        ), "Tab name is required for the start of the conversation unless it is not preference elicitation."
        if system_description_without_context is not None and system_instruction_context is not None:
            system_instruction = system_description_without_context + "\n" + system_instruction_context
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                user_id,
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                terminator=terminator,
            )
        history = gradio_to_huggingface_message(history)
        history = format_context(system_instruction, history)
        first_message = FIRST_MESSAGE
        history_with_user_utterance = format_user_message(first_message, history)
        max_length = 128 if user_elicitation else 256
        if API_TYPE == "local":
            outputs_text, history = generate_response_local_api(
                history_with_user_utterance, terminator, max_length, API_URL
            )
        elif API_TYPE == "together":
            outputs_text, history = generate_response_together_api(
                history_with_user_utterance, max_length, TOGETHER_CLIENT
            )
        else:
            outputs_text, history = generate_response_debugging(history_with_user_utterance)
        # Format
        history = huggingface_to_gradio_message(history)
        if tab_name is not None:
            # Log the user message and response
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": tab_name, "role": "user", "content": first_message},
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": tab_name, "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )
            tab_data["history"] = history
        if user_elicitation:
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": "user_elicitation", "role": "user", "content": first_message},
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "Interaction",
            #     {"type": "user_elicitation", "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )
            tab_data["history"] = history
        return (
            tab_data,
            history,
            gr.Button(value="Start Conversation", interactive=False),
            gr.Button(value="Send This Message to Advisor", interactive=True),
            gr.Button(value="Show More of the Advisor’s Answer", interactive=True),
        )

    def respond_continue(
        user_id,
        tab_data,
        history,
        system_instruction,
        tab_name=None,
        user_elicitation=False,
        user_preference_elicitation_data=None,
        system_description_without_context=None,
        system_instruction_context=None,
    ):
        assert (
            tab_name is not None or user_elicitation is True
        ), "Tab name is required for the start of the conversation."
        # Add user profile to system instruction
        if system_description_without_context is not None and system_instruction_context is not None:
            system_instruction = system_description_without_context + "\n" + system_instruction_context
        if not user_elicitation:
            system_instruction = add_user_profile_to_system_instruction(
                user_id,
                system_instruction,
                user_preference_elicitation_data,
                summary=USER_PREFERENCE_SUMMARY,
                terminator=terminator,
            )
        message = "continue"
        history = gradio_to_huggingface_message(history)
        history = conversation_window(history, CONV_WINDOW)
        history = format_context(system_instruction, history)
        history_with_user_utterance = format_user_message(message, history)
        if API_TYPE == "local":
            outputs_text, history = generate_response_local_api(history_with_user_utterance, terminator, 128, API_URL)
        elif API_TYPE == "together":
            outputs_text, history = generate_response_together_api(history_with_user_utterance, 128, TOGETHER_CLIENT)
        else:
            outputs_text, history = generate_response_debugging(history_with_user_utterance)
        history = huggingface_to_gradio_message(history)
        if tab_name is not None:
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {
            #         "type": tab_name,
            #         "role": "user",
            #         "content": message,
            #     },
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": tab_name, "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )

            # Update history for this tab
            tab_data["history"] = history
        if user_elicitation:
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": "user_elicitation", "role": "user", "content": message},
            #     feedback_file_interaction,
            # )
            # save_feedback(
            #     user_id,
            #     uuid_this_session,
            #     "interaction",
            #     {"type": "user_elicitation", "role": "assistant", "content": outputs_text},
            #     feedback_file_interaction,
            # )
            tab_data["history"] = history
        return tab_data, history

    def get_context(index, raw_context_list, stock_context_list):
        comp = raw_context_list[index]["short_name"]
        context = stock_context_list[index]
        general_instruction, round_instruction = get_task_instruction_for_user(raw_context_list[index])
        return comp, context, general_instruction, round_instruction

    def set_user_id(request: gr.Request):
        #  DEBUG
        user_id = "user_2_0_0"
        # user_id = request.username
        user_in_narrative_id = user_id.split("_")[-1]
        narrative_id = user_id.split("_")[-2]
        experiment_id = user_id.split("_")[-3]
        return user_id, user_in_narrative_id, narrative_id, experiment_id

    def get_inst_without_context(experiment_id):
        # experiment_id = 1 => personalization
        # experiment_id = 2 => no personalization
        # experiment_id == 3 => ext personality
        # experiment_id == 4 => int personality
        if experiment_id == "0":
            return SYSTEM_INSTRUCTION_PERSONALIZATION
        elif experiment_id == "1":
            return SYSTEM_INSTRUCTION_NON_PERSONALIZATION
        elif experiment_id == "2":
            return SYSTEM_INSTRUCTION_PERSONALITY.format(personality=PERSONALITY_EXT)
        elif experiment_id == "3":
            return SYSTEM_INSTRUCTION_PERSONALITY.format(personality=PERSONALITY_INT)

    def get_user_preference_elicitation(experiment_id):
        if experiment_id == "0" or experiment_id == "1":
            return SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION
        elif experiment_id == "2":
            return SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(personality=PERSONALITY_EXT)
        elif experiment_id == "3":
            return SYSTEM_INSTRUCTION_PREFERENCE_ELICITATION_PERSONALITY.format(personality=PERSONALITY_INT)

    def get_stock_related_context(narrative_id, user_in_narrative_id):
        raw_context_list = build_raw_context_list(context_info_list[int(narrative_id)])
        stock_context_list = build_context(context_info_list[int(narrative_id)])
        return raw_context_list, stock_context_list

    def set_initial_values(request: gr.Request):
        # Set user specific information (Session State)
        user_id, user_in_narrative_id, narrative_id, experiment_id = set_user_id(request)
        user_id = "user_2_0_0"
        user_in_narrative_id = "0"
        experiment_id = "2"
        system_description_without_context = get_inst_without_context(experiment_id)
        system_description_user_elicitation = get_user_preference_elicitation(experiment_id)

        first_raw_context_list, first_stock_context_list = get_stock_related_context(0, user_in_narrative_id)
        second_raw_context_list, second_stock_context_list = get_stock_related_context(1, user_in_narrative_id)
        third_raw_context_list, third_stock_context_list = get_stock_related_context(2, user_in_narrative_id)
        raw_context_list = first_raw_context_list + second_raw_context_list + third_raw_context_list
        stock_context_list = first_stock_context_list + second_stock_context_list + third_stock_context_list
        # Tab Context
        first_comp, first_context, first_general_instruction, first_round_instruction = get_context(
            0, raw_context_list, stock_context_list
        )
        second_comp, second_context, second_general_instruction, second_round_instruction = get_context(
            1, raw_context_list, stock_context_list
        )
        third_comp, third_context, third_general_instruction, third_round_instruction = get_context(
            2, raw_context_list, stock_context_list
        )
        fourth_comp, fourth_context, fourth_general_instruction, fourth_round_instruction = get_context(
            3, raw_context_list, stock_context_list
        )
        fifth_comp, fifth_context, fifth_general_instruction, fifth_round_instruction = get_context(
            4, raw_context_list, stock_context_list
        )
        sixth_comp, sixth_context, sixth_general_instruction, sixth_round_instruction = get_context(
            5, raw_context_list, stock_context_list
        )
        seventh_comp, seventh_context, seventh_general_instruction, seventh_round_instruction = get_context(
            6, raw_context_list, stock_context_list
        )
        eighth_comp, eighth_context, eighth_general_instruction, eighth_round_instruction = get_context(
            7, raw_context_list, stock_context_list
        )
        ninth_comp, ninth_context, ninth_general_instruction, ninth_round_instruction = get_context(
            8, raw_context_list, stock_context_list
        )
        tenth_comp, tenth_context, tenth_general_instruction, tenth_round_instruction = get_context(
            9, raw_context_list, stock_context_list
        )
        eleventh_comp, eleventh_context, eleventh_general_instruction, eleventh_round_instruction = get_context(
            10, raw_context_list, stock_context_list
        )
        twelfth_comp, twelfth_context, twelfth_general_instruction, twelfth_round_instruction = get_context(
            11, raw_context_list, stock_context_list
        )
        return (
            user_id,
            user_in_narrative_id,
            narrative_id,
            experiment_id,
            system_description_without_context,
            system_description_user_elicitation,
            raw_context_list,
            stock_context_list,
            # user_narrative,
            first_comp,
            first_context,
            first_general_instruction,
            # first_round_instruction,
            second_comp,
            second_context,
            second_general_instruction,
            # second_round_instruction,
            third_comp,
            third_context,
            third_general_instruction,
            # third_round_instruction,
            fourth_comp,
            fourth_context,
            fourth_general_instruction,
            # fourth_round_instruction,
            fifth_comp,
            fifth_context,
            fifth_general_instruction,
            # fifth_round_instruction,
            sixth_comp,
            sixth_context,
            sixth_general_instruction,
            # sixth_round_instruction,
            seventh_comp,
            seventh_context,
            seventh_general_instruction,
            # seventh_round_instruction,
            eighth_comp,
            eighth_context,
            eighth_general_instruction,
            # eighth_round_instruction,
            ninth_comp,
            ninth_context,
            ninth_general_instruction,
            # ninth_round_instruction,
            tenth_comp,
            tenth_context,
            tenth_general_instruction,
            # tenth_round_instruction,
            eleventh_comp,
            eleventh_context,
            eleventh_general_instruction,
            # eleventh_round_instruction,
            twelfth_comp,
            twelfth_context,
            twelfth_general_instruction,
            # twelfth_round_instruction,
        )

    with gr.Blocks(title="RAG Chatbot Q&A", theme="Soft") as demo:
        # Set user specific information (Session State)
        user_id = gr.State()
        user_in_narrative_id = gr.State()
        narrative_id = gr.State()
        experiment_id = gr.State()
        system_description_without_context = gr.State()
        system_description_user_elicitation = gr.State()
        # Context data
        raw_context_list = gr.State()
        stock_context_list = gr.State()
        first_comp = gr.State()
        first_context = gr.State()
        second_comp = gr.State()
        second_context = gr.State()
        third_comp = gr.State()
        third_context = gr.State()
        fourth_comp = gr.State()
        fourth_context = gr.State()
        fifth_comp = gr.State()
        fifth_context = gr.State()
        sixth_comp = gr.State()
        sixth_context = gr.State()
        seventh_comp = gr.State()
        seventh_context = gr.State()
        eighth_comp = gr.State()
        eighth_context = gr.State()
        ninth_comp = gr.State()
        ninth_context = gr.State()
        tenth_comp = gr.State()
        tenth_context = gr.State()
        eleventh_comp = gr.State()
        eleventh_context = gr.State()
        twelfth_comp = gr.State()
        twelfth_context = gr.State()
        # Tab data
        if DEBUG:
            user_preference_elicitation_session = gr.State(
                value={
                    "history": "",
                    "summary_history": """User Profile collected in the previous conversations: Based on our previous conversation, here's a summary of your investment preferences:

    #     1. **Preferred Industries:** You're interested in investing in the healthcare sector, without a specific preference for sub-industries such as pharmaceuticals, medical devices, biotechnology, or healthcare services.
    #     2. **Value vs. Growth Stocks:** You prefer growth stocks, which have the potential for high returns but may be riskier.
    #     3. **Dividend vs. Non-Dividend Stocks:** You're open to both dividend and non-dividend growth stocks, focusing on reinvesting profits for future growth.
    #     4. **Cyclical vs. Non-Cyclical Stocks:** You're interested in cyclical stocks, which are sensitive to economic fluctuations and tend to perform well during economic expansions.""",
                }
            )
        else:
            user_preference_elicitation_session = gr.State(value={"history": "", "summary_history": ""})
        first_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        second_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        third_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        fourth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        fifth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        sixth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        seventh_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        eighth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        ninth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        tenth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        eleventh_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})
        twelfth_comp_session = gr.State(value={"history": [], "selection": "", "reason": ""})

        # Demonstration Instruction
        with gr.Tab("Demonstration Instruction") as instruction_tab:
            gr.HTML(value=INSTRUCTION_PAGE, label="Demonstration Instruction")
        # User Preference Elicitation Tab
        with gr.Tab("Personality Injection") as personality_injection_tab:
            personality_injection_tab = tab_personality_injection()
            click_control_personality_injection_stage(
                personality_injection_tab, system_description_without_context, system_description_user_elicitation
            )
        with gr.Tab("User Preference Elicitation") as preference_elicitation_tab:
            user_preference_elicitation_tab = tab_creation_preference_stage()
            # user_narrative = user_preference_elicitation_tab["user_narrative"]
            click_control_preference_stage(
                user_preference_elicitation_tab,
                user_id,
                user_preference_elicitation_session,
                system_description_user_elicitation,
            )
        with gr.Tab("Personalized Stock Assessment") as financial_decision:
            # Experiment Tag
            first_tab = tab_creation_exploration_stage(0, first_comp, first_context)
            click_control_exploration_stage(
                first_tab,
                user_id,
                first_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            second_tab = tab_creation_exploration_stage(1, second_comp, second_context)
            click_control_exploration_stage(
                second_tab,
                user_id,
                second_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            third_tab = tab_creation_exploration_stage(2, third_comp, third_context)
            click_control_exploration_stage(
                third_tab,
                user_id,
                third_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            fourth_tab = tab_creation_exploration_stage(3, fourth_comp, fourth_context)
            click_control_exploration_stage(
                fourth_tab,
                user_id,
                fourth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            fifth_tab = tab_creation_exploration_stage(4, fifth_comp, fifth_context)
            click_control_exploration_stage(
                fifth_tab,
                user_id,
                fifth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            sixth_tab = tab_creation_exploration_stage(5, sixth_comp, sixth_context)
            click_control_exploration_stage(
                sixth_tab,
                user_id,
                sixth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            seventh_tab = tab_creation_exploration_stage(6, seventh_comp, seventh_context)
            click_control_exploration_stage(
                seventh_tab,
                user_id,
                seventh_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            eighth_tab = tab_creation_exploration_stage(7, eighth_comp, eighth_context)
            click_control_exploration_stage(
                eighth_tab,
                user_id,
                eighth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            ninth_tab = tab_creation_exploration_stage(8, ninth_comp, ninth_context)
            click_control_exploration_stage(
                ninth_tab,
                user_id,
                ninth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            tenth_tab = tab_creation_exploration_stage(9, tenth_comp, tenth_context)
            click_control_exploration_stage(
                tenth_tab,
                user_id,
                tenth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            eleventh_tab = tab_creation_exploration_stage(10, eleventh_comp, eleventh_context)
            click_control_exploration_stage(
                eleventh_tab,
                user_id,
                eleventh_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            twelfth_tab = tab_creation_exploration_stage(11, twelfth_comp, twelfth_context)
            click_control_exploration_stage(
                twelfth_tab,
                user_id,
                twelfth_comp_session,
                user_preference_elicitation_session,
                system_description_without_context,
            )
            (
                first_general_instruction,
                second_general_instruction,
                third_general_instruction,
                fourth_general_instruction,
                fifth_general_instruction,
                sixth_general_instruction,
                seventh_general_instruction,
                eighth_general_instruction,
                ninth_general_instruction,
                tenth_general_instruction,
                eleventh_general_instruction,
                twelfth_general_instruction,
            ) = (
                first_tab["general_instruction"],
                second_tab["general_instruction"],
                third_tab["general_instruction"],
                fourth_tab["general_instruction"],
                fifth_tab["general_instruction"],
                sixth_tab["general_instruction"],
                seventh_tab["general_instruction"],
                eighth_tab["general_instruction"],
                ninth_tab["general_instruction"],
                tenth_tab["general_instruction"],
                eleventh_tab["general_instruction"],
                twelfth_tab["general_instruction"],
            )

        demo.load(
            set_initial_values,
            inputs=None,
            outputs=[
                user_id,
                user_in_narrative_id,
                narrative_id,
                experiment_id,
                system_description_without_context,
                system_description_user_elicitation,
                raw_context_list,
                stock_context_list,
                # user_narrative,
                first_comp,
                first_context,
                first_general_instruction,
                # first_round_instruction,
                second_comp,
                second_context,
                second_general_instruction,
                # second_round_instruction,
                third_comp,
                third_context,
                third_general_instruction,
                # third_round_instruction,
                fourth_comp,
                fourth_context,
                fourth_general_instruction,
                # fourth_round_instruction,
                fifth_comp,
                fifth_context,
                fifth_general_instruction,
                # fifth_round_instruction,
                sixth_comp,
                sixth_context,
                sixth_general_instruction,
                # sixth_round_instruction,
                seventh_comp,
                seventh_context,
                seventh_general_instruction,
                # seventh_round_instruction,
                eighth_comp,
                eighth_context,
                eighth_general_instruction,
                # eighth_round_instruction,
                ninth_comp,
                ninth_context,
                ninth_general_instruction,
                # ninth_round_instruction,
                tenth_comp,
                tenth_context,
                tenth_general_instruction,
                # tenth_round_instruction,
                eleventh_comp,
                eleventh_context,
                eleventh_general_instruction,
                # eleventh_round_instruction,
                twelfth_comp,
                twelfth_context,
                twelfth_general_instruction,
                # twelfth_round_instruction,
            ],
        )
    return demo


if __name__ == "__main__":
    file_path = os.path.join(ROOT_FILE, "./data/single_stock_data/experiment_processed_data.jsonl")
    topics = [
        "healthcare_growth_defensive",
        "dividend_value_defensive",
        "nondividend_value_cyclical",
    ]
    context_info_list = get_context_list(file_path)  # str to List of Dict
    # system instruction consist of Task, Personality, and Context
    """
    Personality
    ["extroverted", "introverted"]
    ["agreeable", "antagonistic"]
    ["conscientious", "unconscientious"]
    ["neurotic", "emotionally stable"]
    ["open to experience", "closed to experience"]]
    """
    # Global variables
    terminator = ["<eos>", "<unk>", "<sep>", "<pad>", "<cls>", "<mask>"]
    demo = create_demo()
    user_list, demo_list = load_username_and_pwd()
    demo.launch(
        share=True,
        # auth=user_list + demo_list + ["test", "test"],
    )

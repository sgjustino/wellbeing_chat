import os
import json
import gradio as gr
import requests

# Retrieve the API code from the environment variable
api_code = os.getenv("api_code")

# API endpoint
url = "https://api.corcel.io/v1/text/cortext/chat"

# API payload template
payload_template = {
    "model": "gpt-4o",
    "stream": False,
    "top_p": 1,
    "temperature": 0.0001,
    "max_tokens": 4096,
    "messages": []
}

# Headers with authorization
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_code}"
}

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant named Averie. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist named Cora who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX"

def call_api(prompt: str):
    payload = payload_template.copy()
    payload["messages"] = [{"role": "user", "content": prompt}]
    response = requests.post(url, json=payload, headers=headers)
    choices = response.json().get("choices", [])
    if choices and "message" in choices[0]:
        return choices[0]["message"]["content"]
    else:
        return "No valid response received from the API."

def chat_fn(user_input, chat_history):
    chat_history.append([user_input, None])
    # Create chat prompt with entire chat history
    full_chat_history = [f"User: {user_input}\nAverie: {bot_message}" for user_input, bot_message in chat_history if bot_message]
    chat_prompt = chat_system_prompt + "\n" + "\n".join(full_chat_history) + f"\nUser: {user_input}\nAverie: "
    chat_output = call_api(chat_prompt)
    chat_history[-1][1] = chat_output

    # Create evaluation prompt with entire chat history
    eval_prompt = eval_system_prompt + "\n" + "\n".join(full_chat_history) + f"\nUser: {user_input}"
    eval_output = call_api(eval_prompt)

    return chat_output, chat_history, eval_output

# Set up the Gradio interface
with gr.Blocks(css="style.css") as interface:
    with gr.Tabs():
        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane"):
                    gr.Markdown("### Chat with Averie")
                    chatbot = gr.ChatInterface(chat_fn, title="Averie Chatbot", description="Chat with Averie, your mental therapy assistant.", show_progress="full")
                with gr.Column(elem_id="right-pane"):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.HTML(elem_id="eval-output")
                    chat_history = gr.State([])

            def handle_submit(user_input, chat_history):
                # Process chat and evaluation
                chat_output, updated_chat_history, eval_response = chat_fn(user_input, chat_history)
                return updated_chat_history, gr.update(value=eval_response, scroll_to_output=True)

            chatbot.submit(fn=handle_submit, inputs=[chatbot.textbox, chat_history], outputs=[chatbot.chatbot, eval_output])
        
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About Averie and Cora

            ### Averie
            Averie is your friendly mental health assistant designed to provide supportive conversations. She aims to offer helpful and cheerful responses to improve mental well-being until professional help can be sought. Averie is always ready to listen and provide comfort.

            ### Cora
            Cora is a trained psychologist who evaluates the interactions between Averie and users. She conducts mental health analyses to identify potential issues and likely reasons. Cora provides insights based on the conversations to ensure users receive the best possible support and guidance.

            **Disclaimer:** This app is not a substitute for professional mental health treatment. If you are experiencing a mental health crisis or need professional help, please contact a qualified mental health professional.
            """)
            
# Launch the Gradio app
interface.launch(share=True)
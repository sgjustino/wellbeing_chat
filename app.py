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
    response.raise_for_status()  # Raise an error for unsuccessful requests
    choices = response.json()
    if isinstance(choices, list) and len(choices) > 0 and 'choices' in choices[0]:
        return choices[0]["choices"][0]["delta"]["content"]
    else:
        return "No valid response received from the API."

def chat_fn(user_input, history):
    # Create chat prompt with entire chat history
    chat_history = [f"User: {user}\nAverie: {bot}" for user, bot in history]
    chat_prompt = chat_system_prompt + "\n" + "\n".join(chat_history) + f"\nUser: {user_input}\nAverie: "
    chat_output = call_api(chat_prompt)
    
    # Update history
    history.append([user_input, chat_output])

    # Create evaluation prompt with entire chat history
    eval_prompt = eval_system_prompt + "\n" + "\n".join(chat_history) + f"\nUser: {user_input}"
    eval_output = call_api(eval_prompt)

    return chat_output, history, eval_output

def chat_interface(user_input, chat_history):
    chat_output, updated_chat_history, eval_response = chat_fn(user_input, chat_history)
    formatted_chat = format_chat(updated_chat_history)
    return formatted_chat, eval_response

def format_chat(chat_history):
    formatted_history = ""
    for user_message, averie_message in chat_history:
        formatted_history += f'<div class="user-message">User: {user_message}</div>'
        formatted_history += f'<div class="averie-message">Averie: {averie_message}</div>'
    return formatted_history

# Set up the Gradio interface
with gr.Blocks(css="style.css") as interface:
    with gr.Tabs():
        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane"):
                    gr.Markdown("### Chat with Averie")
                    chatbot = gr.HTML(elem_id="chat-output")
                    chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=2)
                    chat_submit = gr.Button("Submit", elem_id="submit-button")
                with gr.Column(elem_id="right-pane"):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.HTML(elem_id="eval-output")
                    chat_history = gr.State([])

            def handle_submit(user_input, chat_history):
                # Process chat and evaluation
                formatted_chat, eval_response = chat_interface(user_input, chat_history)
                return formatted_chat, eval_response

            chat_submit.click(fn=handle_submit, inputs=[chat_input, chat_history], outputs=[chatbot, eval_output])
            chat_input.submit(fn=handle_submit, inputs=[chat_input, chat_history], outputs=[chatbot, eval_output])
        
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

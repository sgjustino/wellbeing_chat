import os
import json
import gradio as gr
import requests
import time

# Retrieve the API code from the environment variable
api_code = os.getenv("api_code")

# API endpoint
url = "https://api.corcel.io/v1/text/cortext/chat"

# API payload template
payload_template = {
    "model": "gpt-4o",
    "stream": True,
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

def call_api(messages):
    payload = payload_template.copy()
    payload["messages"] = messages
    response = requests.post(url, json=payload, headers=headers, stream=True)
    response_text = ""
    for chunk in response.iter_lines():
        if chunk:
            token = json.loads(chunk.decode()[6:])['choices'][0]['delta'].get('content', '')
            response_text += token
            yield token

def respond(message, history):
    messages = [{"role": "system", "content": chat_system_prompt}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = ""
    for token in call_api(messages):
        response += token
        yield response, history + [(message, response)]

def chat_and_evaluate(user_input, chat_history):
    messages = [{"role": "system", "content": eval_system_prompt}, {"role": "user", "content": user_input}]
    eval_output = call_api(messages)
    return user_input, list(eval_output)

# Set up the Gradio interface
with gr.Blocks(css="style.css") as interface:
    with gr.Tabs():
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About Averie and Cora
            ### Averie
            Averie is your friendly mental health assistant designed to provide supportive conversations. She aims to offer helpful and cheerful responses to improve mental well-being until professional help can be sought. Averie is always ready to listen and provide comfort.
            ### Cora
            Cora is a trained psychologist who evaluates the interactions between Averie and users. She conducts mental health analyses to identify potential issues and likely reasons. Cora provides insights based on the conversations to ensure users receive the best possible support and guidance.
            **Disclaimer:** This app is not a substitute for professional mental health treatment. If you are experiencing a mental health crisis or need professional help, please contact a qualified mental health professional.
            """)

        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane"):
                    gr.Markdown("### Chat with Averie")
                    chat = gr.Chatbot(label="Averie")
                    chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1)
                    chat_submit = gr.Button("Submit", elem_id="submit-button", variant="primary")
                    chat_history = gr.State([])

                    def handle_submit(user_input, chat_history):
                        chat_output = respond(user_input, chat_history)
                        eval_input, eval_output = chat_and_evaluate(user_input, chat_history)
                        chat_history.append((user_input, chat_output))
                        return chat_output, eval_output

                    chat_input.submit(fn=handle_submit, inputs=[chat_input, chat_history], outputs=[chat, eval_output])
                    chat_submit.click(fn=handle_submit, inputs=[chat_input, chat_history], outputs=[chat, eval_output])

                with gr.Column(elem_id="right-pane"):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.Textbox(label="Cora", interactive=False, placeholder="Evaluation responses will appear here...", lines=20)

# Launch the Gradio app
interface.launch(share=True)

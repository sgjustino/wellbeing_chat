import os
import json
import gradio as gr
import requests
import time

# Retrieve the API code from the environment variable
api_code = os.getenv("api_code")

# API endpoint
url = "https://api.corcel.io/v1/text/vision/chat"

# Headers with authorization
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {api_code}"
}

# System prompts
chat_system_prompt = """
You are a helpful and joyous mental therapy assistant named Averie. Always answer as helpfully and cheerfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses 
are socially unbiased and positive in nature. Chat as you would in a natural, friendly conversation. Avoid using bullet points or overly long 
responses. Keep your replies concise and engaging, similar to how you would speak with a friend.
"""
eval_system_prompt = """
You are a trained psychologist named Cora who is examining the interaction between a mental health assistant and someone who is troubled. 
Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. 
Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX
"""

def call_api_streaming(prompt: str):
    payload = {
        "model": "llama-3",
        "temperature": 0.1,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, json=payload, headers=headers, stream=True)
    response_text = ""
    for line in response.iter_lines():
        if line:
            line_json = json.loads(line.decode('utf-8'))
            if "choices" in line_json:
                response_text += line_json["choices"][0]["delta"].get("content", "")
                yield response_text

def chat_fn(user_input, chat_history, eval_history):
    chat_prompt = chat_system_prompt + "\n" + "\n".join(chat_history) + f"\nUser: {user_input}\nAverie: "
    
    chat_history.append(f"User: {user_input}")
    chat_response = ""
    
    # Stream the chat response
    for partial_response in call_api_streaming(chat_prompt):
        chat_response = partial_response
        chat_history[-1] = f"Averie: {chat_response}"
        yield [(parse_codeblock(chat_history[i]), parse_codeblock(chat_history[i + 1])) for i in range(0, len(chat_history), 2)], chat_history, ""
        time.sleep(0.5)  # Adjust sleep time as needed

    chat_history[-1] = f"Averie: {chat_response}"

    eval_prompt = eval_system_prompt + "\n" + "\n".join(chat_history) + f"\nUser: {user_input}"
    eval_response = call_api(chat_prompt)
    
    yield [(parse_codeblock(chat_history[i]), parse_codeblock(chat_history[i + 1])) for i in range(0, len(chat_history), 2)], chat_history, eval_response

def reset_textbox():
    return gr.update(value='')

def parse_codeblock(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if "```" in line:
            if line != "```":
                lines[i] = f'<pre><code class="{lines[i][3:]}">'
            else:
                lines[i] = '</code></pre>'
        else:
            lines[i] = "<br/>" + line.replace("<", "&lt;").replace(">", "&gt;")
    return "".join(lines)

light_mode_js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

title = "Chat with Averie and Evaluation by Cora"
description = "A friendly mental health assistant chatbot and its evaluation by a trained psychologist."

with gr.Blocks(css="style.css", js=light_mode_js) as interface:
    with gr.Tabs():
        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane", scale=1):
                    gr.Markdown("### Chat with Averie")
                    chatbot = gr.Chatbot(elem_id='chatbot')
                    user_input = gr.Textbox(placeholder="Type a message and press enter", label="Your message")
                    state = gr.State([])
                    eval_state = gr.State([])

                    user_input.submit(chat_fn, [user_input, state, eval_state], [chatbot, state, eval_state], stream=True)
                    user_input.submit(reset_textbox, [], [user_input])
                with gr.Column(elem_id="right-pane", scale=1):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.HTML(elem_id="eval-output")
                    eval_state.change(lambda eval_text: eval_text, inputs=[eval_state], outputs=[eval_output])

        with gr.TabItem("About"):
            gr.Markdown("""
            ## About Averie and Cora

            ### Averie
            Averie is your friendly mental health assistant designed to provide supportive conversations. She aims to offer helpful and cheerful responses to improve mental well-being until professional help can be sought. Averie is always ready to listen and provide comfort.

            ### Cora
            Cora is a trained psychologist who evaluates the interactions between Averie and users. She conducts mental health analyses to identify potential issues and likely reasons. Cora provides insights based on the conversations to ensure users receive the best possible support and guidance.

            ## Disclaimer 
            This app is not a substitute for professional mental health treatment. If you are experiencing a mental health crisis or need professional help, please contact a qualified mental health professional.
            """)

# Launch the Gradio app
interface.launch(share=False)

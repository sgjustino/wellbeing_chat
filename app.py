import os
import json
import gradio as gr
from gradio import Block
import requests
import time


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
    choices = response.json()[0].get("choices", [])
    if choices and "delta" in choices[0]:
        return choices[0]["delta"]["content"]
    else:
        return "No valid response received from the API."

def chat_and_evaluate(user_input, chat_history):
    # Update chat history with user input and typing indicator
    chat_history.append(f"User: {user_input}")
    chat_history.append("Averie is typing...")

    # Chat model response
    chat_prompt = f"{chat_system_prompt}\nUser: {user_input}"
    chat_output = call_api(chat_prompt)
    chat_output_words = chat_output.split()
    
    # Evaluation model response
    eval_prompt = f"{eval_system_prompt}\nUser: {user_input}"
    eval_output = call_api(eval_prompt)

    # Update chat history with Averie's response
    chat_history[-1] = f"Averie: {chat_output}"

    # Return updated history and evaluation output
    return chat_output_words, eval_output

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
                    chat_output = gr.Textbox(label="Averie", interactive=False, placeholder="Hi there, I am Averie. How are you today?", lines=20)
                    chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...", lines=1)
                    chat_submit = gr.Button("Submit", elem_id="submit-button", variant="primary")
                    chat_history = gr.State([])
                    
                    def submit_on_enter(event):
                        if event.key == "Enter":
                            return chat_submit.click()
                            
                with gr.Column(elem_id="right-pane"):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.Textbox(label="Cora", interactive=False, placeholder="Evaluation responses will appear here...", lines=20)
                    
                chat_input.submit(chat_submit.click, inputs=[chat_input, chat_history], outputs=[chat_output, eval_output])
                chat_input.change(submit_on_enter)

            def handle_submit(user_input, chat_history):
                # Append user input to chat history
                chat_history.append(f"<div class='user-message'>User: {user_input}</div>")
                chat_history.append("<div class='typing-indicator'>Averie is typing...</div>")
                
                # Process chat and evaluation
                chat_output_words, eval_response = chat_and_evaluate(user_input, chat_history)
                
                # Word-by-word output for Averie's response
                averie_response = ""
                for i in range(len(chat_output_words)):
                    word = chat_output_words[i]
                    averie_response += f" {word}"
                    typing_indicator = "." * ((i % 3) + 1)
                    chat_history[-1] = f"<div class='typing-indicator'>Averie is typing{typing_indicator}</div>"
                    yield "\n".join(chat_history), eval_response
                    time.sleep(0.1)  # Adjust the delay between words as needed
                
                # Final response without typing indicator
                chat_history[-1] = f"<div class='averie-message'>{averie_response.strip()}</div>"
                yield "\n".join(chat_history), eval_response

            chat_submit.click(fn=handle_submit, inputs=[chat_input, chat_history], outputs=[chat_output, eval_output])

# Launch the Gradio app
interface.launch(share=True)
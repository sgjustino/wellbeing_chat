import os
import json
import gradio as gr
from huggingface_hub import InferenceClient

# Retrieve the access token from the environment variable
access_token = os.getenv("access_token")

# Initialize Inference API client for the model
model_id = "unsloth/llama-3-8b-bnb-4bit"
client = InferenceClient(model=model_id, token=access_token, timeout=300)  # Increased timeout to 300 seconds

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX"

def call_llm(client: InferenceClient, prompt: str):
    response = client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]

def chat_and_evaluate(user_input):
    # Chat model response
    chat_prompt = f"{chat_system_prompt}\nUser: {user_input}"
    chat_output = call_llm(client, chat_prompt)

    # Evaluation model response
    eval_prompt = f"{eval_system_prompt}\nAssistant: {chat_output}"
    eval_output = call_llm(client, eval_prompt)
    
    return chat_output, eval_output

# Set up the Gradio interface
with gr.Blocks(css="style.css") as interface:
    with gr.Tabs():
        with gr.TabItem("About"):
            gr.Markdown("""
            ## About
            This is a mental health assistant designed to provide supportive conversations. It aims to offer helpful and cheerful responses to improve mental well-being until professional help can be sought.
            """)
        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane"):
                    chat_output = gr.Textbox(label="Chat", interactive=False, placeholder="Chat responses will appear here...", lines=20)
                    chat_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
                    chat_submit = gr.Button("Submit", elem_id="submit-button")
                with gr.Column(elem_id="right-pane"):
                    eval_output = gr.Textbox(label="Evaluation Response", interactive=False, placeholder="Evaluation responses will appear here...", lines=20)

            chat_submit.click(fn=chat_and_evaluate, inputs=chat_input, outputs=[chat_output, eval_output])

# Launch the Gradio app
interface.launch()

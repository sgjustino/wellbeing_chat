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

def call_api(prompt: str):
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
            line_decoded = line.decode('utf-8').strip()
            if line_decoded.startswith("data: "):
                line_decoded = line_decoded[len("data: "):]  # Remove the "data: " prefix
            if line_decoded:
                try:
                    line_json = json.loads(line_decoded)
                    if "choices" in line_json and line_json["choices"]:
                        delta_content = line_json["choices"][0]["delta"].get("content", "")
                        response_text += delta_content
                        yield response_text
                except json.JSONDecodeError:
                    pass

def chat_fn(message, history):
    chat_prompt = chat_system_prompt + "\n" + "\n".join([f"User: {h[0]}\nAverie: {h[1]}" for h in history]) + f"\nUser: {message}\nAverie: "
    response_generator = call_api(chat_prompt)
    for response in response_generator:
        history.append((message, response))
        yield history, ""

def eval_fn(message, history):
    eval_prompt = eval_system_prompt + "\n" + "\n".join([f"User: {h[0]}\nAverie: {h[1]}" for h in history]) + f"\nUser: {message}"
    eval_response = call_api(eval_prompt)
    eval_text = ""
    for response in eval_response:
        eval_text += response
    return eval_text

demo = gr.ChatInterface(
    chat_fn,
    chatbot=gr.Chatbot(placeholder="Hi, I am Averie. How are you today?"),
    title="Chat with Averie and Evaluation by Cora",
    description="A friendly mental health assistant chatbot and its evaluation by a trained psychologist.",
    theme="soft",
    examples=["Hello", "I am feeling down today", "Can you help me with my anxiety?"],
    additional_inputs=[
        gr.Textbox(placeholder="Type your message here...", container=False),
        gr.Button("Submit", variant="primary"),
    ]
)

demo.launch()

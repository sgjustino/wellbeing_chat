import os
import gradio as gr
import json
from huggingface_hub import InferenceClient

# Retrieve the access token from the environment variable
access_token = os.getenv("access_token")

# Initialize Inference API client for the model
model_id = "unsloth/llama-3-8b-bnb-4bit"
client = InferenceClient(model=model_id, token=access_token, timeout=120)

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
interface = gr.Interface(
    fn=chat_and_evaluate,
    inputs="text",
    outputs=[gr.Textbox(label="Chat Response"), gr.Textbox(label="Evaluation Response")],
    live=True
)

# Launch the Gradio app
interface.launch()

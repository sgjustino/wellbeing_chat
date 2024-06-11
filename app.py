import os
import gradio as gr
from huggingface_hub import InferenceClient

# Retrieve the access token from the environment variable
access_token = os.getenv("access_token")

# Initialize Inference API client for the model
model_id = "unsloth/llama-3-8b-bnb-4bit"
client = InferenceClient(model=model_id, token=access_token)

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX"

def chat_and_evaluate(user_input):
    # Chat model response
    chat_input = {
        "inputs": f"{chat_system_prompt}\nUser: {user_input}",
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.01
        }
    }
    chat_response = client(inputs=chat_input["inputs"], parameters=chat_input["parameters"])
    chat_output = chat_response.get("generated_text", "")

    # Evaluation model response
    eval_input = {
        "inputs": f"{eval_system_prompt}\nAssistant: {chat_output}",
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.01
        }
    }
    eval_response = client(inputs=eval_input["inputs"], parameters=eval_input["parameters"])
    eval_output = eval_response.get("generated_text", "")
    
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
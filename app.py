import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load the chat model using PEFT
chat_model_id = "zementalist/llama-3-8B-chat-psychotherapist"
base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

chat_config = PeftConfig.from_pretrained(chat_model_id)
chat_base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
chat_model = PeftModel.from_pretrained(chat_base_model, chat_model_id)
chat_model.to("cuda")  # Ensure the model runs on GPU

chat_tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Load the evaluation model directly
eval_model_id = "klyang/MentaLLaMA-chat-7B-hf"
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
eval_model = AutoModelForCausalLM.from_pretrained(eval_model_id)
eval_model.to("cuda")  # Ensure the model runs on GPU

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX"

def chat_and_evaluate(user_input):
    # Chat model response
    chat_messages = [
        {"role": "system", "content": chat_system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    chat_input_ids = chat_tokenizer(chat_messages, return_tensors="pt").to(chat_model.device)
    chat_outputs = chat_model.generate(
        chat_input_ids.input_ids,
        max_new_tokens=256,
        eos_token_id=chat_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.01
    )
    chat_response = chat_outputs[0][chat_input_ids.input_ids.shape[-1]:]
    chat_output = chat_tokenizer.decode(chat_response, skip_special_tokens=True)
    
    # Evaluation model response
    eval_messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": chat_output}
    ]
    
    eval_input_ids = eval_tokenizer(eval_messages, return_tensors="pt").to(eval_model.device)
    eval_outputs = eval_model.generate(
        eval_input_ids.input_ids,
        max_new_tokens=256,
        eos_token_id=eval_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.01
    )
    eval_response = eval_outputs[0][eval_input_ids.input_ids.shape[-1]:]
    eval_output = eval_tokenizer.decode(eval_response, skip_special_tokens=True)
    
    return chat_output, eval_output

# Set up the Gradio interface
interface = gr.Interface(
    fn=chat_and_evaluate,
    inputs="text",
    outputs=[gr.outputs.Textbox(label="Chat Response"), gr.outputs.Textbox(label="Evaluation Response")],
    live=True
)

# Launch the Gradio app
interface.launch()
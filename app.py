import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Load the chat model
model_id = "zementalist/llama-3-8B-chat-psychotherapist"
config = PeftConfig.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, model_id)
model.to("cuda")  # Ensure the model runs on GPU

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify potential issues and likely reasons. Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX"

def chat_and_evaluate(user_input):
    # Chat model response
    chat_messages = [
        {"role": "system", "content": chat_system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    chat_input_ids = tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    chat_outputs = model.generate(
        chat_input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.01
    )
    chat_response = chat_outputs[0][chat_input_ids.shape[-1]:]
    chat_output = tokenizer.decode(chat_response, skip_special_tokens=True)
    
    # Evaluation model response
    eval_messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": chat_output}
    ]
    
    eval_input_ids = tokenizer.apply_chat_template(
        eval_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    eval_outputs = model.generate(
        eval_input_ids,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.01
    )
    eval_response = eval_outputs[0][eval_input_ids.shape[-1]:]
    eval_output = tokenizer.decode(eval_response, skip_special_tokens=True)
    
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
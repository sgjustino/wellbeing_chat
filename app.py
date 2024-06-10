import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from huggingface_hub import InferenceClient

# Load the chat model
chat_model_id = "zementalist/llama-3-8B-chat-psychotherapist"
chat_config = PeftConfig.from_pretrained(chat_model_id)
chat_model = AutoModelForCausalLM.from_pretrained(chat_config.base_model_name_or_path)
chat_model = PeftModel.from_pretrained(chat_model, chat_model_id)
chat_model.to("cuda")  # Ensure the model runs on GPU

chat_tokenizer = AutoTokenizer.from_pretrained(chat_config.base_model_name_or_path)

# Load the evaluation model
eval_model_id = 'klyang/MentaLLaMA-chat-7B'
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_id)
eval_model = AutoModelForCausalLM.from_pretrained(eval_model_id, device_map='auto')

# System prompts
chat_system_prompt = "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."
eval_system_prompt = "You are a trained psychologist who is examining the interaction between a mental health assistant and someone who is troubled. Always look at their answers and conduct a mental health analysis to identify possible issues and likely reasons."

def chat_and_evaluate(message, history, system_message, max_tokens, temperature, top_p):
    # Chat model response
    chat_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message}
    ]
    
    chat_input_ids = chat_tokenizer.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(chat_model.device)

    chat_terminators = [
        chat_tokenizer.eos_token_id,
        chat_tokenizer.convert_tokens_to_ids("")
    ]

    chat_outputs = chat_model.generate(
        chat_input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=chat_terminators,
        do_sample=True,
        temperature=temperature
    )
    chat_response = chat_outputs[0][chat_input_ids.shape[-1]:]
    chat_output = chat_tokenizer.decode(chat_response, skip_special_tokens=True)
    
    # Evaluation model response
    eval_messages = [
        {"role": "system", "content": eval_system_prompt},
        {"role": "user", "content": chat_output}
    ]
    
    eval_input_ids = eval_tokenizer.apply_chat_template(
        eval_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(eval_model.device)

    eval_terminators = [
        eval_tokenizer.eos_token_id,
        eval_tokenizer.convert_tokens_to_ids("")
    ]

    eval_outputs = eval_model.generate(
        eval_input_ids,
        max_new_tokens=max_tokens,
        eos_token_id=eval_terminators,
        do_sample=True,
        temperature=temperature
    )
    eval_response = eval_outputs[0][eval_input_ids.shape[-1]:]
    eval_output = eval_tokenizer.decode(eval_response, skip_special_tokens=True)
    
    return chat_output, eval_output

demo = gr.ChatInterface(
    fn=chat_and_evaluate,
    additional_inputs=[
        gr.Textbox(value=chat_system_prompt, label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
    outputs=[
        gr.outputs.Textbox(label="Chat Response"),
        gr.outputs.Textbox(label="Evaluation Response")
    ]
)

if __name__ == "__main__":
    demo.launch()
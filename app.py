import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Load chatbot model
chat_tokenizer = AutoTokenizer.from_pretrained("Sgjustino/futaro_chat")
chat_model = AutoModelForCausalLM.from_pretrained("Sgjustino/futaro_chat", device_map="auto")

# Load evaluation model
eval_tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-7B")
eval_model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-7B", device_map="auto")

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
Format the output as:\nPotential Issues: XXX \nLikely Causes: XXX \nNext steps: XXX.
Only output accordingly to the format, keep it concise and clear and do not output anything extra.
"""

MAX_INPUT_TOKEN_LENGTH = 4096

@spaces.GPU(duration=120)
def generate_response(model, tokenizer, prompt, max_new_tokens=500, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH).input_ids.to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        streamer=streamer,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    return "".join([text for text in streamer])

def chat_fn(user_input, chat_history):
    chat_prompt = chat_system_prompt + "\n" + "\n".join([f"User: {h[0]}\nAverie: {h[1]}" for h in chat_history]) + f"\nUser: {user_input}\nAverie: "
    response = generate_response(chat_model, chat_tokenizer, chat_prompt)
    chat_history.append((user_input, response))
    return chat_history, chat_history

def eval_fn(chat_history):
    eval_prompt = eval_system_prompt + " " + " ".join([f"User: {h[0]} Averie: {h[1]}" for h in chat_history])
    eval_response = generate_response(eval_model, eval_tokenizer, eval_prompt)
    
    # Clean the response
    cleaned_response = eval_response.split("**Analysis**")[-1].strip()
    if "Potential Issues:" in cleaned_response:
        cleaned_response = cleaned_response.split("Potential Issues:")[-1]
        cleaned_response = "Potential Issues:" + cleaned_response

    return cleaned_response

# Gradio interface setup
with gr.Blocks() as interface:
    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot()
            msg = gr.Textbox()
            clear = gr.Button("Clear")

        with gr.TabItem("Evaluation"):
            eval_output = gr.Textbox()
            eval_button = gr.Button("Evaluate Chat")

    msg.submit(chat_fn, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    eval_button.click(eval_fn, [chatbot], [eval_output])

# Launch the Gradio app
interface.launch()
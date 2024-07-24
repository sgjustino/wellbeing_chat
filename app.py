import gradio as gr
import os
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import torch

# Load chatbot model
chat_tokenizer = AutoTokenizer.from_pretrained("Sgjustino/futaro_chat")
chat_model = AutoModelForCausalLM.from_pretrained("Sgjustino/futaro_chat")

# Load evaluation model
eval_tokenizer = AutoTokenizer.from_pretrained("klyang/MentaLLaMA-chat-7B")
eval_model = AutoModelForCausalLM.from_pretrained("klyang/MentaLLaMA-chat-7B")

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

@spaces.GPU
def generate_response(model, tokenizer, prompt, max_new_tokens=500, temperature=0.7):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKEN_LENGTH).input_ids.to('cuda')
    model.to('cuda')
    
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

def reset_textbox():
    return gr.update(value='')

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
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.TabItem("Chat", elem_id="chat-tab"):
            with gr.Row():
                with gr.Column(elem_id="left-pane", scale=1):
                    gr.Markdown("### Chat with Averie")
                    chatbot = gr.Chatbot(placeholder="Hi, I am Averie. How are you today?", elem_id='chatbot')
                    user_input = gr.Textbox(placeholder="Type a message and press enter", label="Your message")
                    state = gr.State([])
                    
                    user_input.submit(chat_fn, [user_input, state], [chatbot, state], queue=True)
                    user_input.submit(reset_textbox, [], [user_input])
                    
                with gr.Column(elem_id="right-pane", scale=1):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.HTML("<p>Click to evaluate the chat.</p>", elem_id="eval-output")
                    eval_button = gr.Button("Evaluate Chat")
                    
                    eval_button.click(eval_fn, [chatbot], [eval_output])

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
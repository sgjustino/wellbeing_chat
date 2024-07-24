import gradio as gr
import os
from groq import Groq

# Create the Groq client
client = Groq(api_key=os.environ.get("groq_api"))

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

def generate_response(prompt, system_prompt, max_tokens=500, temperature=0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.choices[0].message.content

def chat_fn(user_input, chat_history):
    chat_prompt = "\n".join([f"User: {h[0]}\nAverie: {h[1]}" for h in chat_history]) + f"\nUser: {user_input}\nAverie: "
    response = generate_response(chat_prompt, chat_system_prompt)
    chat_history.append((user_input, response))
    return chat_history

def eval_fn(chat_history):
    eval_prompt = " ".join([f"User: {h[0]} Averie: {h[1]}" for h in chat_history])
    eval_response = generate_response(eval_prompt, eval_system_prompt)
    return eval_response

title = "Chat with Averie and Evaluation by Cora"
description = "A friendly mental health assistant chatbot and its evaluation by a trained psychologist."

with gr.Blocks() as interface:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Tabs():
        with gr.TabItem("Chat"):
            chatbot = gr.Chatbot(label="Chat with Averie")
            user_input = gr.Textbox(label="Your message")
            send_button = gr.Button("Send")

        with gr.TabItem("Evaluation"):
            eval_output = gr.Textbox(label="Evaluation by Cora")
            eval_button = gr.Button("Evaluate Chat")

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

    send_button.click(chat_fn, inputs=[user_input, chatbot], outputs=chatbot)
    eval_button.click(eval_fn, inputs=[chatbot], outputs=eval_output)

# Launch the Gradio app
interface.launch()
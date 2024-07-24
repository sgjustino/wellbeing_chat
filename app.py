import os
import gradio as gr
from groq import Groq

# Create the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat_fn(user_input, chat_history):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant named Averie. You are a mental health assistant who provides supportive conversations. You reply with helpful and cheerful responses."
        }
    ]
    
    # Add conversation history
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    # Add the new user message
    messages.append({"role": "user", "content": user_input})

    stream = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=True,
    )

    assistant_message = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            assistant_message += chunk.choices[0].delta.content
            yield chat_history + [(user_input, assistant_message)]

    chat_history.append((user_input, assistant_message))
    return chat_history

def eval_fn(chat_history):
    messages = [
        {
            "role": "system",
            "content": "You are a trained psychologist named Cora. Analyze the following conversation and provide a brief mental health analysis. Format the output as: Potential Issues: XXX | Likely Causes: XXX | Next steps: XXX."
        }
    ]
    
    # Add entire conversation history
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    stream = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=True,
    )

    evaluation = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            evaluation += chunk.choices[0].delta.content
            yield evaluation

    # Ensure the output follows the specified format
    formatted_output = "Potential Issues: | Likely Causes: | Next steps:"
    parts = evaluation.split("|")
    if len(parts) == 3:
        formatted_output = f"Potential Issues: {parts[0].split(':')[-1].strip()} | Likely Causes: {parts[1].split(':')[-1].strip()} | Next steps: {parts[2].split(':')[-1].strip()}"

    return formatted_output

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
                    send_button = gr.Button("Send")
                
                with gr.Column(elem_id="right-pane", scale=1):
                    gr.Markdown("### Evaluation by Cora")
                    eval_output = gr.HTML("<p>Click to evaluate the chat.</p>", elem_id="eval-output")
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
    user_input.submit(chat_fn, inputs=[user_input, chatbot], outputs=chatbot)
    user_input.submit(reset_textbox, [], [user_input])
    eval_button.click(eval_fn, inputs=[chatbot], outputs=eval_output)

# Launch the Gradio app
interface.launch(share=False)
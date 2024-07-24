import os
import gradio as gr
from groq import Groq

# Create the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Set the system prompt
system_prompt = {
    "role": "system",
    "content": "You are a helpful assistant named Averie. You are a mental health assistant who provides supportive conversations. You reply with helpful and cheerful responses."
}

# Initialize the chat history
chat_history = [system_prompt]

def chat_fn(user_input, history):
    # Append the user input to the chat history
    chat_history.append({"role": "user", "content": user_input})
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=chat_history,
        max_tokens=100,
        temperature=0.7
    )
    
    # Append the response to the chat history
    assistant_message = response.choices[0].message.content
    chat_history.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    # Update the visible chat history for Gradio
    history.append((user_input, assistant_message))
    return history

def eval_fn(history):
    eval_prompt = {
        "role": "system",
        "content": "You are a trained psychologist named Cora. Analyze the following conversation and provide a brief mental health analysis. Format the output as: Potential Issues: XXX | Likely Causes: XXX | Next steps: XXX."
    }
    
    eval_messages = [eval_prompt] + [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} for i, msg in enumerate(sum(history, ()))]
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=eval_messages,
        max_tokens=200,
        temperature=0.5
    )
    
    return response.choices[0].message.content

# Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Chat with Averie and Evaluation by Cora")
    gr.Markdown("A friendly mental health assistant chatbot and its evaluation by a trained psychologist.")
    
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
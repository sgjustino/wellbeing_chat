import os
import json
import gradio as gr
import re
from groq import Groq

# Create the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat_fn(user_input, chat_history, next_question=""):
    messages = [
        {
            "role": "system",
            "content": f"""You are Averie, a supportive mental health assistant. Respond in a friendly, conversational manner. 
            Keep your responses brief, ideally 2-3 sentences. If provided, incorporate this question naturally: {next_question}
            Focus on one main point or question at a time. Be empathetic but not overly wordy."""
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
        model="llama3-70b-8192",
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
        else:
            yield chat_history + [(user_input, assistant_message)]

    chat_history.append((user_input, assistant_message))
    return chat_history

def eval_fn(chat_history):
    
    messages = [
        {
            "role": "system",
            "content": """You are a trained psychologist named Cora. Analyze the following conversation and provide a brief mental health analysis. 
            Format your response exactly as follows:
            Potential Issues: [List issues here, separated by commas]
            Likely Causes: [List causes here, separated by commas]
            Follow-up for Conversation: [One important follow-up question to assist in the mental health analysis]
            Keep each section brief and concise."""
        }
    ]
    
    # Add entire conversation history
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
        
    response = client.chat.completions.create(
        messages=messages,
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )
    content = response.choices[0].message.content
    # Use regex to extract the sections
    potential_issues = re.search(r'Potential Issues:(.*?)(?:Likely Causes:|$)', content, re.DOTALL)
    likely_causes = re.search(r'Likely Causes:(.*?)(?:Follow-up for Conversation:|$)', content, re.DOTALL)
    follow_up = re.search(r'Follow-up for Conversation:(.*?)$', content, re.DOTALL)
    # Format the output
    formatted_output = "Potential Issues: {} | Likely Causes: {} | Follow-up for Conversation: {}".format(
        potential_issues.group(1).strip() if potential_issues else "N/A",
        likely_causes.group(1).strip() if likely_causes else "N/A",
        follow_up.group(1).strip() if follow_up else "N/A"
    )
    return formatted_output, follow_up.group(1).strip() if follow_up else ""

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
import os
import json
import gradio as gr
import re
from groq import Groq

# Create the Groq client
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

def chat_fn(user_input, chat_history, follow_up_question=""):
    messages = [
        {
            "role": "system",
            "content": f"""You are Averie, a supportive mental health assistant. Respond in a friendly, conversational manner. 
            Keep your responses helpful but not overwhelming, ideally 3 sentences and not exceeding 5 sentences. If provided, incorporate this question naturally: {follow_up_question}
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
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=500,
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
    max_history_length = 5  # Maximum number of user-assistant pairs to keep in history
    if len(chat_history) > max_history_length:
        chat_history = chat_history[-max_history_length:]

    messages = [
        {
            "role": "system",
            "content": """You are a trained psychologist named Cora. Analyze the following conversation and provide a brief mental health analysis. 
            Format your response EXACTLY as follows and keep each section brief and concise:
            Potential Issues: [List issues here, separated by commas]
            Likely Causes: [List causes here, separated by commas]
            Follow-up Areas: [List follow-up queries very concisely in a few words to assist in the mental health analysis]
            """
        }
    ]
    
    # Add entire conversation history
    for user_msg, assistant_msg in chat_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    
    response = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=600,
        top_p=1,
        stream=False,
    )
    content = response.choices[0].message.content
    
    # Use regex to extract the sections
    potential_issues = re.search(r'Potential Issues:(.*?)(?:Likely Causes:|$)', content, re.DOTALL)
    likely_causes = re.search(r'Likely Causes:(.*?)(?:Follow-up Areas:|$)', content, re.DOTALL)
    follow_up = re.search(r'Follow-up Areas:(.*?)$', content, re.DOTALL)
    
    formatted_output = """
    <div style="text-align:left;">
        <p style="margin: 0;"><strong>Potential Issues:</strong> {}</p>
        <p style="margin: 0;"><strong>Likely Causes:</strong> {}</p>
        <p style="margin: 0;"><strong>Follow-up Areas:</strong> {}</p>
    </div>
    """.format(
        potential_issues.group(1).strip() if potential_issues else "N/A",
        likely_causes.group(1).strip() if likely_causes else "N/A",
        follow_up.group(1).strip() if follow_up else "N/A"
    )
    
    return formatted_output, follow_up.group(1).strip() if follow_up else "N/A"



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

title = "An MVP Project to Integrate a Wellbeing Chatbot with a Conversation Analysis LLM for Counselling Insights"
description= ""

with gr.Blocks(css="style.css", js=light_mode_js) as interface:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Tabs():

        with gr.TabItem("About"):
            gr.Markdown("""
# **Background**
This project showcases a proof-of-concept where two LLMs work together: Averie, a friendly wellbeing chatbot, assists users by providing supportive conversations, while Cora, a separate LLM, analyzes the conversations to identify potential issues, likely causes, and suggest follow-up queries to improve the interactions. Both LLMs are based on the LLaMA 3 7B model via the Groq.com API. 
# Areas For Improvement
1) **Training with Actual Counselling Transcripts for Wellbeing Chatbot**: The LLMs can be fine-tuned with transcripts from real-life therapy sessions conducted by qualified psychologists. Post-training, validation tests should be conducted to ensure the chatbot's responses are rated as suitable for deployment by qualified psychologists.
2) **Integrating Validated Mental Health Questionnaires for Conversation Analysis LLM**: Incorporating validated mental health questionnaires (e.g. PHQ-9 and GAD-7)) into the LLM can enable the system to prompt relevant questions to identify issues and concerns or determine if there is a need to escalate the situation. The MentaLLaMA project from the [Interpretable Mental Health Instruction (IMHI)](https://arxiv.org/abs/2309.13567) paper is an example of this approach, enabling interpretable mental health analysis on social media.
3) **Ensuring Adherence to Medical Protocols**: It is essential that the system follows a medical protocol that escalates the situation if the user appears to be harming themselves or require immediate medical attention. These instructions should be embedded within the mental health analysis LLM (Cora) to ensure appropriate and timely actions are taken.
# **Disclaimer**
This app is not a substitute for professional mental health treatment. If you are experiencing a mental health crisis or need professional help, please contact a qualified mental health professional.
""")

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
                    follow_up_input = gr.Textbox(label="Follow-up Question for Averie", interactive=False)
                    eval_button = gr.Button("Evaluate Chat")
                    
    send_button.click(chat_fn, inputs=[user_input, chatbot, follow_up_input], outputs=chatbot)
    user_input.submit(chat_fn, inputs=[user_input, chatbot, follow_up_input], outputs=chatbot)
    user_input.submit(reset_textbox, [], [user_input])
    eval_button.click(eval_fn, inputs=[chatbot], outputs=[eval_output, follow_up_input])

# Launch the Gradio app
interface.launch(share=False)
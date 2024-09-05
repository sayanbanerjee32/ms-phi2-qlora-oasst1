from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import gradio as gr
import time

REPO_ID = "sayanbanerjee32/ms-phi2-qlora-oasst1"

model = AutoModelForCausalLM.from_pretrained(REPO_ID)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

def generate_text(prompt, chat_history, num_new_tokens = 100):
    # prompt = "<|prompter|>What is 2 + 2?<|endoftext|><|assistant|>"  # change to your desired prompt
    
    
    input_prompt = ''
    if len(chat_history) > 0:
        input_prompt += "<|prompter|>" + chat_history[-1][0] + "<|endoftext|><|assistant|>" + chat_history[-1][1] + "<|endoftext|>"
    input_prompt += "<|prompter|>" + prompt + "<|endoftext|><|assistant|>"
    # Count the number of tokens in the prompt  
    num_prompt_tokens = len(tokenizer(input_prompt)['input_ids'])
    # Calculate the maximum length for the generation
    max_length = num_prompt_tokens + num_new_tokens
    gen = pipeline('text-generation', model=model,
                tokenizer=tokenizer, max_length= max_length )
    result = gen(prompt)
    return result[0]['generated_text'].replace(prompt, '')

with gr.Blocks() as demo:
    gr.HTML("<h1 align = 'center'> Chat </h1>")
    gr.HTML("<h4 align = 'center'> ChatBot powered by Microsoft-Phi-2 finetuned on OpenAssistant dataset</h4>")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    gr.Examples(["What do you think about ChatGPT?",
                "How would the Future of AI in 10 Years look?",
                "Write a announcement tweet for medium.com readers about the new blogpost on 'Open Assistant is open source ChatGPT that you don\'t wanna miss out'",
                "Please implement the Timsort algorithm on Lean 4 and explain your code",
                "How do I build a PC?"], 
                inputs = msg)
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = generate_text(message, chat_history)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# # for collab
# demo.launch(debug=True) 

if __name__ == '__main__':
    demo.launch() 

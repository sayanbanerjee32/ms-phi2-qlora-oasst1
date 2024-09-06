# Fine tuning Microsoft Phi-2 using QLoRA

## Objectives
1. Use dataset: [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1?row=0)
2. Fine-tune [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
3. Use QLoRA strategy
4. Upload to Hugging Face Spaces

## Solution

### Reference notebooks
1. [QLoRA strategy](https://github.com/mshumer/gpt-llm-trainer/blob/main/One_Prompt___Fine_Tuned_LLaMA_2.ipynb)
2. [Data pre-processing](https://github.com/LAION-AI/Open-Assistant/blob/main/notebooks/openassistant-oasst1/getting-started.ipynb)
3. [Data pre-processing](https://huggingface.co/dfurman/Falcon-7B-Chat-v0.1/blob/main/finetune_falcon7b_oasst1_with_bnb_peft.ipynb)

### Dataset

The [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1?row=0) dataset is a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 fully annotated conversation trees. The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.

#### Pre-processing steps
- all linked messages in a single conversation are merged together using the message tree id
- message from the prompter / user and assistant / bot is prefixed using w special token based on the role associated with the message.
- if there are multiple responses against a single message then only one message is selected based on lowest rank.
- This generated ~9k conversations for training and ~500 conversations for validation  

One example conversation will look like the following  
`<|prompter|>Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.<|endoftext|>
<|assistant|>"Monopsony" refers to a market structure where there is only one buyer for a particular good or service. In economics, this term is particularly relevant in the labor market, where a monopsony employer has significant power over the wages and working conditions of their employees. The presence of a monopsony can result in lower wages and reduced employment opportunities for workers, as the employer has little incentive to increase wages or provide better working conditions.  Recent research has identified potential monopsonies in industries such as retail and fast food, where a few large companies control a significant portion of the market (Bivens & Mishel, 2013). In these industries, workers often face low wages, limited benefits, and reduced bargaining power, leading to a situation where they are dependent on the employer for their livelihood. This dependence can result in further suppression of wages and a decline in working conditions.  Overall, the concept of monopsony is essential to understanding the dynamics of labor markets and the impact of market power on workers. Further research is needed to understand the extent and impact of monopsonies on the economy and to develop policies to address this issue.  References: Bivens, J., & Mishel, L. (2013). The Pay of Corporate Executives and Financial Professionals as Evidence of Rents in Top 1 Percent Incomes. Journal of Economic Perspectives, 27(3), 57-78.<|endoftext|>
<|prompter|>Now explain it to a dog<|endoftext|>`

### Training Notebook

Training notebook is available [here](https://github.com/sayanbanerjee32/ms-phi2-qlora-oasst1/blob/main/ms_phi2_fine_tune_q_lora.ipynb)
Fine-tuned model is uploaded out [HuggingFace modle Hub](https://huggingface.co/sayanbanerjee32/ms-phi2-qlora-oasst1)

#### Q LoRA configurations
- LoRA rank: 32
- Layers that got fine tuned - w, q, k, v projection layers and fully connected layers
- 

#### Training Time
- It took around 13 hours to train on T4 GPU in collab with 15GB GPU RAM
- Trainable params: 26214400 || all params: 1547607040 || trainable: 1.69%


### HuggingFace App

The app is running [here](https://huggingface.co/spaces/sayanbanerjee32/askme-ms-phi2-qlora-oasst1)

This app is a chatbot app. Last pair of messages from user and bot passed along with the current message from the user as context to the model for prediction.

See scheen shots below. 
After receiving the response against the first message  
![image](https://github.com/user-attachments/assets/beba3c59-460f-41cf-873d-8caec568c365)

After receiving the response againt a follow-up message
![image](https://github.com/user-attachments/assets/fca79952-f39c-4b89-aaf0-b50b7ceac68b)


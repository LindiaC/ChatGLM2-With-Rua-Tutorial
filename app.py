import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import gradio as gr

#These texts support HTML and Markdown
title = "👸ChatGLM2 with Rua"
description = "我拥有Rua17岁至20岁的记忆。<b>注意</b>，比起ChatGPT类项目，我无法帮你完成任务，甚至有时回答不出你的简单问题。我能做出的反馈更像是：在你提供的语境下，真正的Rua会说什么？<br>训练所使用的数据全部来源于Rua自己发出的消息，信息经过清洗和脱敏。任何涉及个人信息的回答均为模型自己编的，<b>别信</b>。<br>加载可能会比较慢，点击下面的例子可能需要几秒才会显示在input中，点一次就可以啦不要一直戳。<br>一个对话的加载时间需要<b>1~2分钟</b>，是硬件原因，请耐心等待拜托啦。如果你愿意也可以给Rua充钱让她升级仓库配置。"
examples = [["你晚上想吃什么"],["你在干什么"],["什么时候出去玩"]] #Those options can be clicked directly to input on the web page for players 

#Below are the same as testing on kaggle
model_path = "THUDM/chatglm2-6b-int4"
prefix_state_dict = "./{your-pytorch-model}.bin" #make sure the ./ is added!!!

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

prefix_state_dict = torch.load(prefix_state_dict,map_location=torch.device('cpu'))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

model = model.quantize(4)
model = model.float()
model.transformer.prefix_encoder.float()

model = model.eval()


def predict(input, state=[]):
    response, dialog = model.chat(tokenizer,input, history=[])
    print(response, dialog)
    history = state + dialog #this is to ensure the chat history will be displayed
    return history, history


gr.Interface(
    fn=predict,
    title=title,
    description=description,
    examples=examples,
    inputs=["text", "state"],
    outputs=["chatbot","state"], # adding state is to ensure the history can be passed to the next round
    theme="ParityError/Anime", #choose any theme you want on https://huggingface.co/spaces/gradio/theme-gallery
).launch()
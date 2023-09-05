# [chinese-medical-llama2-学习版](https://https://github.com/jialinzhang/chinese-medical-llama2)

<p align='center'>
    <br>
    <img src='./pic/羊驼男孩.png' width=300 height=300>
    <br>
</p>

本项目基于LLaMA2在中文医疗数据集上进行二次预训练、指令微调、RLHF和模型量化，旨在梳理和学习大模型相关的技术。

**本项目主要内容：**

- 🚀 基于Llama2在[中文医疗数据集](https://huggingface.co/datasets/shibing624/medical/viewer/pretrain/train)上进行二次预训练PT
- 🚀 基于Llama2在[中文医疗指令数据集](https://huggingface.co/datasets/shibing624/medical/viewer/finetune/train)上进行指令微调SFT
- 🚀 基于Llama2在[中文医疗偏好数据集](https://huggingface.co/datasets/shibing624/medical/viewer/reward/train)上进行RLHF
- 🚀 对微调后的医疗大模型进行量化

**本项目亮点**

- 🌹 相比[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)在训练时，直接利用transformers库封装的Trainer，我们重写了Trainer，便于自定义训练流程和添加模型监控信息
- 🌹 在二次预训练阶段，我们实现了三种不同方式的训练脚本：使用DDP和AMP来训练embed_tokens layer 和 lm_head layer；使用DDP和AMP来训练embed_tokens layer 、lm_head layer 和 lora layer；使用Deepspeed框架训练embed_tokens layer 、lm_head layer 和 lora layer
- 🌹 我们记录下了在训练过程中踩到的一些坑，以供后来者参考


**相关资源推荐**

- ✈️ 中英文医疗数据集，请移步[Hugging Face](https://huggingface.co/datasets/shibing624/medical)
- ✈️ 中文医疗大模型，请移步[MedicalGPT](https://github.com/shibing624/MedicalGPT)
- ✈️ 中文大模型，请移步[Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

**关于我们｜微信公众号：NLP Journey**

<p align='center'>
    <br>
    <img src='./pic/nlp-journey-wechat-account.jpg' width=300 height=300>
    <br>
</p>

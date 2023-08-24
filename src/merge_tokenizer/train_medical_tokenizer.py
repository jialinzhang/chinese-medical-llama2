#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: train_medical_tokenizer.py
# @Instruction: 基于shibing624/medical中文医疗数据集，训练中文tokenizer
# @CreateTime: 2023/08/13 14:41:45
# @WeChat: damo894127201
# @WeChat Official Accounts: NLP Journey
# @Huggingface Organizations: NLP Journey
# @Github: jialinzhang

import datasets
import sentencepiece as spm
import re

# 加载预训练数据集
base_dir = '../../data/shibing624/medical/'
data_files = {
    'train': base_dir + 'pretrain' + '/train_encyclopedia.json',
    'test': base_dir + 'pretrain' + '/test_encyclopedia.json',
    'validation': base_dir + 'pretrain' + '/valid_encyclopedia.json',
}
data = datasets.load_dataset('json', data_files=data_files)
# 构建训练数据
with open('./train.txt', 'w', encoding='utf-8') as fo:
    count = 0
    for data_type in ['train', 'validation', 'test']:
        for item in data[data_type]:
            # 切割句子，并保留分隔符
            sentences = re.split(r'([\?？。!！;；])', str(item['text']))
            # 将分隔符保留在被分割的上一个句子中
            sentences = ["".join(s) for s in zip(sentences[0::2], sentences[1::2])]
            for sentence in sentences:
                # 舍弃单个字符
                if len(sentence) < 2:
                    continue
                if count % 30000 == 0:
                    print(sentence)
                    print()
                fo.write(sentence.strip())
                fo.write('\n')
                count += 1
print("训练集构建完毕!")

# 开始训练
train_config = '--input=./train.txt --model_prefix=zh_medical_tokenizer\
                --vocab_size=30000 --character_coverage=0.9995\
                --max_sentence_length=6000 --model_type=bpe --num_threads=5'
spm.SentencePieceTrainer.Train(train_config)
print("训练完毕!")
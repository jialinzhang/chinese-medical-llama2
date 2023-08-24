#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: merge_tokenizer.py
# @Instruction: 合并LlaMA2分词器和中文医疗分词器
# @CreateTime: 2023/08/15 10:44:11
# @WeChat: damo894127201
# @WeChat Official Accounts: NLP Journey
# @Huggingface Organizations: NLP Journey
# @Github: jialinzhang

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]='python'
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2
import sentencepiece as spm
import argparse

# huggingface access token for llama model
hf_access_token = os.getenv('HF_ACCESS_TOKEN')
# cli 参数
parser = argparse.ArgumentParser()
parser.add_argument('--llama_tokenizer_dir', default=None, type=str, required=True, help='llama tokenizer dir')
parser.add_argument('--chinese_medical_spm_file', default=None, type=str, required=True, help='chinese medical tokenizer model dir')
parser.add_argument('--save_path', default=None, type=str, required=True, help='merged tokenizer local save path')
args = parser.parse_args()

# parser 
llama_tokenizer_dir = args.llama_tokenizer_dir
chinese_medical_spm_file = args.chinese_medical_spm_file
save_path = args.save_path

# 加载分词模型
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir, use_auth_token=hf_access_token)
chinese_medical_sp_model = spm.SentencePieceProcessor()
chinese_medical_sp_model.Load(chinese_medical_spm_file)

# spm model
llama_spm = sentencepiece_model_pb2.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
# 保存llama原始词表
with open(save_path+'/raw_llama.vocab', 'w', encoding='utf-8') as fi:
    for p in llama_spm.pieces:
        fi.write(f'{p.piece}\t{p.score}\n')
chinese_medical_spm = sentencepiece_model_pb2.ModelProto()
chinese_medical_spm.ParseFromString(chinese_medical_sp_model.serialized_model_proto())

# print number of tokens
print(f"llama tokenizer vocab size: {len(llama_tokenizer)}")
print(f"chinese medical tokenizer vocab size: {len(chinese_medical_sp_model)}")
print(f"llama tokenizer speical token: {llama_tokenizer.all_special_tokens}")
print(f"llama tokenizer speical token id: {llama_tokenizer.all_special_ids}")
print(f"llama tokenizer speical token map: {llama_tokenizer.special_tokens_map}")

# 分词器合并：将中文医疗分词器的piece添加进llama tokenizer
llama_tokens = set(p.piece for p in llama_spm.pieces)
print(f"合并前llama tokens num: {len(llama_tokens)}")
for p in chinese_medical_spm.pieces:
    piece = p.piece
    if piece not in llama_tokens:
        new_p = sentencepiece_model_pb2.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"合并后llama tokens num: {len(llama_spm.pieces)}")

# 保存
output_sp_dir = save_path + '/sp_dir'
output_hf_dir = save_path + '/hf_dir'
os.makedirs(output_sp_dir, exist_ok=True)
os.makedirs(output_hf_dir, exist_ok=True)
with open(output_sp_dir+'/chinese_medical_llama.model', 'wb') as fi:
    fi.write(llama_spm.SerializeToString())
with open(output_sp_dir+'/chinese_medical_llama.vocab', 'w', encoding='utf-8') as fi:
    for p in llama_spm.pieces:
        fi.write(f'{p.piece}\t{p.score}\n')
    
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+'/chinese_medical_llama.model')
tokenizer.save_pretrained(output_hf_dir)
print(f"chinese-medical-llama tokenizer has been saved to {output_hf_dir}")

# 测试
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir, use_auth_token=hf_access_token)
chinese_medical_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(f"chinese-medical-llama tokenizer speical token: {chinese_medical_llama_tokenizer.all_special_tokens}")
print(f"chinese-medical-llama tokenizer speical token id: {chinese_medical_llama_tokenizer.all_special_ids}")
print(f"chinese-medical-llama tokenizer speical token map: {chinese_medical_llama_tokenizer.special_tokens_map}")

text = '卵巢肌瘤会有什么症状？卵巢肌瘤是妇科常见病之一'
print(f"Test data: {text}")
print(f"Tokenized by LlaMA tokenizer: {llama_tokenizer.tokenize(text)}")
print(f"Tokenized by chinese-medial-LlaMA tokenizer: {chinese_medical_llama_tokenizer.tokenize(text)}")
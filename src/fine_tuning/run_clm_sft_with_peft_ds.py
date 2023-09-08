#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: run_clm_sft_with_peft_ds.py
# @CreateTime: 2023/08/25 14:38:22
# @WeChat: damo894127201
# @WeChat Official Accounts: NLP Journey
# @Huggingface Organizations: NLP Journey
# @Github: jialinzhang
# @Instruction: 对LlaMA2-7b进行指令微调：训练embedding和lm_head层，以及lora，并冻结其它参数
'''
一、模块结构:
    1、命令行参数类
        1.1 数据相关参数
        1.2 模型相关参数
        1.3 训练相关参数
    2、辅助函数
        2.1 指标计算函数
        2.2 tokenizer加载函数
    3、数据预处理
    4、数据迭代器
    5、模型加载
    6、自定义训练器
二、训练类型: 单机多卡_单进程单卡(模型可加载至单卡中) llama2-7b float32占28GB float16占14GB
三、训练技术:
    1、数据并行DDP
    2、混合精度AMP
    3、Lora微调
    4、deepspeed
四、模型监控: deepspeed集成的wandb
'''
import numpy as np
import datasets
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.distributed_c10d import ReduceOp
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    set_seed
)
from peft import get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM, PeftModel
import deepspeed
from sklearn.metrics import accuracy_score

import sys
import os
import math
import shutil
import json
from dataclasses import dataclass, field
from contextlib import contextmanager
from itertools import chain
from typing import Optional, Dict, Tuple, List, Callable
from tqdm import trange
import logging


# Step1: 命令行参数设置
@dataclass
class DataArguments:
    """ 
    与数据集相关的参数
    """
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": ("huggingface数据集路径")}
    )
    
    dataset_subset_name: Optional[str] = field(
        default=None, metadata={"help": ("huggingface数据集的子数据集名, 如果存在的话")}
    )
    
    data_file_type: Optional[str] = field(
        default=None, metadata={"help": ("数据文件的类型, 可选: json/txt/csv, 训练数据和测试数据类型应一致")}
    )
    
    train_file_path: Optional[str] = field(default=None, metadata={"help": ("训练集文件路径")})

    eval_file_path: Optional[str] = field(default=None, metadata={"help": ("验证集文件路径")})

    max_train_samples: Optional[int] = field(
        default=20, metadata={"help": ("通过截断训练集，来控制参与训练的数据量，常用于调试或快速训练")}
    )

    max_eval_samples: Optional[int] = field(
        default=10, metadata={"help": ("通过截断验证集，来控制参与验证的数据量，常用于调试或快速训练")}
    )

    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "词元化后(tokenization)的输入序列的最大长度"
                "默认取值为模型最大支持的单个序列长度(特殊符号也涵盖)"
            )
        },
    )
    
    overwrite_cache_processed_dataset: Optional[bool] = field(
        default=False, metadata={"help": ("是否覆盖缓存的预处理后的训练和评估集")}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=5, metadata={"help": ("用于数据集预处理的进程数")}
    )

    processed_data_cache_dir: Optional[str] = field(
        default="./processed", metadata={"help": ("预处理后的数据集存储路径")}
    )

@dataclass
class ModelArguments:
    """ 
    与模型相关的参数
    """
    
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("模型的huggingface路径或本地存储路径")},
    )

    model_revision: Optional[str] = field(
        default="main",
        metadata={
            "help": (
                "指定模型加载版本。由于在huggingface.co上是基于git系统来存储模型和其它组件，故可用Git分支名、tag 或者 commit id来标识"
            )
        },
    )

    cache_dir: Optional[str] = field(
        default='./',
        metadata={
            "help": (
                "从huggingface官网下载的模型本地存储路径。如果未设置，模型会存储在：~/.cache/huggingface/transformers/"
            )
        },
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": ("模型分词器的huggingface名称或本地存储路径。")}
    )

    use_fast_tokenizer: bool = field(
        default=True, metadata={"help": ("是否开启fast tokenizer版本")}
    )

    use_auth_token: Optional[str] = field(
        default=None,
        metadata={"help": ("如果需要访问huggingface私有模型时，请设置huggingface access token")},
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "覆盖默认的配置信息，格式为：n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": ("覆盖默认的“torch.dtype”并在此dtype下加载模型。如果传递“auto”,则“dtype“将自动从模型的权重中得出"),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    
@dataclass
class MyTrainingArguments(TrainingArguments):
    """ 
    与模型训练有关的参数
    
    1、learning_rate: 学习率
    2、weight_decay: 权重衰减
    3、adam_beta1: AdamW参数
    4、adam_beta2: AdamW参数
    5、adam_epsilon: AdamW参数
    6、max_grad_norm: 梯度最大范数,用于梯度裁剪
    7、num_train_epochs: 训练epoch数
    8、max_steps: 最大训练步数,会覆盖掉num_train_epochs
    9、lr_scheduler_type: 学习率调度策略类型,可选[linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt]
    10、warmup_ratio: 学习率热身步占总训练步的比率
    11、warmup_steps: 学习率热身步数, 会覆盖warmup_ratio
    12、local_rank: 当前设备的本地编号,是torch.distributed.launch的环境变量
    13、output_dir: 模型预测结果和checkpoints保存路径
    14、overwrite_output_dir: 是否覆盖掉output_dir路径中的内容
    15、do_train: 是否进行训练
    16、do_eval: 是否进行评估
    17、do_predict: 是否进行预测
    18、_n_gpu: GPU数
    19、n_gpu: 通过_n_gpu设定
    20、per_device_train_batch_size: 训练期间,每个GPU上的batch_size
    21、train_batch_size: 训练期间所有GPU上batch_size之和,无需设置,通过per_device_train_batch_size*n_gpu自动计算
    22、per_device_eval_batch_size: 训练期间,每个GPU上的batch_size
    23、eval_batch_size: 评估期间所有GPU上batch_size之和,无需设置,通过per_device_eval_batch_size*n_gpu自动计算
    24、gradient_accumulation_steps: 梯度累积步数
    25、save_steps: 每多少个step保存模型
    26、resume_from_checkpoint: 从断点加载模型
    ....
    
    与微调模型相关的参数
    
    """
    debug_mode: Optional[bool] = field(default=False, metadata={"help": ("是否调试模式")})
    steps_update: Optional[int] = field(default=0, metadata={"help": "参数已更新次数, 用于加载checkpoint中的step"})
    epochs_run: Optional[int] = field(default=0, metadata={"help": "已训练过的epoch次数, 用于加载checkpoint中的epoch"})
    # Lora
    lora_rank: Optional[int] = field(default=8, metadata={"help": "Lora矩阵秩"})
    lora_alpha: Optional[float] = field(default=32., metadata={"help": " The alpha parameter for Lora scaling, real value = lora_alpha / r"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout probability for Lora layers"})
    lora_trainable: Optional[str] = field(default="q_proj,v_proj", metadata={"help": "The names of the modules to apply Lora to"})
    modules_to_save: Optional[str] = field(default="embed_tokens,lm_head", metadata={"help": "modules apart from LoRA layers to be set as trainable and saved in the final checkpoint"})
    # deepspeed
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "deepspeed config json path"})

# Step2: 辅助函数
def metric(y_pred, y_true, normalize=True, sample_weight=None):
    # y_pred: [batch_size*seq_len,]
    # y_true: [batch_size*seq_len,]
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
        )
    }

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray):
    # y_pred have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the y_true
    # y_pred: [batch_size, seq_len]
    # y_true: [batch_size, seq_len]
    y_true = y_true[:, 1:].reshape(-1)
    y_pred = y_pred[:, :-1].reshape(-1)
    return metric(y_pred=y_pred, y_true=y_true)        
 
def load_tokenizer(modelArguments: ModelArguments,
                   trainingArguments: MyTrainingArguments,
                   logger: logging.RootLogger,
                   logger_name: str) -> LlamaTokenizer:
    tokenizer_kwargs = {
        "cache_dir": modelArguments.cache_dir,
        "use_fast": modelArguments.use_fast_tokenizer,
        "revision": modelArguments.model_revision,
        "use_auth_token": modelArguments.use_auth_token,
    }
    
    if modelArguments.tokenizer_name_or_path:
        logger.info(f"Local Rank: {trainingArguments.local_rank} {logger_name} 分词器加载成功......")
        return LlamaTokenizer.from_pretrained(modelArguments.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(f"Local Rank: {trainingArguments.local_rank} {logger_name} 请配置分词器名称或路径: tokenizer_name_or_path")

@contextmanager
def torch_distributed_zero_first(rank: int):
    """ Decorator to make all processes in distributed training wait for each local_master to do something """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if rank == 0:
        torch.distributed.barrier()

def deepspeed_setup():
    deepspeed.init_distributed(dist_backend="nccl", init_method="env://")

def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fi:
        return json.load(fi)

IGNORE_INDEX = 100 # padding id 交叉墒函数在计算损失时会忽略掉该位置的logit

PROMPT_TEMPLATE = (
    "下面是一条说明任务的指令。"
    "写出一个能适当完成请求的回复。\n\n"
    "### 指令：\n{instruction}\n\n### 回复："
)

# Step3: 数据预处理
def preprocess_dataset(dataArguments: DataArguments,
                       modelArguments: ModelArguments,
                       trainingArguments: MyTrainingArguments,
                       logger: logging.RootLogger):
    tokenizer = load_tokenizer(modelArguments, trainingArguments, logger, logger_name='preprocess_dataset')
    if dataArguments.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
    else:
        max_seq_length = min(dataArguments.max_seq_length, tokenizer.model_max_length)
    def tokenize_function(examples: datasets.Dataset) -> Dict:
        """ 
        数据处理函数: 
            1、拼接instruction, input, output 字段内容,并在output字段句尾添加</s>符号表示终止符eos
            2、对拼接后的内容进行分词, 并基于允许的序列最大长度进行截断
            3、对instruction和input 部分对应的label进行mask(原始token id 用特殊index替代), 确保其不参与loss计算
        examples: 
        例如, Dataset({
                        features: ['text',...],
                        num_rows: 2
                    })
        return:  {
                'input_ids': [[1, 447, 29882, 801], [1, 447, 29882, 801]], 
                'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1]]
                }
        """
        sources, targets, prompt = [], [], PROMPT_TEMPLATE
        for instruction, input_, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input_ is not None and input_ != "":
                instruction = instruction + '\n' + input_
            source = prompt.format_map({'instruction': instruction})
            target = f"{output}{tokenizer.eos_token}"
            sources.append(source)
            targets.append(target)
        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)
        # 拼接source 和 target
        all_input_ids, all_labels = [], []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            # 将 source 对应的位置遮蔽掉，不进行loss计算，仅计算target部分的loss
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
        logger.info(f'rank:{trainingArguments.local_rank} preprocessed dataset num is {len(all_input_ids)}')
        return {'input_ids': all_input_ids, 'labels': labels}

    # 加载数据
    if dataArguments.dataset_dir:
        raw_datasets = load_dataset(path=dataArguments.dataset_dir, name=dataArguments.dataset_subset_name)
    elif dataArguments.train_file_path and dataArguments.eval_file_path and dataArguments.data_file_type:
        data_files = {
            "train": dataArguments.train_file_path,
            "validation": dataArguments.eval_file_path
        }
        raw_datasets = load_dataset(dataArguments.data_file_type, data_files=data_files)
    else:
        raise ValueError("请配置 [dataset_dir] or [train_file_path、eval_file_path、test_file_path and data_file_type]")

    os.makedirs(dataArguments.processed_data_cache_dir, exist_ok=True)

    processed_datasets = raw_datasets.map(
        function=tokenize_function,
        batched=True,
        num_proc=dataArguments.preprocessing_num_workers,
        remove_columns=['instruction', 'input', 'output'],
        load_from_cache_file=True,
        keep_in_memory=False,
        desc='preprocessing on dataset',
    )
    if dataArguments.overwrite_cache_processed_dataset:
        shutil.rmtree(dataArguments.processed_data_cache_dir)
        os.makedirs(dataArguments.processed_data_cache_dir, exist_ok=True)
    processed_datasets.save_to_disk(dataArguments.processed_data_cache_dir)
    
    logger.info(f"rank:{trainingArguments.local_rank} the num of preprocessed train dataset is {processed_datasets['train'].num_rows}")
    logger.info(f"rank:{trainingArguments.local_rank} the num of preprocessed eval dataset is {processed_datasets['validation'].num_rows}")
    logger.info(f"rank:{trainingArguments.local_rank} the num of preprocessed test dataset is {processed_datasets['test'].num_rows}")
    logger.info(f"Local Rank: {trainingArguments.local_rank} 数据预处理完毕......") 

# Step4: 加载数据
def prepare_dataloader(dataArguments: DataArguments,
                       modelArguments: ModelArguments,
                       trainingArguments: MyTrainingArguments,
                       logger: logging.RootLogger) -> Tuple[DataLoader]:
    tokenizer = load_tokenizer(modelArguments, trainingArguments, logger, logger_name='prepare_dataloader')
    processed_datasets = load_from_disk(dataArguments.processed_data_cache_dir)
    
    if trainingArguments.do_train:
        train_dataset = processed_datasets['train']
    if trainingArguments.do_eval:
        eval_dataset = processed_datasets['validation']
    if trainingArguments.debug_mode:
        max_train_samples = min(len(train_dataset), dataArguments.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        logger.info(f"Local Rank: {trainingArguments.local_rank} Num train_samples  {len(train_dataset)}")
        
        max_eval_samples = min(len(eval_dataset), dataArguments.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info(f"Local Rank: {trainingArguments.local_rank} Num eval_samples  {len(eval_dataset)}")
    
    def collate_fn(batch: List[Dict]):
        batch_input_ids, batch_label = tuple([instance[key] for instance in batch for key in ['input_ids', 'labels']])
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        batch_label = torch.nn.utils.rnn.pad_sequence(
            sequences=batch_label, batch_first=True, padding_value=IGNORE_INDEX
        )
        batch_attention_mask = batch_input_ids.ne(tokenizer.pad_token_id)
        return batch_input_ids, batch_attention_mask, batch_label
    
     # 加载deepspeed配置
    deepspeed_config = read_json(trainingArguments.deepspeed_config)
    # mini batch size
    trainingArguments.per_device_train_batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
    trainingArguments.per_device_eval_batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
    trainingArguments.gradient_accumulation_steps = deepspeed_config["gradient_accumulation_steps"]
        
    train_sampler = DistributedSampler(dataset=train_dataset)
    eval_sampler = DistributedSampler(dataset=eval_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=trainingArguments.per_device_train_batch_size, 
                                  sampler=train_sampler,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                 batch_size=trainingArguments.per_device_eval_batch_size,
                                 sampler=eval_sampler,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    
    return train_dataloader, eval_dataloader, train_sampler

# Step5: 加载模型
def load_model(modelArguments: ModelArguments,
               trainingArguments: MyTrainingArguments, 
               logger: logging.RootLogger) -> LlamaForCausalLM:
    tokenizer = load_tokenizer(modelArguments, trainingArguments, logger, logger_name='load_model')
    model_config_kwargs = {
        "cache_dir": modelArguments.cache_dir,
        "revision": modelArguments.model_revision,
        "use_auth_token": modelArguments.use_auth_token
    }
    model_config = LlamaConfig.from_pretrained(modelArguments.model_name_or_path, **model_config_kwargs)

    # 更新配置
    if modelArguments.config_overrides:
        model_config.update_from_string(modelArguments.config_overrides)
    # 精度
    torch_dtype = (
            modelArguments.torch_dtype
            if modelArguments.torch_dtype in ['auto', None]
            else getattr(torch, modelArguments.torch_dtype)
        )
    # 设置随机种子
    set_seed(trainingArguments.seed)
    
    if modelArguments.model_name_or_path:
        model = LlamaForCausalLM.from_pretrained(
            modelArguments.model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype
        )
    else:
        raise ValueError("请配置model_name_or_path")

    # 扩充词嵌入矩阵
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    # 扩充embedding矩阵大小和lm_head的输出维度大小
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Local Rank: {trainingArguments.local_rank} Llama raw vocab size is {model_vocab_size}")
    logger.info(f"Local Rank: {trainingArguments.local_rank} chinese medical tokenizer vocab size is {len(tokenizer)}")
    logger.info(f"Local Rank: {trainingArguments.local_rank} The vocab embedding has been expanded.")
    if os.path.exists(trainingArguments.resume_from_checkpoint):
        logger.info(f'Local Rank: {trainingArguments.local_rank} start loading snapshot')
        model = PeftModelForCausalLM.from_pretrained(model=model, 
                                                     model_id=trainingArguments.resume_from_checkpoint,
                                                     device_map={"": "cpu"},
                                                     torch_dtype=torch_dtype)
        info = read_json(path=trainingArguments.resume_from_checkpoint + "/training_process.json")
        trainingArguments.epochs_run = info['EPOCHS_RUN']
        trainingArguments.steps_update = info['STEP']
        logger.info(f"Local Rank: {trainingArguments.local_rank} Resuming training from snapshot at Epoch {trainingArguments.epochs_run}")
    else:
        # 创建lora config
        target_modules = trainingArguments.lora_trainable.strip().split(",")
        modules_to_save = trainingArguments.modules_to_save.strip().split(',')
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                 target_modules=target_modules,
                                 inference_mode=False, 
                                 r=trainingArguments.lora_rank, 
                                 lora_alpha=trainingArguments.lora_alpha,
                                 lora_dropout=trainingArguments.lora_dropout,
                                 modules_to_save=modules_to_save)
        model = get_peft_model(model, lora_config)

    return model

# Step6: 构建训练器
class MyTrainer:
    """ 自定义训练器 """
    def __init__(self,
                 model: PeftModelForCausalLM,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 trainSampler: DistributedSampler,
                 trainingArguments: MyTrainingArguments,
                 compute_metrics: Callable[[np.ndarray, np.ndarray], Dict],
                 logger: logging.RootLogger
                 ):
        # 静态参数
        self.n_gpus = torch.cuda.device_count()
        self.gpu_id = trainingArguments.local_rank
        # lora
        self.lora_rank = trainingArguments.lora_rank
        self.lora_alpha = trainingArguments.lora_alpha
        self.lora_dropout = trainingArguments.lora_dropout
        self.lora_trainable = trainingArguments.lora_trainable
        self.modules_to_save = trainingArguments.modules_to_save
        self.output_dir = trainingArguments.output_dir
        
        self.epochs_run = trainingArguments.epochs_run # 已训练epoch个数
        self.steps_update = trainingArguments.steps_update # 已更新参数的次数
        self.batchs_update = 0 # 已更新的batch个数
        
        self.num_train_epochs = trainingArguments.num_train_epochs
        self.gradient_accumulation_steps = trainingArguments.gradient_accumulation_steps
        self.snapshot_path = trainingArguments.resume_from_checkpoint
        self.save_steps = trainingArguments.save_steps
        
        # 采用梯度累积时，指代参数更新的次数
        self.eval_steps = trainingArguments.eval_steps
        self.logging_steps = trainingArguments.logging_steps
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.trainSampler = trainSampler
        self.compute_metrics = compute_metrics
        # 加载模型
        self.model = model.to(self.gpu_id)
        self.num_training_steps = self._get_num_training_steps(trainingArguments.max_steps) # 参数最大更新次数
        self.every_epoch_training_steps = len(self.train_dataloader) // trainingArguments.gradient_accumulation_steps
        self.training_params = self._get_training_params()
        self.num_training_params, self.total_num_params = self._get_num_params()
        
        # 数据并行
        self.model, _, _, _ = deepspeed.initialize(args=trainingArguments, 
                                                   model=self.model, 
                                                   model_parameters=self.training_params)

    def train(self):
        if self.gpu_id == 0:
            percentage = round(self.num_training_params / self.total_num_params * 100, 2)
            self.logger.info(f'Training parameters number is {format(self.num_training_params,",d")}, Total parameters number is {format(self.total_num_params, ",d")}, accounted for {percentage}%')
        self.model.module.train()
        torch.cuda.empty_cache()
        for epoch in trange(math.ceil(self.num_train_epochs), desc='Epoch', disable=False):
            self._run_epoch(epoch)
            # 达到最大更新步数，退出训练
            if self.steps_update > self.num_training_steps:
                break
        # 保存最终模型
        if self.gpu_id == 0:
            self._save_snapshot(epoch)
        self.logger.info(f'Local Rank: {self.gpu_id} 训练结束!')
    
    @torch.no_grad()
    def evaluate(self, epoch):
        self.logger.info(f'Local Rank: {self.gpu_id} 开始评测...')
        self.model.module.eval()
        metric = 0
        avg_loss = 0
        for step, batch in enumerate(self.eval_dataloader):
            batch_input_ids, batch_attention_mask, batch_label = batch
            batch_input_ids, batch_attention_mask, batch_label = batch_input_ids.cuda(self.gpu_id), batch_attention_mask.cuda(self.gpu_id), batch_label.cuda(self.gpu_id)
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_label)
            loss, logits = outputs[0].item(), outputs[1].cpu()
            batch_label = batch_label.cpu()
            metric += self.compute_metrics(logits.argmax(dim=-1), batch_label)['accuracy']
            avg_loss += loss
        self.model.module.train()
        metric, avg_loss = torch.tensor(metric).cuda(self.gpu_id), torch.tensor(avg_loss).cuda(self.gpu_id)
        # 聚合所有进程中的值到rank=0
        torch.distributed.reduce(tensor=metric, dst=0, op=ReduceOp.SUM)
        torch.distributed.reduce(tensor=avg_loss, dst=0, op=ReduceOp.SUM)
        if self.gpu_id == 0:
            metric, avg_loss = metric.mean().item() / self.n_gpus, avg_loss.mean().item() / self.n_gpus
            metric, avg_loss = metric / (step + 1), avg_loss / (step + 1)    
            self.logger.info(f'Local Rank: {self.gpu_id} Evaluate Epoch: {epoch}/{self.num_train_epochs} Step: {self.steps_update} Metric: {metric} Eval Loss: {avg_loss}')
            summary_events = [('Eval/Samples/eval_loss', avg_loss, self.model.global_samples), ('Eval/Samples/accuracy', metric, self.model.global_samples)]
            self.model.monitor.write_events(summary_events)
    
    def _run_epoch(self, epoch: int):
        self.trainSampler.set_epoch(epoch)
        cur_loss = 0
        for batch in self.train_dataloader:
            loss = self._run_batch(batch)
            cur_loss += loss
            if (self.batchs_update + 1) % self.gradient_accumulation_steps == 0:
                self.steps_update += 1 # 记录参数更新次数
                if self.gpu_id == 0 and self.steps_update % self.logging_steps == 0:
                    self._log_training_info(epoch, cur_loss)
                cur_loss = 0
                if self.steps_update % self.eval_steps == 0:
                    self.evaluate(epoch)
                # 只在主进程中保存模型，同时阻塞其它副本进程
                if self.gpu_id == 0:
                    if self.steps_update % self.save_steps == 0:
                        self._save_snapshot(epoch)
                    torch.distributed.barrier()
                else:
                    torch.distributed.barrier()
            self.batchs_update += 1 # 记录batch更新个数
            # 达到最大更新步数，退出训练，并保存最后模型
            if self.steps_update > self.num_training_steps:
                break
                     
    def _run_batch(self, batch):
        batch_input_ids, batch_attention_mask, batch_label = batch
        batch_input_ids, batch_attention_mask, batch_label = batch_input_ids.cuda(self.gpu_id), batch_attention_mask.cuda(self.gpu_id), batch_label.cuda(self.gpu_id)
        outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_label)
        loss = outputs[0]
        loss.requires_grad_(True)
        # 计算梯度值/梯度累积
        loss = self.model.backward(loss)
        # 参数更新/学习率调整
        self.model.step()
        return loss.item()
        
    def _log_training_info(self, epoch:int, loss: float):
        content = f'Local Rank: {self.gpu_id} | Epoch {epoch}/{self.num_train_epochs} | Step {self.steps_update}/{self.num_training_steps} | Loss: {loss}'
        self.logger.info(content)
    
    def _save_snapshot(self, epoch: int):
        peft_model_dir = os.path.join(self.output_dir, f"pt_lora_model_epoch_{epoch}_step_{self.steps_update}")
        os.makedirs(peft_model_dir, exist_ok=True)
        # saves the adapter model and the adapter configuration files to a directory
        self.model.module.save_pretrained(peft_model_dir)
        # 保存训练进度
        with open(peft_model_dir + "/training_process.json", "w", encoding="utf-8") as fo:
            info = {"EPOCHS_RUN": self.epochs_run, "STEP": self.steps_update}
            json.dump(info, fo, indent=4, ensure_ascii=False)
        self.logger.info(f'Local Rank: {self.gpu_id} Epoch {epoch} STEP {self.steps_update}| Training snapshot saved at {peft_model_dir}')

    def _get_num_training_steps(self, max_steps):
        if max_steps > 0:
            return max_steps
        len_dataloader = len(self.train_dataloader)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil((self.num_train_epochs - self.epochs_run) * num_update_steps_per_epoch)
        return max_steps
    
    def _get_training_params(self):
        """ 获取训练参数 """
        return [p for n,p in filter(lambda item:item[1].requires_grad, self.model.named_parameters())]
    
    def _get_num_params(self) -> int:
        """ 获取训练参数和总参数数量 参照 peft.peft_model.PeftModel.print_trainable_parameters"""
        trainable_params, all_param = 0, 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param

def main():
    # 初始化进程组
    deepspeed_setup()
    # 参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    modelArguments, dataArguments, trainingArguments = parser.parse_args_into_dataclasses()
    # 日志记录
    logger = logging.getLogger(f'Rank:{trainingArguments.local_rank} {__name__}')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO, 
                        handlers=[logging.StreamHandler(sys.stdout)])
    # 只在主进程中进行数据预处理，同时阻塞其它副本进程
    if trainingArguments.local_rank == 0:
        preprocess_dataset(dataArguments, modelArguments, trainingArguments, logger)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
    # 加载数据
    train_dataloader, eval_dataloader, trainSampler = prepare_dataloader(dataArguments, modelArguments, trainingArguments, logger)
    # 加载模型
    model = load_model(modelArguments, trainingArguments, logger)
    # 构建训练器
    trainer = MyTrainer(model=model, 
                        train_dataloader=train_dataloader, 
                        eval_dataloader=eval_dataloader, 
                        trainSampler=trainSampler, 
                        trainingArguments=trainingArguments,
                        compute_metrics=compute_metrics,
                        logger=logger)
    trainer.train()
    
if __name__ == '__main__':
    main()
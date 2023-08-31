#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: run_clm_pt_with_peft.py
# @CreateTime: 2023/08/25 14:38:22
# @WeChat: damo894127201
# @WeChat Official Accounts: NLP Journey
# @Huggingface Organizations: NLP Journey
# @Github: jialinzhang
# @Instruction: 对LlaMA2-7b进行二次预训练：训练embedding和lm_head层，以及lora，并冻结其它参数
'''
一、模块结构:
    1、命令行参数类
        1.1 数据相关参数
        1.2 模型相关参数
        1.3 训练相关参数
        1.4 实验记录相关参数: wandb
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
'''
import numpy as np
import datasets
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from transformers.optimization import get_scheduler
from torch.cuda.amp import autocast, GradScaler
from peft import get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM, PeftModel
from sklearn.metrics import accuracy_score
import wandb

import sys
import os
import math
import shutil
import json
from dataclasses import dataclass, field
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

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "词元化后(tokenization)的输入序列的长度"
                "由于样本之间有长有短, 为了加速训练, 训练样本常被拼接到一个block中"
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
    
    TrainingArguments预定义的参数:
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
    save_interval_epoch: Optional[int] = field(default=1, metadata={"help": "每多少个epoch保存checkpoint"})
    steps_update: Optional[int] = field(default=0, metadata={"help": "参数已更新次数, 用于加载checkpoint中的step"})
    epochs_run: Optional[int] = field(default=0, metadata={"help": "已训练过的epoch次数, 用于加载checkpoint中的epoch"})
    use_amp: Optional[bool] = field(default=False, metadata={"help": "是否启用混合精度"})
    # Lora
    lora_rank: Optional[int] = field(default=8, metadata={"help": "Lora矩阵秩"})
    lora_alpha: Optional[float] = field(default=32., metadata={"help": " The alpha parameter for Lora scaling, real value = lora_alpha / r"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout probability for Lora layers"})
    lora_trainable: Optional[str] = field(default="q_proj,v_proj", metadata={"help": "The names of the modules to apply Lora to"})
    modules_to_save : Optional[str] = field(default="embed_tokens,lm_head", metadata={"help": "modules apart from LoRA layers to be set as trainable and saved in the final checkpoint"})
    
@dataclass
class WandbArguments:
    project_name: Optional[str] = field(default="llama2", metadata={"help": "wandb实验项目名"})
    team_name: Optional[str] = field(default="nlp-journey", metadata={"help": "wandb团队名"})
    commit_message: Optional[str] = field(
        default=None, 
        metadata={
            "help": (
                "A longer description of the run, like a `-m` commit message in git." 
                "This helps you remember what you were doing when you ran this run."
            )
        }
    )
    experiment_name: Optional[str] = field(default=None, metadata={"help": "实验名称"})
    experiment_group: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specify a group to organize individual runs into a larger experiment."
                "For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate a model against different test sets."
                "Group gives you a way to organize runs together into a larger whole, and you can toggle this on and off in the UI"
                "用于归类实验，比如训练的实验、评估的实验、测试的实验"
            )
        }
    )
    job_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Specify the type of run, which is useful when you're grouping runs together into larger experiments using group."
                "For example, you might have multiple jobs in a group, with job types like train and eval." 
                "Setting this makes it easy to filter and group similar runs together in the UI so you can compare apples to apples."
            )
        }
    )

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
                   logger: logging.RootLogger) -> LlamaTokenizer:
    tokenizer_kwargs = {
        "cache_dir": modelArguments.cache_dir,
        "use_fast": modelArguments.use_fast_tokenizer,
        "revision": modelArguments.model_revision,
        "use_auth_token": modelArguments.use_auth_token,
    }
    
    if modelArguments.tokenizer_name_or_path:
        logger.info(f"Local Rank: {trainingArguments.local_rank} 分词器加载成功......")
        return LlamaTokenizer.from_pretrained(modelArguments.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(f"Local Rank: {trainingArguments.local_rank} 请配置分词器名称或路径: tokenizer_name_or_path")
 
def ddp_setup():
    init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# Step3: 数据预处理
def preprocess_dataset(dataArguments: DataArguments,
                       modelArguments: ModelArguments,
                       trainingArguments: MyTrainingArguments,
                       logger: logging.RootLogger):
    tokenizer = load_tokenizer(modelArguments, trainingArguments, logger)
    if dataArguments.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = min(dataArguments.block_size, tokenizer.model_max_length)
    def tokenize_function(examples: datasets.Dataset) -> Dict:
        """ 
        数据处理函数: 批量对数据集的制定列进行分词,并在句首添加<s>符号表示起始符bos
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
        return tokenizer(examples['text'])

    def group_texts(examples: datasets.Dataset) -> Dict:
        """
        数据处理函数: 拼接分词后数据集中的所有文本, 并依据block_size, 生成block块
        examples: 
        例如, Dataset({
                        features: ['input_ids','attention_mask',...],
                        num_rows: 2
                    })
        return: 
        """
        # 拼接数据集中同一个特征的所有数据
        # {'input_ids': [value,...], 'attention_mask': [value,...],....}
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        # 丢掉切分后剩余的部分数据
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # 切分文本块
        result = {
            k: [t[i:i+block_size] for i in range(0, total_length, block_size)] 
            for k,t in concatenated_examples.items()
        }
        result['label'] = result["input_ids"].copy()
        return result
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

    tokenized_datasets = raw_datasets.map(
        function=tokenize_function,
        batched=True,
        num_proc=dataArguments.preprocessing_num_workers,
        remove_columns='text',
        load_from_cache_file=True,
        keep_in_memory=False,
        desc='Runing tokenizer on raw dataset',
    )
    processed_datasets = tokenized_datasets.map(
        function=group_texts,
        batched=True,
        num_proc=dataArguments.preprocessing_num_workers,
        load_from_cache_file=True,
        keep_in_memory=False,
        desc='Grouping texts in chunks of {block_size}',
    )
    if dataArguments.overwrite_cache_processed_dataset:
        shutil.rmtree(dataArguments.processed_data_cache_dir)
        os.makedirs(dataArguments.processed_data_cache_dir, exist_ok=True)
    processed_datasets.save_to_disk(dataArguments.processed_data_cache_dir)
    logger.info(f"Local Rank: {trainingArguments.local_rank} 数据预处理完毕......") 

# Step4: 加载数据
def prepare_dataloader(dataArguments: DataArguments,
                       trainingArguments: MyTrainingArguments,
                       logger: logging.RootLogger) -> Tuple[DataLoader]:
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
        batch_input_ids, batch_attention_mask, batch_label = [], [], []
        for item in batch:
            batch_input_ids.append(item['input_ids'])
            batch_attention_mask.append(item['attention_mask'])
            batch_label.append(item['label'])
        batch_input_ids = torch.tensor(batch_input_ids).cuda(trainingArguments.local_rank)
        batch_attention_mask = torch.tensor(batch_attention_mask).cuda(trainingArguments.local_rank)
        batch_label = torch.tensor(batch_label).cuda(trainingArguments.local_rank)
        return batch_input_ids, batch_attention_mask, batch_label
        
    train_sampler = DistributedSampler(dataset=train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=trainingArguments.train_batch_size, 
                                  sampler=train_sampler,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                 batch_size=trainingArguments.eval_batch_size,
                                 collate_fn=collate_fn,
                                 drop_last=False)
    
    return train_dataloader, eval_dataloader, train_sampler

# Step5: 加载模型
def load_model(modelArguments: ModelArguments,
               trainingArguments: MyTrainingArguments, 
               logger: logging.RootLogger) -> LlamaForCausalLM:
    tokenizer = load_tokenizer(modelArguments, trainingArguments, logger)
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

    # 如果指定精度为float16或bfloat16则不进行混合精度计算
    if modelArguments.torch_dtype in ['float16', 'bfloat16']:
        trainingArguments.use_amp = False
        
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
        with open(trainingArguments.resume_from_checkpoint + "/training_process.json", "r", encoding='utf-8') as fi:
            info = json.load(fi)
            trainingArguments.epochs_run = info['EPOCHS_RUN']
            trainingArguments.steps_update = info['STEP']
        logger.info(f"Local Rank: {trainingArguments.local_rank} Resuming training from snapshot at Epoch {trainingArguments.epochs_run}")
    else:
        # 创建lora config
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                                inference_mode=False, 
                                r=trainingArguments.lora_rank, 
                                lora_alpha=trainingArguments.lora_alpha,
                                lora_dropout=trainingArguments.lora_dropout,
                                modules_to_save=trainingArguments.modules_to_save.split(','))
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
                 wandbArguments: WandbArguments,
                 compute_metrics: Callable[[np.ndarray, np.ndarray], Dict],
                 logger: logging.RootLogger
                 ):
        # 静态参数
        self.n_gpus = torch.cuda.device_count()
        self.gpu_id = trainingArguments.local_rank
        self.lr = trainingArguments.learning_rate
        self.lr_scheduler_type = trainingArguments.lr_scheduler_type
        self.adam_beta1 = trainingArguments.adam_beta1
        self.adam_beta2 = trainingArguments.adam_beta2
        self.adam_epsilon = trainingArguments.adam_epsilon
        self.weight_decay = trainingArguments.weight_decay
        # 每多少个batch进行梯度更新
        self.gradient_accumulation_steps = trainingArguments.gradient_accumulation_steps
        self.max_grad_norm = trainingArguments.max_grad_norm
        # lora
        self.lora_rank = trainingArguments.lora_rank
        self.lora_alpha = trainingArguments.lora_alpha
        self.lora_dropout = trainingArguments.lora_dropout
        self.lora_trainable = trainingArguments.lora_trainable
        self.modules_to_save = trainingArguments.modules_to_save
        self.output_dir = trainingArguments.output_dir
        
        self.epochs_run = trainingArguments.epochs_run # 已训练epoch个数
        self.steps_update = trainingArguments.steps_update # 已更新参数的次数
        self.num_train_epochs = trainingArguments.num_train_epochs
        self.train_batch_size = trainingArguments.train_batch_size
        self.eval_batch_size = trainingArguments.eval_batch_size
        self.snapshot_path = trainingArguments.resume_from_checkpoint
        self.save_steps = trainingArguments.save_steps
        self.save_interval_epoch = trainingArguments.save_interval_epoch
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
        # 动态参数
        self.num_training_steps = self._get_num_training_steps(trainingArguments.max_steps) # 参数最大更新次数
        self.every_epoch_training_steps = len(self.train_dataloader) // self.gradient_accumulation_steps
        self.warmup_steps = trainingArguments.get_warmup_steps(self.num_training_steps)
        self.training_params = self._get_training_params()
        self.num_training_params, self.total_num_params = self._get_num_params()
        self.optimizer, self.scheduler = self._create_optimizer_and_scheduler()
        # 混合精度训练
        self.use_amp = trainingArguments.use_amp
        self.scaler = GradScaler()
        
        # 数据并行
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        
        import wandb
        self.wandb = wandbArguments
        self.post_init()
        
    def post_init(self):
        training_config = {
            'model': 'llama2-7b',
            'learning_rate': self.lr,
            'lr_sheduler_type': self.lr_scheduler_type,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_epsilon': self.adam_epsilon,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'num_training_epochs': self.num_train_epochs,
            'num_training_steps': self.num_training_steps,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'lora_trainable': self.lora_trainable,
            'modules_to_save': self.modules_to_save,
            'max_grad_norm': self.max_grad_norm,
            'gradient_accumulation_steps': self.max_grad_norm,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.eval_batch_size,
            'n_gpus': self.n_gpus,
            'every_epoch_training_steps': self.every_epoch_training_steps,
            'num_training_params': self.num_training_params,
            'total_num_params': self.total_num_params
        }
        wandb.init(
            config=training_config,
            project=self.wandb.project_name,
            entity=self.wandb.team_name,
            name=self.wandb.experiment_name,
            notes=self.wandb.commit_message,
            group=self.wandb.experiment_group,
            job_type=self.wandb.job_type,
            reinit=True
        )
    
    def train(self):
        if self.gpu_id == 0:
            percentage = round(self.num_training_params / self.total_num_params * 100, 2)
            self.logger.info(f'Training parameters number is {format(self.num_training_params,",d")}, Total parameters number is {format(self.total_num_params, ",d")}, accounted for {percentage}%')
        torch.cuda.empty_cache()
        self.model.train()
        for epoch in trange(1, math.ceil(self.num_train_epochs+1), desc='Epoch', disable=False):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and (epoch % self.save_interval_epoch == 0) or (self.steps_update % self.save_steps == 0):
                self._save_snapshot(epoch)
            # 达到最大更新步数，退出训练
            if self.gpu_id == 0 and self.steps_update > self.num_training_steps:
                self._save_snapshot(epoch)
                break
        self.logger.info(f'Local Rank: {self.gpu_id} 训练结束!')
    
    @torch.no_grad()
    def evaluate(self, epoch: int):
        self.logger.info(f'Local Rank: {self.gpu_id} 开始评测...')
        self.model.eval()
        metric = 0
        avg_loss = 0
        for step, batch in enumerate(self.eval_dataloader):
            batch_input_ids, batch_attention_mask, batch_label = batch
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_label)
            loss, logits = outputs[0].item(), outputs[1].cpu()
            batch_label = batch_label.cpu()
            metric += self.compute_metrics(logits.argmax(dim=-1), batch_label)['accuracy']
            avg_loss += loss
        self.model.train()
        metric, avg_loss = metric / (step + 1), avg_loss / (step + 1)
        self.logger.info(f'Local Rank: {self.gpu_id} Evaluate Epoch: {epoch}/{self.num_train_epochs} Step: {self.steps_update} Metric: {metric} Eval Loss: {avg_loss}')
        wandb.log({"eval_loss": avg_loss}, step=self.steps_update)
        wandb.log({"eval_accuracy": metric}, step=self.steps_update)
    
    def _run_epoch(self, epoch: int):
        self.trainSampler.set_epoch(epoch)
        self.optimizer.zero_grad()
        # 在梯度累积步数内的损失
        cur_loss = 0
        for batch_index, batch in enumerate(self.train_dataloader):
            loss = self._run_batch(batch)
            cur_loss += loss
            # 当经过梯度累积步数个batch后，进行参数更新
            if (batch_index + 1) % self.gradient_accumulation_steps == 0:
                self._update_params()
                cur_loss = 0
                self.steps_update += 1 # 记录参数更新次数
                if self.gpu_id == 0:
                    if self.steps_update % self.logging_steps == 0:
                        self._log_training_info(epoch, cur_loss)
                    if self.steps_update % self.eval_steps == 0:
                        self.evaluate(epoch)
            # 达到最大更新步数，退出训练
            if self.steps_update > self.num_training_steps:
                break
                     
    def _run_batch(self, batch):
        batch_input_ids, batch_attention_mask, batch_label = batch
        if self.use_amp:
            with autocast():
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_label)
        else:
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_label)
        loss = outputs[0]
        loss.requires_grad_(True)
        # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
        if self.gradient_accumulation_steps > 0:
            loss = loss / self.gradient_accumulation_steps
        # 计算梯度值,并累加梯度
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        # 裁剪梯度
        torch.nn.utils.clip_grad_norm_([p for n,p in self.training_params], self.max_grad_norm)
        return loss.item()
    
    def _update_params(self):
        # 执行参数更新
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        # 学习率调整
        self.scheduler.step()
        # 梯度清零
        self.optimizer.zero_grad()
        
    def _log_training_info(self, epoch:int, loss: float):
        content = f'Local Rank: {self.gpu_id} | Epoch {epoch}/{self.num_train_epochs} | Step {self.steps_update}/{self.num_training_steps} | Loss: {loss}'
        self.logger.info(content)
        wandb.log({'training_loss': loss}, step=self.steps_update)
    
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
    
    def _create_optimizer_and_scheduler(self):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        return optimizer, scheduler
    
    def _create_optimizer(self):
        no_decay = ['bias', 'layernorm', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.training_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.training_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                      lr=self.lr,
                                      betas=(self.adam_beta1, self.adam_beta2),
                                      eps=self.adam_epsilon,
                                      weight_decay=self.weight_decay)
        return optimizer
    
    def _create_scheduler(self, optimizer: torch.optim.Optimizer = None):
        lr_scheduler = get_scheduler(
            self.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return lr_scheduler

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
        return list(filter(lambda item:item[1].requires_grad, self.model.named_parameters()))
    
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
    ddp_setup()
    # 参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments, WandbArguments))
    modelArguments, dataArguments, trainingArguments, wandbArguments = parser.parse_args_into_dataclasses()
    # 日志记录
    logger = logging.getLogger(f'Rank:{trainingArguments.local_rank} {__name__}')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO, 
                        handlers=[logging.StreamHandler(sys.stdout)])
    # 只在主进程中进行数据预处理，同时阻塞其它副本进程
    with trainingArguments.main_process_first(desc='数据预处理'):
        preprocess_dataset(dataArguments, modelArguments, trainingArguments, logger)
    # 加载数据
    train_dataloader, eval_dataloader, trainSampler = prepare_dataloader(dataArguments, trainingArguments, logger)
    # 加载模型
    model = load_model(modelArguments, trainingArguments, logger)
    # 构建训练器
    trainer = MyTrainer(model=model, 
                        train_dataloader=train_dataloader, 
                        eval_dataloader=eval_dataloader, 
                        trainSampler=trainSampler, 
                        trainingArguments=trainingArguments,
                        wandbArguments=wandbArguments,
                        compute_metrics=compute_metrics,
                        logger=logger)
    trainer.train()
    destroy_process_group()
    
if __name__ == '__main__':
    main()

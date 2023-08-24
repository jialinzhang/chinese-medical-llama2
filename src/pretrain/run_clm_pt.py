#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @File: run_clm_pt.py
# @CreateTime: 2023/08/23 08:24:54
# @WeChat: damo894127201
# @WeChat Official Accounts: NLP Journey
# @Huggingface Organizations: NLP Journey
# @Github: jialinzhang
# @Instruction: 对LlaMA2进行二次预训练：训练embedding和lm_head层，冻结其它参数

import datasets
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from transformers.optimization import get_scheduler
from sklearn.metrics import accuracy_score

import sys
import os
import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Dict, Tuple
from tqdm import tqdm, trange
import logging

# Step1: 参数设置和评测指标
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
        default=True, metadata={"help": ("是否覆盖缓存的预处理后的训练和评估集")}
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
            "help": ("覆盖默认的“torch.dtype”并在此dtype下加载模型。如果传递“auto”，则“dtype“将自动从模型的权重中得出"),
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
    9、lr_scheduler_type: 学习率调度策略类型
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
    ....
    
    与微调模型相关的参数
    
    """
    
    debug_mode: Optional[bool] = field(default=False, metadata={"help": ("是否调试模式")})
    modules_to_train : Optional[str] = field(default=None, metadata={"help": "参与训练的网络层, example: embed_tokens,lm_head"})

def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

def compute_metrics(preds, labels):
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics but we need to shift the labels
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)        
    

# Step2: 数据预处理
def load_tokenizer_and_preprocess_dataset(dataArguments: DataArguments,
                                          modelArguments: ModelArguments,
                                          trainingArguments: MyTrainingArguments,
                                          only_load_tokenizer: bool,
                                          logger: logging.RootLogger) -> LlamaTokenizer:
    tokenizer_kwargs = {
        "cache_dir": modelArguments.cache_dir,
        "use_fast": modelArguments.use_fast_tokenizer,
        "revision": modelArguments.model_revision,
        "use_auth_token": modelArguments.use_auth_token,
    }
    
    if modelArguments.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(modelArguments.tokenizer_name_or_path, **tokenizer_kwargs)
        logger.info("分词器加载成功......")
    else:
        raise ValueError("请配置分词器名称或路径: tokenizer_name_or_path")
    # 只加载tokenizer，不进行数据预处理
    if only_load_tokenizer:
        return tokenizer
    block_size = dataArguments.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
    if block_size > 1024:
        logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if dataArguments.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({dataArguments.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
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
    with trainingArguments.main_process_first(desc='数据预处理'):
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
            os.removedirs(dataArguments.processed_data_cache_dir)
            os.makedirs(dataArguments.processed_data_cache_dir, exist_ok=True)
        processed_datasets.save_to_disk(dataArguments.processed_data_cache_dir)
        logger.info("数据预处理完毕......") 
        return tokenizer

# Step3: 加载数据
def create_dataloader(tokenizer: LlamaTokenizer,
                      dataArguments: DataArguments,
                      trainingArguments: MyTrainingArguments) -> Tuple[DataLoader]:
    
    processed_datasets = datasets.load_from_disk(dataArguments.processed_data_cache_dir)
    if trainingArguments.do_train:
        train_dataset = processed_datasets['train']
    if trainingArguments.do_eval:
        eval_dataset = processed_datasets['validation']
    if trainingArguments.debug_mode:
        max_train_samples = min(len(train_dataset), dataArguments.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
        
        logger.info(f"Num train_samples  {len(train_dataset)}")
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
        
        max_eval_samples = min(len(eval_dataset), dataArguments.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        logger.info("evaling example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))
    
    def collate_fn(batch: datasets.Dataset):
        return None
        
    train_sampler = DistributedSampler(dataset=train_dataset)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=trainingArguments.train_batch_size, 
                                  sampler=train_sampler,
                                  collate_fn=collate_fn)
    eval_dataloader = DataLoader(dataset=eval_dataset, 
                                 batch_size=trainingArguments.eval_batch_size,
                                 collate_fn=collate_fn)
    
    return train_dataloader, eval_dataloader, train_sampler

# Step4: 加载模型
def load_ddp_fsdp_model(tokenizer: LlamaTokenizer,
                        modelArguments: ModelArguments,
                        trainingArguments: MyTrainingArguments, 
                        logger: logging.RootLogger) -> LlamaForCausalLM:
    model_config_kwargs = {
        "cache_dir": modelArguments.cache_dir,
        "revision": modelArguments.model_revision,
        "use_auth_token": modelArguments.use_auth_token
    }
    model_config = LlamaConfig.from_pretrained(modelArguments.model_name_or_path, **model_config_kwargs)

    # 更新配置
    if modelArguments.config_overrides:
        model_config.update_from_string(modelArguments.config_overrides)

    if modelArguments.model_name_or_path:
        torch_dtype = (
            modelArguments.torch_dtype
            if modelArguments.torch_dtype in ['auto', None]
            else getattr(torch, modelArguments.torch_dtype)
        )
        model = LlamaForCausalLM.from_pretrained(
            modelArguments.model_name_or_path,
            config=model_config,
            torch_dtype=torch_dtype
        )
        model = DDP(module=model.cuda(trainingArguments.local_rank), device_ids=[trainingArguments.local_rank])
        model = FSDP(module=model, cpu_offload=CPUOffload(offload_params=True))
    else:
        raise ValueError("请配置model_name_or_path")

    # 扩充词嵌入矩阵
    model_vocab_size = model.get_output_embeddings().weight.size(0)
    # model.resize_token_embeddings: 扩充embedding矩阵大徐爱和lm_head的输出维度大小
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Llama raw vocab size is {model_vocab_size}")
    logger.info(f"chinese medical tokenizer vocab size is {len(tokenizer)}")
    logger.info("The vocab embedding has been expanded.")
    
    return model

# Step5: 训练器
class MyTrainer:
    """ 自定义训练器 """
    def __init__(self, 
                 model: torch.nn.Module,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader,
                 trainSampler: DistributedSampler,
                 trainingArguments: MyTrainingArguments,
                 logger: logging.RootLogger
                 ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.trainSampler = trainSampler
        self.trainingArguments = trainingArguments
        self.logger = logger
        self.n_gpus = torch.cuda.device_count()
        self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()
        
        
    def train(self):
        if self.trainingArguments.device == "cuda":
            torch.cuda.empty_cache()
        self.model.train()
        for epoch_index in trange(self.trainingArguments.num_train_epochs, desc="Epoch", disable=False):
            self.trainSampler.set_epoch(epoch_index)
            ema_loss = 0
            num_batches = len(self.train_dataloader)
            for batch_index, batch in enumerate(self.train_dataloader):
                cur_step = num_batches*epoch_index + batch_index + 1
                input_ids = batch['input_ids'].cuda(self.trainingArguments.local_rank)
                attention_mask = batch['attention_mask'].cuda(self.trainingArguments.local_rank)
                label = batch['label'].cuda(self.trainingArguments.local_rank)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
                loss = outputs[0]
                # 判断是否进行梯度累积，如果进行，则将损失值除以累积步数
                if self.trainingArguments.gradient_accumulation_steps > 0:
                    loss /= self.trainingArguments.gradient_accumulation_steps
                ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
                # 损失回传计算梯度值,并累加梯度
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm([p for n,p in self.train_params], self.trainingArguments.max_grad_norm)
                # 当训练步数整除累积步数时，进行参数更新
                if (batch_index + 1) % self.trainingArguments.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    # 梯度清零
                    self.optimizer.zero_grad()
                    # 日志输出
                    if self.trainingArguments.local_rank == 0:
                        epoch_des = f'{epoch_index}/{self.trainingArguments.num_train_epochs}'
                        step_des = f'{cur_step}/{self.get_num_training_steps}'
                        self.logger.info(f'Epoch: ({epoch_des}) Step: ({step_des}) Loss: ({loss}) EMA Loss: ({ema_loss})')
                    # 评估
                    if (batch_index + 1) % self.trainingArguments.eval_steps == 0:
                        self.logger.info("开始评测......")
                        metric = self.evaluate()
                        self.logger.info(f'Epoch: ({epoch_des}) Step: ({step_des}) Accuracy: ({metric})')
                        self.logger.info("评测结束......")
                # 模型保存
                if (epoch_index + 1) % self.trainingArguments.num_train_epochs == 0 and self.trainingArguments.local_rank == 0:
                    os.makedirs(self.trainingArguments.model_save_path, exist_ok=True)
                    self.save_model(epoch_index, batch_index, ema_loss)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        metric = 0
        for step, batch in enumerate(self.eval_dataloader):
            input_ids = batch['input_ids'].cuda(self.trainingArguments.local_rank)
            attention_mask = batch['attention_mask'].cuda(self.trainingArguments.local_rank)
            label = batch['label'].cuda(self.trainingArguments.local_rank)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=label)
            logits = outputs[1]
            metric += accuracy(logits.item(), label)
        self.model.train()
        return metric / (step + 1)
    
    def save_model(self, epoch_index, batch_index, loss):
        os.makedirs(self.trainingArguments.output_dir, exist_ok=True)
        save_file = os.path.join(self.trainingArguments.output_dir, f"epoch_{epoch_index}.pt")
        torch.save({
            'epoch': epoch_index,
            'step': batch_index,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'ema_loss': loss,
        }, save_file)
        self.logger.info(f'checkpoint has been save in {save_file}')
          
    def create_optimizer_and_scheduler(self):
        num_training_steps = self.get_num_training_steps()
        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(num_training_steps, optimizer)
        return optimizer, scheduler
    
    def create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.train_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.trainingArguments.weight_decay
            },
            {
                'params': [p for n, p in self.train_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                      lr=self.trainingArguments.learning_rate,
                                      betas=(self.trainingArguments.adam_beta1, self.trainingArguments.adam_beta2),
                                      eps=self.trainingArguments.adam_epsilon,
                                      weight_decay=self.trainingArguments.weight_decay)
        return optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        torch.optim.lr_scheduler.CosineAnnealingLR
        lr_scheduler = get_scheduler(
            self.trainingArguments.lr_scheduler_type,
            optimizer=self.optimizer if self.optimizer else optimizer,
            num_warmup_steps=self.trainingArguments.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler
    
    def get_num_training_steps(self):
        if self.trainingArguments.max_steps > 0:
            return self.trainingArguments.max_steps
        len_dataloader = len(self.train_dataloader)
        num_update_steps_per_epoch = len_dataloader // self.trainingArguments.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.trainingArguments.num_train_epochs * num_update_steps_per_epoch)
        return max_steps
    
    @property
    def train_params(self):
        """ 获取训练参数 """
        train_modules = self.trainingArguments.modules_to_train.strip().split(',')
        for name, value in self.model.named_parameters():
            module_name = name.split('.')[0]
            if module_name in train_modules:
                value.requires_grad = True
                continue
            value.requires_grad = False
        return filter(lambda item:item[1].requires_grad, self.model.named_parameters())
    
    @property
    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """ 获取参数数量"""
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight" for name, module_type in self.named_modules() if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if p.requires_grad or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad or not only_trainable)

if __name__ == '__main__':
    # 参数解析
    parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
    modelArguments, dataArguments, trainingArguments = parser.parse_args_into_dataclasses()
    logger = logging.getLogger(f'Rank:{trainingArguments.local_rank} {__name__}')
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.INFO, 
                        handlers=[logging.StreamHandler(sys.stdout)])
    # 数据预处理
    tokenizer = load_tokenizer_and_preprocess_dataset(dataArguments, modelArguments, trainingArguments, False, logger)
    # 加载数据
    train_dataloader, eval_dataloader, trainSampler = create_dataloader(tokenizer, dataArguments, trainingArguments)
    # 加载模型
    model = load_ddp_fsdp_model(tokenizer, modelArguments, trainingArguments, logger)
    # 构建训练器
    trainer = MyTrainer(model=model, 
                        train_dataloader=train_dataloader, 
                        eval_dataloader=eval_dataloader, 
                        trainSampler=trainSampler, 
                        trainingArguments=trainingArguments,
                        logger=logger)
    trainer.train()
    logger.info("训练结束!")
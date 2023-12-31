# WandbArguments
team_name=nlp-journey
project_name=llama2
experiment_name=train_embed_token_and_lm_head_with_peft
experiment_group=llama2-7b训练
commit_message=llama2-7b训练
job_type=training
# DataArguments
dataset_dir=shibing624/medical
dataset_subset_name=pretrain
block_size=512
processed_data_cache_dir=./processed
# ModelArguments
model_name_or_path=meta-llama/Llama-2-7b-hf
tokenizer_name_or_path=../../data/chinese-medical-llama-tokenizer/hf_dir/
use_auth_token=your_huggingface_access_token
torch_dtype=float16
# TrainingArguments
lora_rank=8
lora_alpha=32.
lora_dropout=0.1
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save=embed_tokens,lm_head
use_amp=True
debug_mode=False
learning_rate=0.0003
weight_decay=0.1
adam_beta1=0.9
adam_beta2=0.95
adam_epsilon=0.00001
max_grad_norm=1.0
num_train_epochs=1
lr_scheduler_type=cosine
warmup_ratio=0.05
do_train=True
do_eval=True
logging_steps=100
eval_steps=10000
save_steps=10000
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=8
output_dir=./output_dir
resume_from_checkpoint=./output_dir


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --nnodes 1 --node_rank 0 run_clm_pt_with_peft.py \
                               --team_name ${team_name} \
                               --project_name ${project_name} \
                               --experiment_name ${experiment_name} \
                               --experiment_group ${experiment_group} \
                               --commit_message ${commit_message} \
                               --job_type ${job_type} \
                               --dataset_dir ${dataset_dir} \
                               --dataset_subset_name ${dataset_subset_name} \
                               --block_size ${block_size} \
                               --processed_data_cache_dir ${processed_data_cache_dir} \
                               --model_name_or_path ${model_name_or_path} \
                               --tokenizer_name_or_path ${tokenizer_name_or_path} \
                               --use_auth_token ${use_auth_token} \
                               --torch_dtype ${torch_dtype} \
                               --lora_rank ${lora_rank} \
                               --lora_alpha ${lora_alpha} \
                               --lora_dropout ${lora_dropout} \
                               --lora_trainable ${lora_trainable} \
                               --modules_to_save ${modules_to_save} \
                               --use_amp ${use_amp} \
                               --debug_mode ${debug_mode} \
                               --learning_rate ${learning_rate} \
                               --weight_decay ${weight_decay} \
                               --adam_beta1 ${adam_beta1} \
                               --adam_beta2 ${adam_beta2} \
                               --adam_epsilon ${adam_epsilon} \
                               --max_grad_norm ${max_grad_norm} \
                               --num_train_epochs ${num_train_epochs} \
                               --lr_scheduler_type ${lr_scheduler_type} \
                               --warmup_ratio ${warmup_ratio} \
                               --do_train ${do_train} \
                               --do_eval ${do_eval} \
                               --eval_steps ${eval_steps} \
                               --save_steps ${save_steps} \
                               --logging_steps ${logging_steps} \
                               --per_device_train_batch_size ${per_device_train_batch_size} \
                               --per_device_eval_batch_size ${per_device_eval_batch_size} \
                               --gradient_accumulation_steps ${gradient_accumulation_steps} \
                               --output_dir ${output_dir} \
                               --resume_from_checkpoint ${resume_from_checkpoint}
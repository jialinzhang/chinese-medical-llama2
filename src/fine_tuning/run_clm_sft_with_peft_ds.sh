# DataArguments
dataset_dir=shibing624/medical
dataset_subset_name=finetune
max_seq_length=512
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
debug_mode=False
deepspeed_config=./ds_config.json
max_steps=20000 # 参数最大更新次数
num_train_epochs=1
do_train=True
do_eval=True
logging_steps=100
eval_steps=10000
save_steps=10000
output_dir=./output_dir
resume_from_checkpoint=./output_dir


deepspeed --num_gpus=2 --num_nodes=1 run_clm_sft_with_peft_ds.py \
                               --dataset_dir ${dataset_dir} \
                               --dataset_subset_name ${dataset_subset_name} \
                               --max_seq_length ${max_seq_length} \
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
                               --debug_mode ${debug_mode} \
                               --deepspeed_config ${deepspeed_config} \
                               --max_steps ${max_steps} \
                               --num_train_epochs ${num_train_epochs} \
                               --do_train ${do_train} \
                               --do_eval ${do_eval} \
                               --logging_steps ${logging_steps} \
                               --eval_steps ${eval_steps} \
                               --save_steps ${save_steps} \
                               --output_dir ${output_dir} \
                               --resume_from_checkpoint ${resume_from_checkpoint}
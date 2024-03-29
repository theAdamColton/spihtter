python -m spihtter.scripts.train\
	--dataset mnist \
	--do_train \
	--max_bits 256 \
	--generate_steps 50 \
	--model_conf ./model_configurations/llama_small.json \
	--warmup_steps 10 \
	--max_steps 1000 \
	--output_dir out \
	--logging_steps 1 \
	--optim adamw_torch \
	--learning_rate 1e-3 \
	--save_total_limit 1 \
	--lr_scheduler_type cosine \
	--weight_decay 0.1 \
	--adam_beta1 0.9 \
	--adam_beta2 0.95 \
	--adam_epsilon 1e-5 \
	--per_device_train_batch_size 16 \

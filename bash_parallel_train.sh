export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --input_size 64 \
    --tensor_model_parallel_size 2 \
    --distributed_timeout_minutes 1.0 \
    --batch_size 32 \
    --init_lr 0.001 \
    --momentum 0.9 \
    --num_iterations 200 \
    --update_freq 10 \
    --data_dir data \
"

torchrun $DISTRIBUTED_ARGS parallel_train.py \
	$TRAINING_ARGS \
    	--distributed_backend nccl

#!/bin/bash

# Set the important arguments
ARCH="resnet50_cifar100"
DSET="cifar100"
NUM_WORKERS=10
EXP_NAME="default_exp"
EXP_ID="default_id"
SHUFFLE_TRAIN=False
TEST_BATCH_SIZE=100
SAMPLE_SIZE=3000
UNSTR=0
SEED=42
SPARSITY=0.95
LAMBDA_INV=0.0001
ALGO="Newton"
LOWR=0.1
PARALLEL=False
NONUNIFORM=""
NM_N=1
NM_M=4
MASK_ALG="MP"
MAX_CG_ITERATIONS=1000

# Run the Python script with the specified arguments
python run.py \
  --arch $ARCH \
  --dset $DSET \
  --num_workers $NUM_WORKERS \
  --exp_name $EXP_NAME \
  --exp_id $EXP_ID \
  --shuffle_train $SHUFFLE_TRAIN \
  --test_batch_size $TEST_BATCH_SIZE \
  --sample_size $SAMPLE_SIZE \
  --unstr $UNSTR \
  --seed $SEED \
  --sparsity $SPARSITY \
  --lambda_inv $LAMBDA_INV \
  --algo $ALGO \
  --lowr $LOWR \
  --parallel $PARALLEL \
  $NONUNIFORM \
  --NM_N $NM_N \
  --NM_M $NM_M \
  --mask_alg $MASK_ALG \
  --max_CG_iterations $MAX_CG_ITERATIONS

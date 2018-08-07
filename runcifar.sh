#! /bin/sh

MODEL=alexnet   
NGPUS=1
BATCH_SIZE=128
ITERATIONS=1000
rm /home/shidong/Lab/cifar10/train/*
python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=${MODEL} \
  --data_name=cifar10 --data_dir=/home/shidong/Lab/cifar10/data --train_dir=/home/shidong/Lab/cifar10/train \
  --save_summaries_steps 10 --summary_verbosity=0 --save_model_secs=1 --print_training_accuracy=True \
  --variable_update=parameter_server --local_parameter_device=cpu --num_batches=${ITERATIONS} \
  --num_gpus=${NGPUS} --batch_size=${BATCH_SIZE} --log_animation=True 2>&1 | tee /home/shidong/Lab/cifar10/tf-cifar.txt

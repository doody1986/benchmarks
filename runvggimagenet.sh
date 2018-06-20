#! /bin/sh

MODEL=vgg16   
NGPUS=1
BATCH_SIZE=32
ITERATIONS=1000000
echo "Batch size: " ${BATCH_SIZE}
python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=${MODEL} \
  --data_name=imagenet --data_dir=/data/imagenet/tf --train_dir=/home/shidong/Lab/vgg/train \
  --save_summaries_steps 100 --summary_verbosity=1 --save_model_secs=1000 --print_training_accuracy=True \
  --display_every 1000\
  --variable_update=parameter_server --local_parameter_device=cpu --num_batches=${ITERATIONS} \
  --num_gpus=${NGPUS} --batch_size=${BATCH_SIZE} 2>&1 | tee /home/shidong/Lab/vgg/tf-vgg.txt

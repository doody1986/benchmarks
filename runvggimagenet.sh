#! /bin/sh

if [ $# -eq 1 ]; then
  ITERATIONS=$1
elif [ $# -gt 1 ]; then
  ITERATIONS=$1
  DATA_DIR=$2
else
  echo "At least one input argument"
  exit
fi

MODEL=vgg16   
NGPUS=1
BATCH_SIZE=32
DATE=$(date +%Y-%m-%d)

TRAIN_DIR=${HOME}/Lab/vgg/train-${DATE}
if [ ! -d "${TRAIN_DIR}" ]; then
  mkdir ${TRAIN_DIR}
fi

python ./scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=${MODEL} \
  --data_name=imagenet --data_dir=/data/imagenet/tf --train_dir=${TRAIN_DIR} \
  --save_summaries_steps 10 --summary_verbosity=0 --save_model_secs=1000 --print_training_accuracy=True \
  --display_every 1\
  --variable_update=parameter_server --local_parameter_device=cpu --num_batches=${ITERATIONS} \
  --num_gpus=${NGPUS} --batch_size=${BATCH_SIZE} 2>&1 | tee ${TRAIN_DIR}/tf-vgg.txt

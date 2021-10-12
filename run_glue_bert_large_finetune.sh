LOG_DIR="log"
if [ ! -d "$LOG_DIR" ]; then
  mkdir $LOG_DIR
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

NGPU=1

echo "Started scripts"

TASK=$1
NUM_EPOCH=$2
base_dir=`pwd`
JOBNAME=$3
model_name="bert-large-uncased"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${model_name}/${JOBNAME}_bsz${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}"

GLUE_DIR="/data/GlueData"

# python -m torch.distributed.launch --nproc_per_node=${NGPU} \ --master_port=12346 \

echo "Fine Tuning $CHECKPOINT_PATH"
run_cmd="deepspeed --num_gpus=2 bert-large-finetune.py \
       --deepspeed
       --deepspeed_config bert-large-uncased.json \
       --task_name $TASK \
       --do_lower_case \
       --model_name ${model_name}\
       --data_dir $GLUE_DIR/$TASK/ \
       --max_seq_length 128 \
       --num_train_epochs ${NUM_EPOCH} \
       --output_dir ${OUTPUT_DIR}_${TASK} \
       &> ${LOG_DIR}/${model_name}/${JOBNAME}_${TASK}_bzs${EFFECTIVE_BATCH_SIZE}_lr${LR}_epoch${NUM_EPOCH}_${NGPU}_deepspeed-kernel.log
       "
echo ${run_cmd}
eval ${run_cmd}

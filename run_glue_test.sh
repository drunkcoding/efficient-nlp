LOG_DIR="TestResult"
if [ ! -d "$LOG_DIR" ]; then
  mkdir $LOG_DIR
fi

TASK=$1

python collaboration_hf.py --data_dir /data/GlueData/${TASK}/ --task_name ${TASK} --max_seq_length 512 --do_lower_case &> ${LOG_DIR}/${TASK}.log
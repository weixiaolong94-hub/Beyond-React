cd toolbench/tooleval
export RAW_ANSWER_PATH=
export CONVERTED_ANSWER_PATH=
export MODEL_NAME=
export test_set=

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json

python convert_to_answer_format.py\
    --answer_dir ${answer_dir} \
    --method DAG \
    --output ${output_file}
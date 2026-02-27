cd  toolbench/tooleval
export API_POOL_FILE=
export CONVERTED_ANSWER_PATH=
export SAVE_PATH=
mkdir -p ${SAVE_PATH}
export CANDIDATE_MODEL=virtual_chatgpt_cot
export EVAL_MODEL=
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}


python eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids  \
    --max_eval_threads 10 \
    --evaluate_times 3 \
    --test_set  G3_instruction --overwrite
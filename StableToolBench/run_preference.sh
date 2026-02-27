cd  toolbench/tooleval
export API_POOL_FILE=
export CONVERTED_ANSWER_PATH=
export SAVE_PATH=
export PASS_RATE_PATH=
export REFERENCE_MODEL=
export CANDIDATE_MODEL=
export EVAL_MODEL=
mkdir -p ${SAVE_PATH}


python eval_preference.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids  \
    --save_path ${SAVE_PATH} \
    --pass_rate_result_path ${PASS_RATE_PATH} \
    --max_eval_threads 10 \
    --use_pass_rate true \
    --evaluate_times 3 \
    --test_set G1_instruction 
export TOOLBENCH_KEY=""

export OPENAI_KEY=""
export OPENAI_API_BASE=""
export PYTHONPATH=./


export GPT_MODEL=""
export SERVICE_URL=""

export OUTPUT_DIR=""
group=G3_instruction
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir  \
    --backbone_model chatgpt_function \
    --chatgpt_model $GPT_MODEL \
    --openai_key $OPENAI_KEY \
    --base_url $OPENAI_API_BASE \
    --max_observation_length 1024 \
    --method DAG \
    --input_query_file solvable_queries_filter/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key "" \
    --num_thread 2 \
    --planning_model_path 
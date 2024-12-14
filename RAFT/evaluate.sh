# invoke two evaluation task on GSM8K and MATH and save result into eval_result dir

mkdir -p eval_result
python eval_gsm8k.py \
    --model $1 \
    --data_file data/test/GSM8K_test.jsonl \
    --batch_size 1000 \
    --tensor_parallel_size $2 \
    --output_dir gsm8k_$3_$1.json \

python eval_math.py \
    --model $1 \
    --data_file data/test/MATH_test_500.jsonl \
    --batch_size 1000 \
    --tensor_parallel_size $2 \
    --output_dir math_$3_$1.json \

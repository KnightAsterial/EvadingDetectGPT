export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$SCRATCH/.cache/huggingface/datasets
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/test_baseline"

# python maml.py --device gpu --batch_size 1 --num_support 4 --checkpoint_step 60000 --log_dir "/scratch/users/ryanzhao/EvadingDetectGPT/logs/maml/evadegpt.support_4.query_1.inner_steps_1.inner_lr_0.4.learn_inner_lrs_False.outer_lr_0.001.batch_size_2.iters_300000" --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp6k"
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp0"
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp0_noinnerloop" --test_skip_innerloop
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_humanai"

# python maml.py --device gpu --batch_size 2 --num_support 4 --num_train_iterations 300000 --checkpoint_step 30000
# python main.py
# python evaluate.py --method "modeltest_cp6k" --ai_label "ai_sample" --human_label "rephrased_sample" --dataset_dir "./modeltest_cp6k"
# python evaluate.py --method "modeltest_cp0_nil" --ai_label "ai_sample" --human_label "rephrased_sample" --dataset_dir "./modeltest_cp0_noinnerloop"
# python evaluate.py --method "test_480k" --ai_label "ai_sample" --human_label "rephrased_sample" --dataset_dir "./test_multitask_cp480k"
python evaluate.py --method "test_480k" --ai_label "rephrased_sample" --human_label "human_sample" --dataset_dir "./test_multitask_cp480k"

# python multitask_learning.py --device gpu --batch_size 1 --num_support 10 --num_query 1 --max_num_edits 11 --checkpoint_step 27000 --num_train_iterations 2000000
# python multitask_learning.py --device gpu --batch_size 1 --num_support 10 --num_query 1 --max_num_edits 11 --outer_lr 0.01 --checkpoint_step 36000 --num_train_iterations 2000000
# python multitask_learning.py --device gpu --batch_size 1 --num_support 4 --num_query 1 --max_num_edits 11 --checkpoint_step 27000 --log_dir "/scratch/users/ryanzhao/EvadingDetectGPT/logs/multitask/evadegpt.support_10.query_1.outer_lr_0.001.batch_size_1.iters_2000000" --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/multitasktest_cp27k_f32"
# python multitask_learning.py --device gpu --batch_size 1 --num_support 4 --num_query 1 --max_num_edits 11 --checkpoint_step 594000 --log_dir "/scratch/users/ryanzhao/EvadingDetectGPT/logs/multitask/evadegpt.support_10.query_1.outer_lr_0.001.batch_size_1.iters_3000000" --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/multitasktest_cp594k"
# python multitask_learning.py --device gpu --batch_size 1 --num_support 4 --num_query 1 --max_num_edits 11 --checkpoint_step 72000 --log_dir "/scratch/users/ryanzhao/EvadingDetectGPT/logs/multitask/evadegpt.support_10.query_1.outer_lr_0.01.batch_size_1.iters_2000000" --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/multitasktest_highlr"

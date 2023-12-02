export TRANSFORMERS_CACHE=$SCRATCH/.cache/huggingface/transformers
export HF_DATASETS_CACHE=$SCRATCH/.cache/huggingface/datasets
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/test_baseline"

# python maml.py --device gpu --batch_size 1 --num_support 4 --checkpoint_step 60000 --log_dir "/scratch/users/ryanzhao/EvadingDetectGPT/logs/maml/evadegpt.support_4.query_1.inner_steps_1.inner_lr_0.4.learn_inner_lrs_False.outer_lr_0.001.batch_size_2.iters_300000" --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp6k"
# python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp0"
python maml.py --device gpu --batch_size 1 --num_support 4 --test --test_output_dir "/scratch/users/ryanzhao/EvadingDetectGPT/modeltest_cp0_noinnerloop" --test_skip_innerloop

# python maml.py --device gpu --batch_size 2 --num_support 4 --num_train_iterations 300000 --checkpoint_step 30000
# python main.py
# python evaluate.py

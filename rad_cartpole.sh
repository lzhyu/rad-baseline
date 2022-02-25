CUDA_VISIBLE_DEVICES=6 python -u SACv2.py \
--learn_step 2000 --data_aug translate --action_repeat 4 \
--domain_name cartpole --task_name swingup \
--experiment_id rad_test --render_size 100 --input_size 108
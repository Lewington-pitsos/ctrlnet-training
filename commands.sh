img2dataset --url_list training/laion-100k-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder training/laion-100k-data --processes_count 16 --thread_count 128 --image_size 768\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb 

img2dataset --url_list training/laion-20-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder training/laion-20-data --processes_count 1 --thread_count 4 --image_size 768\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb --number_sample_per_shard=2

python ./tutorial_train_sd21.py --model_config_path=models\cldm_v21_singleton.yaml --resume_path=models\ctrl.ckpt
python ./tutorial_train_sd21.py --model_config_path=models\cldm_v21_singleton.yaml --resume_path=models\ctrl.ckpt --train_url="training/laion-20-data/{00000..00008}.tar" --test_url="training/laion-20-data/00009.tar"


python ./tool_add_control_sd21.py models/512.ckpt ./checkpoints/cldm_v21.ckpt             ./models/cldm_v21.yaml && python ./tool_add_control_sd21.py models/512.ckpt ./checkpoints/cldm_v21_normal_conv.ckpt ./models/cldm_v21_normal_conv.yaml && python ./tool_add_control_sd21.py models/512.ckpt ./checkpoints/cldm_v21_singleton.ckpt    ./models/cldm_v21_singleton.yaml && python ./tool_add_control_sd21.py models/512.ckpt ./checkpoints/cldm_v21_rising.ckpt      ./models/cldm_v21_rising.yaml


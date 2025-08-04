python visualize_attention.py \
  --arch vit_small \
  --patch_size 8 \
  --pretrained_weights dino_outputs_leukemia_patch4/checkpoint0040.pth \
  --image_path /home/aic/leukemia/datasets/leukemia_dataset/test/KSC_0007.jpg \
  --image_size 480 480 \
  --output_dir attention_output \
  --checkpoint_key teacher \
  --threshold 0.6

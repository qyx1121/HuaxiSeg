# CUDA_VISIBLE_DEVICES=6 python train.py --num_classes 3 --max_epoch 600 \
# --batch_size 36 --img_size 256 --task_name bvr --vmf_loss

CUDA_VISIBLE_DEVICES=6 python train.py --num_classes 3 --max_epoch 600 \
--batch_size 36 --img_size 256 --task_name bvr --vmf_loss


# CUDA_VISIBLE_DEVICES=0 python train.py --num_classes 3 --max_epoch 600 \
# --batch_size 36 --img_size 256 --task_name evans & \
# CUDA_VISIBLE_DEVICES=1 python train.py --num_classes 3 --max_epoch 600 \
# --batch_size 36 --img_size 256 --task_name evans --vmf_loss \
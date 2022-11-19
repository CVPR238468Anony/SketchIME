python train.py \
    -project limit \
    -dataset sketch \
    -dataroot data \
    -epochs_base 20 \
    -lr_base 0.0002 \
    -lrg 0.0002  \
    -gamma 0.3 \
    -gpu 1 \
    -model_dir ./params/pretrain.pth \
    -temperature 16 \
    -schedule Milestone \
    -batch_size_base 64 \
    -milestones 2 4 6 \
    -num_tasks 32 \
    >> SketchResult.txt

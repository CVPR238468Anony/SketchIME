python train.py \
    -projec fact \
    -dataset sketch \
    -dataroot data \
    -base_mode "ft_cos" \
    -new_mode "avg_cos" \
    -gamma 0.1 \
    -lr_base 0.1 \
    -lr_new 0.1 \
    -decay 0.0005 \
    -epochs_base 60 \
    -schedule Cosine \
    -gpu 1 \
    -temperature 16 \
    -batch_size_base 256 \
    -balance 0.01 \
    -loss_iter 0 \
    -alpha 0.5 \
    >> SketchResult.txt
echo "This script will perform resampling on a trained model\n"

cd ..
python main.py \
    --white \
    --no_pgd_optim \
    --hidden_init \
    --batch_size=128 \
    --LAMBDA=0.05 \
    --carlini_loss \
    --prepared_data='dataloader/128_prepared_data.pickle' \
    --diff_nn \
    --nearest_neigh_all \
    --load_model \
    --adv_model_path='saved_models/adv_model.pt' \
    --namestr="Carlini BMD text resample train 1" \
    --resample_iterations=100 \
    --comet \
    --offline_comet \
    --resample_test


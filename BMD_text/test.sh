# Useful to test adv attacks

# Do 20 internal iterations or until samples are fooled
python main.py --white \
    --no_pgd_optim \
    --hidden_init \
    --namestr="carlini_Text" \
    --LAMBDA=0.01 \
    --carlini_loss \
    --save_adv_samples \
    --batch_size=16 \
    --prepared_data='dataloader/16_prepared_data.pickle' \
    --diff_nn \
    --max_batches 1 \
    --max_iter 1 \
    --save_model \
    --comet \
    --offline_comet

# test temperature decay on few batches
#python main.py --white --no_pgd_optim --hidden_init --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --save_adv_samples --batch_size=32 --prepared_data='dataloader/32_prepared_data.pickle' --nearest_neigh_all --diff_nn --temp_decay_schedule 20 --max_batches 20 --comet


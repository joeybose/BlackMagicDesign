# Useful command to debug default white box attack
# Does single batch per epoch, single adversarial iteration
# Will save adversarial samples to "temp/adv_samples.txt"

# Only 1 iteration, only 1 batch
#python -m pudb main.py \
python main.py \
    --white \
    --no_pgd_optim \
    --hidden_init \
    --batch_size=16  \
    --namestr="carlini_Text" \
    --LAMBDA=0.01 \
    --carlini_loss \
    --max_iter 1 \
    --max_batches 1 \
    --save_adv_samples \
    --prepared_data='dataloader/16_prepared_data.pickle' \
    --diff_nn \
    --load_model \
    --temp_decay_schedule 1 \
    --resample_test

# Do 20 internal iterations or until samples are fooled
#python -m pudb main.py --white --no_pgd_optim --hidden_init --batch_size=128  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --max_batches 1 --save_adv_samples

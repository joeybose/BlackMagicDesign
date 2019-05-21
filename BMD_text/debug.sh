# Useful command to debug default white box attack
# Does single batch per epoch, single adversarial iteration
# Will save adversarial samples to "temp/adv_samples.txt"

# Only 1 iteration, only 1 batch
#python main.py \
python -m pudb main.py \
    --white \
    --no_pgd_optim \
    --hidden_init \
    --batch_size=16  \
    --LAMBDA=0.01 \
    --carlini_loss \
    --max_iter 1 \
    --max_batches=100 \
    --save_adv_samples \
    --prepared_data='dataloader/new_16_prepared_data.pickle' \
    --load_model \
    --adv_model_path='saved_models/adv_model_lambda005_2gpus.pt' \
    --namestr="resample testing" \
    --resample_iterations=5 \
    --diff_nn \
    --resample_test

    #--temp_decay_schedule 5 \
    #--no_parallel \
    #--load_model \

# Do 20 internal iterations or until samples are fooled
#python -m pudb main.py --white --no_pgd_optim --hidden_init --batch_size=128  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --max_batches 1 --save_adv_samples

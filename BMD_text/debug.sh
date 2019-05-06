# Useful command to debug default white box attack
# Does single batch per epoch, single adversarial iteration
# Will save adversarial samples to "temp/adv_samples.txt"

# Only 1 iteration, only 1 batch
python -m pudb main.py --white --no_pgd_optim --no_parallel --hidden_init --batch_size=32  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --max_iter 1 --max_batches 1 --save_adv_samples --prepared_data='dataloader/32_prepared_data.pickle'

# Only 1 iteration, all neighbours
#python -m pudb main.py --white --no_pgd_optim --no_parallel --hidden_init --batch_size=32  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --max_iter 1 --max_batches 1 --save_adv_samples --prepared_data='dataloader/32_prepared_data.pickle' --nearest_neigh_all

# Do 20 internal iterations or until samples are fooled
#python -m pudb main.py --white --no_pgd_optim --hidden_init --batch_size=128  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --max_batches 1 --save_adv_samples

# Useful to test adv attacks

# Do 20 internal iterations or until samples are fooled
python main.py --white --no_pgd_optim --hidden_init --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --save_adv_samples --batch_size=64 --prepared_data='dataloader/64_prepared_data.pickle' --nearest_neigh_all --comet


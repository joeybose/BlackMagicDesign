# Useful to test adv attacks

# Do 20 internal iterations or until samples are fooled
python main.py --white --no_pgd_optim --hidden_init --batch_size=128  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss  --save_adv_samples --nearest_neigh_all

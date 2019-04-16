# Black Magic Design

This repo contains code for Domain Agnostic Black-Box Adversarial Attacks. The current codebase is Research code and is subject to change. Currently, the code is only for vision tasks. Future goals include text and graphs.

## Requirements

comet_ml (latest version) \
pytorch==1.0.0 \
PIL \
numpy \
json \
argparse \
tqdm \
[advertorch](https://github.com/BorealisAI/advertorch) \
os

## Sample Commands
```bash
# Eg run on CIFAR
python main.py --cifar --white --batch_size=256
--namestr="Cifar Carlini Adam Gen Epsilon=0.1"
--epsilon=0.1 --comet --carlini_loss
--test_batch_size=300

# Eg NLP run
cd BMD_text
python main.py --white --no_pgd_optim --hidden_init --batch_size=8 --namestr="BMD Text" --LAMBDA=10

python main.py --white --no_pgd_optim --hidden_init --batch_size=128  --namestr="carlini_Text" --LAMBDA=0.01 --carlini_loss --comet
```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


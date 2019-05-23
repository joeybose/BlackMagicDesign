# Black Magic Design

This repo contains code for Domain Agnostic Black-Box Adversarial Attacks. The current codebase is Research code and is subject to change. Currently, the code is only for vision tasks. Future goals include text and graphs.

## Requirements
Assuming you are in a clean Python 3.6 environment, to install all required packages and download required data, simply run:
```
bash setup.sh
```

This will download a pretrained LSTM, Glove embeddings, IMDB data, install required packages including:
```
comet_ml (latest version) \
pytorch==1.0.1 \
PIL \
numpy \
json \
tqdm \
[advertorch](https://github.com/BorealisAI/advertorch) \
```


## Sample Commands
A basic command for white box attack on the CIFAR dataset
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

## Reproducibility
### Text experiments
To train the adversarial model


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## References
- [Distances for KNN](https://arxiv.org/pdf/1708.04321.pdf)

## License
[MIT](https://choosealicense.com/licenses/mit/)


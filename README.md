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
DGL\
networkx\
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

# Eg Graph run
cd BMD_graph
# DAGAE RUN
python main.py --no_pgd_optim --carlini_loss --white --attack_epochs=200 --LAMBDA=1e-2 --dataset=citeseer --deterministic_G --namestr="CiteSeer Deterministic Carlini Node Graph Direct" --comet

#DAG-Attack Direct Run
python main.py --no_pgd_optim --carlini_loss --white --attack_epochs=200 --LAMBDA=1e-2 --dataset=citeseer --namestr="CiteSeer Deterministic Carlini Node Graph Direct" --comet

#DAG-Attack Direct Run with Resampling
python main.py --no_pgd_optim --carlini_loss --white --attack_epochs=200 --LAMBDA=1e-2 --dataset=citeseer --namestr="CiteSeer Deterministic Carlini Node Graph Direct" --resample_test --comet

#DAG-Attack Influencer Run
python main.py --no_pgd_optim --carlini_loss --white --attack_epochs=200 --LAMBDA=1e-2 --dataset=citeseer --namestr="CiteSeer Deterministic Carlini Node Graph Direct" --single_node_attack -- influencer_attack --comet
```

## Reproducibility

### Image Experiments
Running main.py with the example runs will first train a VGG model (you should change the number of training epochs to something small in the code) then it will automatically execute the attack on the trained model.

### Text experiments
The following scripts will reproduce the results in Table 4: Adversarial success rates on IMDB

Train DAG-autoencoder:
```
cd BMD_text/runs/
bash autoencoder.sh
```

Train DAG-autoencoder with differentiable nearest neighbour:
```
cd BMD_text/runs/
bash autoencoder_diff.sh
```

Train DAG-VAE:
```
cd BMD_text/runs/
bash dag_attack.sh
```

Train DAG-VAE with differentiable nearest neighbour:
```
cd BMD_text/runs/
bash dag_attack_diff.sh
```

Test resampling generalization on DAG-VAE-diff:
```
cd BMD_text/runs/
bash resample.sh
```
### Graph Experiments
cd BMD_graph/
You will have to train the target model first on each dataset using main_graph_classifier.py
python main_graph_classifier.py --dataset cora --gpu 0 --self-loop

After use any of the sample commands to reproduce table results

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## References
- [Distances for KNN](https://arxiv.org/pdf/1708.04321.pdf)

## License
[MIT](https://choosealicense.com/licenses/mit/)


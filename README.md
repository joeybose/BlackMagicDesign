# Black Magic Design

This repo contains code for Domain Agnostic Black-Box Adversarial Attacks. The current codebase is Research code and is subject to change. Currently, the code is only for vision tasks. Future goals include text and graphs.

## Requirements
In addition to default packages, the following are necessary:
```
comet_ml (latest version) \
pytorch==1.0.1 \
PIL \
numpy \
json \
tqdm \
[advertorch](https://github.com/BorealisAI/advertorch) \
```

The version details are specified in `requirements.txt`. To ensure reproducibility, in a new conda or python environment running python 3.6+, install dependencies matching the development machine:
```
bash setup.sh
```

## Sample Commands
A basic command for white box attack on the CIFAR dataset
```bash
python main.py --cifar --white --batch_size=256
--namestr="Cifar Carlini Adam Gen Epsilon=0.1"
--epsilon=0.1 --comet --carlini_loss
--test_batch_size=300
```

## Reproducibility
### Text experiments
To train..


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)


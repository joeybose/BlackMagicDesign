import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import models
from torch import nn, optim
from torchvision.models import resnet50
import json

def white_box_untargeted(args, model, image, normalize):
    source_class = 341 # pig class
    epsilon = 2./255
    # Create noise vector
    delta = torch.zeros_like(image, requires_grad=True)
    # Optimize noise vector (only) to fool model
    opt = optim.SGD([delta], lr=1e-1)

    for t in range(30):
        pred = model(normalize(pig_tensor + delta))
        loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([source_class]))
        if t % 5 == 0:
            print(t, loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        # Clipping is equivalent to projecting back onto the l_\infty ball
        # This technique is known as projected gradient descent (PGD)
        delta.data.clamp_(-epsilon, epsilon)
    return pred, delta


def load_unk_model():
    """
    Load an unknown model. Used for convenience to change models
    """
    # load pre-trained ResNet50
    model = resnet50(pretrained=True)
    model.eval();
    return model

def main(args):
    # Normalize image for ImageNet
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # The unknown model to attack
    unk_model = load_unk_model()

    # Attack model
    model = models.BlackAttack(args.input_size, args.latent_size)

    # Control Variate


if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser(description='BlackBox')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test', default=False, action='store_true',
                        help='just test model and print accuracy')
    parser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                        help='where to save/load')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.device = torch.device("cuda" if use_cuda else "cpu")

    main()


from torch import nn, optim

class Normalize(nn.Module):
    """
    Normalize an image as part of a torch nn.Module
    """
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def display_tensor(tensor):
    plt.imshow((tensor)[0].detach().numpy().transpose(1,2,0))
    plt.show()


def load_imagenet_classes():
    with open("references/adver_robust/introduction/imagenet_class_index.json") as f:
        imagenet_classes = {int(i):x[1] for i,x in json.load(f).items()}
    return imagenet_classes




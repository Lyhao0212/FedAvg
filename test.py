import torch
from torch import nn
from models import ResNet34,TwoHiddenLayerFc,LogisticRegression


def test(test_loader,para,arg):

    device = torch.device(arg.device)

    model = TwoHiddenLayerFc(arg.input_dim,arg.num_class).eval()
    model.to(device)
    model.load_state_dict(para)
    loss_func=nn.CrossEntropyLoss()
    test_loss=0
    step=0
    total=0
    correct=0
    with torch.no_grad():
        for images, labels in test_loader:
            images=images.reshape(-1,28*28)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            test_loss += loss_func(output, labels).item()
            values, predicte = torch.max(output, 1)
            total += labels.size(0)
            step += 1
            correct += (predicte == labels).sum().item()
    return 100 * correct / total, test_loss / step
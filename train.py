import torch
from torch import nn
from models import ResNet34,TwoHiddenLayerFc,LogisticRegression


def train_init(train_loader,arg):
    device = torch.device(arg.device)
    model = TwoHiddenLayerFc(arg.input_dim,arg.num_class).train()
    model.to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=arg.lr)
    train_loss=0
    step=0
    loss_func=nn.CrossEntropyLoss()
    for epoch in range(arg.epoches):

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.reshape(-1, 28 * 28)
            output = model(images)
            loss = loss_func(output, labels)
            train_loss += loss.item()
            step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    para=model.state_dict()
    return para,train_loss/step
def train(train_loader,com,arg):

    device = torch.device(arg.device)

    model = TwoHiddenLayerFc(arg.input_dim,arg.num_class).train()

    model.to(device)
    model.load_state_dict(com)

    optimizer=torch.optim.SGD(model.parameters(),lr=arg.lr)
    train_loss=0
    step=0
    loss_func=nn.CrossEntropyLoss()
    for epoch in range(arg.epoches):
        for images,labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            images=images.reshape(-1,28*28)
            output = model(images)
            loss = loss_func(output, labels)
            train_loss += loss.item()
            step += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    para=model.state_dict()
    return para,train_loss/step

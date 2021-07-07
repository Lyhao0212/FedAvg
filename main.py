import random
import time
from tensorboardX import SummaryWriter
from train import train_init,train
from test import test
from utils import set_seed,build_dataset,com_para


writer=SummaryWriter("./result")
class Arguments():
    def __init__(self):
        self.batch_size =50
        self.test_batch_size =1024
        self.epoches =1
        self.Round=300
        self.C=0.1
        self.K=100
        self.lr=0.1
        self.device="cuda"
        self.input_dim=28*28
        self.num_class=10


def main():
    set_seed(2021)
    arg = Arguments()
    max_length = max([len(key) for key in arg.__dict__.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(arg.__dict__.items()):
        print(fmt_string % keyPair)

    Datalist, test_loader = build_dataset(arg)
    client_num = int(arg.K * arg.C)
    index=random.sample(range(arg.K),client_num)


    print(">>> Training...")
    print("=========================================================================================================")
    acclist = []
    train_losslist = []
    test_losslist = []
    starttime = time.time()

    paralist = []
    train_loss = 0
    for client in range(client_num):
        para,loss = train_init(Datalist[index[client]], arg)
        paralist.append(para)
        train_loss += loss
    train_loss = train_loss / client_num

    com = com_para(paralist,arg)
    acc, test_loss = test(test_loader, com, arg)
    writer.add_scalar("acc",acc,global_step=0)
    writer.add_scalar("test_loss",test_loss,global_step=0)
    writer.add_scalar("train_loss",train_loss,global_step=0)
    print(">>> Round:   {} / Acc:   {}% /Train_loss:   {} / Test_loss:   {} / Time:   {}s "
          .format(1, acc, round(train_loss, 4), round(test_loss, 4), round(time.time() - starttime, 3)))
    print("=========================================================================================================")

    acclist.append(acc)
    test_losslist.append(test_loss)
    train_losslist.append(train_loss)

    for epoch in range(arg.Round - 1):
        starttime = time.time()
        index = random.sample(range(arg.K), client_num)
        paralist = []
        train_loss = 0
        for client in range(client_num):
            para, loss = train(Datalist[index[client]], com,arg)
            paralist.append(para)
            train_loss += loss
        train_loss = train_loss / client_num
        com = com_para(paralist, arg)
        acc, test_loss = test(test_loader, com, arg)
        print(">>> Round:   {} / Acc:   {}% /Train_loss:   {} / Test_loss:   {} / Time:   {}s "
              .format(epoch + 2, acc, round(train_loss, 4), round(test_loss, 4), round(time.time() - starttime, 3),
                      ))
        print(
            "=========================================================================================================")
        acclist.append(acc)
        test_losslist.append(test_loss)
        train_losslist.append(train_loss)

        writer.add_scalar("acc", acc, global_step=epoch+1)
        writer.add_scalar("test_loss", test_loss, global_step=epoch+1)
        writer.add_scalar("train_loss", train_loss, global_step=epoch+1)
        if (epoch + 2) % 100 == 0:
            print(test_losslist)
            print(acclist)
            print(train_losslist)

if __name__=="__main__":
    main()

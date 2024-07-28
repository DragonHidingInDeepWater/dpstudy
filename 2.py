import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data = pd.read_csv("dccc_prepared.csv")
x = data.iloc[:,:-1]
y= data["default payment next month"]

X_new, X_test, y_new, y_test = train_test_split( \
    x, y, test_size=0.2, random_state=0)
dev_per = X_test.shape[0]/X_new.shape[0]
X_train, X_dev, y_train, y_dev = train_test_split( \
    X_new, y_new, test_size=dev_per, random_state=0)
print("Training sets:",X_train.shape, y_train.shape)
print("Validation sets:",X_dev.shape, y_dev.shape)
print("Testing sets:",X_test.shape, y_test.shape)

X_dev_torch = torch.tensor(X_dev.values).float().to(device)
y_dev_torch = torch.tensor(y_dev.values).to(device)
X_test_torch = torch.tensor(X_test.values).float().to(device)
y_test_torch = torch.tensor(y_test.values).to(device)

class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_1 = nn.Linear(input_size, 100);
        self.hidden_2 = nn.Linear(100, 100);
        self.hidden_3 = nn.Linear(100, 50);
        self.hidden_4 = nn.Linear(50,50);
        self.output = nn.Linear(50, 2);
        '''
        在 PyTorch 中，nn.Dropout 是一个用于减少神经网络过拟合的模块。
        self.dropout = nn.Dropout(p=0.1) 这一行代码创建了一个 dropout 层实例，
        其中参数 p=0.1 指定了 dropout 的比率，即在前向传播过程中，每个神经元有 10% 的概率被临时“关闭”或“丢弃”。
        Dropout 的作用
        Dropout 技术是一种正则化方法，旨在减少神经网络的过拟合。它通过在训练过程中随机地、暂时地将一部分神经元的输出设置为 0，
        以此来模拟神经网络的稀疏性。这样做的目的是迫使网络的其他部分学习更加鲁棒和独立的特征，减少神经元间的相互依赖，从而提高模型的泛化能力。
        '''
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        o = self.dropout(F.relu(self.hidden_1(x)));
        o = self.dropout(F.relu(self.hidden_2(o)));
        o = self.dropout(F.relu(self.hidden_3(o)));
        o = self.dropout(F.relu(self.hidden_4(o)));
        result = F.log_softmax(self.output(o) ,dim = 1);
        return  result;

model = Classifier(X_train.shape[1]).to(device);
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001);
epochs = 4000
batch_size = 128


train_losses, dev_losses, train_acc, dev_acc = [], [], [], []
x_axis = []
for e in range(1, epochs+1):
    X_, y_ = shuffle(X_train, y_train)
    running_loss = 0
    running_acc = 0
    iterations = 0
    for i in range(0, len(X_), batch_size):
        iterations += 1
        b = i + batch_size
        X_batch = torch.tensor(X_.iloc[i:b,:].values).float().to(device)
        y_batch = torch.tensor(y_.iloc[i:b].values).to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        '''
        模型的输出 pred 通常是经过 log_softmax 函数处理的，这意味着输出是每个类别的对数概率。为了得到真正的概率，
        我们需要使用 torch.exp() 函数将对数概率转换为概率值。这一步是必要的，因为接下来我们要找出每个样本的最大概率对应的类别。
        '''
        ps = torch.exp(pred)
        '''
        topk 函数用于获取沿着给定维度的 k 个最大元素。在这里，k=1，意味着我们只对每个样本的最大概率感兴趣，
        dim=1 表示我们沿着张量的第二个维度（即类别维度）寻找最大值。top_p 存储了每个样本的最大概率值，而 top_class 存储了这些最大概率对应的类别索引。

        '''
        top_p, top_class = ps.topk(1, dim=1)
        top_class_cpu = top_class.cpu()
        y_batch_cpu = y_batch.cpu()
        '''
        accuracy_score 函数计算了预测类别 top_class_cpu 和真实类别 y_batch_cpu 之间的准确率。running_acc 是一个累积变量，
        用于在整个 epoch 中累计准确率。这里，我们将当前批次的准确率加到了 running_acc 上。
        '''
        running_acc += accuracy_score(y_batch_cpu.numpy(), top_class_cpu.numpy())
    dev_loss = 0
    acc = 0
    with torch.no_grad():
        model.eval();
        pred_dev = model(X_dev_torch)
        dev_loss = criterion(pred_dev, y_dev_torch)
        ps_dev = torch.exp(pred_dev)
        top_p, top_class_dev = ps_dev.topk(1, dim=1)
        top_class_dev_cpu = top_class_dev.cpu()
        y_dev_torch_cpu = y_dev_torch.cpu()
        acc = accuracy_score(y_dev_torch_cpu, top_class_dev_cpu.numpy())
    model.train()
    if e%50 == 0 or e == 1:
        x_axis.append(e)
        train_losses.append(running_loss/iterations)
        dev_losses.append(dev_loss.cpu())
        train_acc.append(running_acc/iterations)
        dev_acc.append(acc)
        print("Epoch: {}/{}.. ".format(e+1, epochs), "Training Loss: {:.3f}.. ".format(running_loss/iterations), "Validation Loss: {:.3f}.. ".format(dev_loss), "Training Accuracy: {:.3f}.. ".format(running_acc/iterations),"Validation Accuracy: {:.3f}".format(acc))

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_losses, label='Training loss')
plt.plot(x_axis, dev_losses, label='Validation loss')
plt.legend(frameon=False, fontsize=15)
plt.show()

fig = plt.figure(figsize=(15, 5))
plt.plot(x_axis, train_acc, label="Training accuracy")
plt.plot(x_axis, dev_acc, label="Validation accuracy")
plt.legend(frameon=False, fontsize=15)
plt.show()

model.eval()
test_pred = model(X_test_torch)
test_pred = torch.exp(test_pred)
top_p, top_class_test = test_pred.topk(1, dim=1)
acc_test = accuracy_score(y_test_torch, top_class_test)
print(acc_test)

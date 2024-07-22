import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#创建随机值的虚拟输入数据（x）和仅包含零和一的虚拟目标数据（y）。 张量x的大小应为(20, 10)，而y的大小应为(20, 1)：
x = torch.randn(20,10).to(device)
y = torch.randint(0,2, (20,1)).type(torch.FloatTensor).to(device)

#定义输入数据的特征数为10（input_units），输出层的节点数为1（output_units）。
input_units = 10
output_units = 1

model = nn.Sequential(nn.Linear(input_units, output_units), nn.Sigmoid()).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

loss_funct = nn.MSELoss()

losses = []
for i in range(20):
    y_pred = model(x)
    loss = loss_funct(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%5 == 0:
        print(i, loss.item())

plt.plot(range(0,20), losses)
plt.show()


'''
导入所需的库，包括用于读取 CSV 文件的 Pandas。

读取包含数据集的 CSV 文件。

注意

建议使用 Pandas 的read_csv函数加载 CSV 文件。 要了解有关此函数的更多信息，请访问这里。

将输入特征与目标分开。 请注意，目标位于 CSV 文件的第一列中。 接下来，将值转换为张量，确保将值转换为浮点数。

注意

要切片 pandas DataFrame，请使用 pandas 的iloc方法。 要了解有关此方法的更多信息，请访问这里。

定义模型的架构，并将其存储在名为model的变量中。 记住要创建一个单层模型。

定义要使用的损失函数。 在这种情况下，请使用 MSE 损失函数。

定义模型的优化器。 在这种情况下，请使用 Adam 优化器，并将学习率设为0.01。

对 100 次迭代运行优化，保存每次迭代的损失值。 每 10 次迭代打印一次损失值。

绘制线图以显示每个迭代步骤的损失值。
'''
torch.manual_seed(0);
data = pd.read_csv("SomervilleHappinessSurvey2015.csv");
print(data.head());
'''
这一行代码做了以下几件事：
data.iloc[:,1:]：使用 iloc 方法从 data DataFrame 中选取所有的行（: 表示所有行），以及从第二列开始到最后的所有列（1: 表示从下标为 1 的列开始，直到最后一列）。这通常意味着选取除了第一列（可能是ID或标签列）之外的所有特征列。
.values：将选取的 DataFrame 部分转换为 NumPy 数组。
torch.tensor(...)：将 NumPy 数组转换为 PyTorch 的 Tensor。
.float()：确保 Tensor 的数据类型是浮点型，这对于大多数机器学习和深度学习操作来说是必需的。
'''
x = torch.tensor(data.iloc[:,1:].values).float().to(device);
y = torch.tensor(data.iloc[:,:1].values).float().to(device);

model=nn.Sequential(nn.Linear(6,1), nn.Sigmoid()).to(device);
loss_funct = torch.nn.MSELoss();
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
losses = []
for i in range(100):
    y_pred = model(x);
    '''
    在这段代码中，loss.backward() 方法用于计算损失函数相对于模型参数的梯度，并将这些梯度存储在每个参数的 .grad 属性中。
    当 loss.backward() 被调用时，PyTorch 会自动利用计算图（compute graph）来执行反向传播算法，计算出损失函数关于所有可训练参数的梯度。
    ptimizer 对象并不直接与梯度计算过程交互，而是依赖于参数的 .grad 属性中的梯度值来进行参数更新。在调用了 loss.backward() 后，每个参数的梯度都会被计算并存储
    '''
    loss = loss_funct(y_pred, y);
    losses.append(loss.item());
    optimizer.zero_grad();
    loss.backward();
    optimizer.step()

    if i%2 == 0:
        print(i, loss.item())


plt.plot(range(0,100), losses)
plt.show()






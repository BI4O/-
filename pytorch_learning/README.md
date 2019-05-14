# Pytorch 板块

#### 1. 什么是Pytorch, 为什么要选择Pytorch?

- Pytorch是最年轻的深度学习框架之一，由Facebook公司开发，一经推出就非常受欢迎

- Pytorch的语法非常接近numpy , 而且它们之间的数据转换也很方便
  一句话总结，Pytorch是很简洁，很适合初学者入门深度学习的框架

#### 2. Pytorch的安装

- 由于我使用的是Anaconda python=3.7的全家桶，所以我只需要打开Anaconda Prompt输入

  `conda install pytorch-cpu torchvision-cpu -c pytorch`

  如果你跟我一样第一次安装失败了，可能原因是conda版本太旧了，输入

  `conda update conda`

#### 3. 配置Pytorch的环境

- 为了以后方便管理，可以在conda中专门建一个专用的虚拟环境

  查看下conda中已有的环境

  `conda env list`

- 新建一个名为your_env_name（随便你起）的虚拟环境，最好指定python的版本如3.7

  `conda create -n your_env_name python=3.7`

  激活你创建的虚拟环境

  `activate your_env_name`

  经过激活后，你再输入安装某些包的命令，这样才可已安装到你选定的环境，

  否则都默认安装到base环境中了

#### 4. Pytorch的基本概念

- 神经网络

  - 根据你的结构设计，有很多种，简单的有NN, 复杂的有CNN, RNN

    比如一个简单的全连接神经网络NN，需要定义要使用的激活函数relu, 确定输入输出的维度

    （in: 28x28=784, out: 10）

    ```python
    from torch import nn
    import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = nn.Linear(784,128)  # 输入层有784单元，即28x28
            self.h2 = nn.Linear(128, 64)
            self.output = nn.Linear(64, 10)
        
        def forward(self, x):
            x = x.view(x.shape[0],784)  # Vectorize
            x = F.relu(self.h1(x))
            x = F.relu(self.h2(x))
            x = F.softmax(self.output(x), dim=1)  
            return x
    ```

- 损失函数

  - loss_function 损失函数有很多种，常见的交叉熵 CrossEntropy, 均方差 MSE(mean square error)

    和负对数似然 Nllloss(negative log likehood loss)

    ```python
    loss_function = nn.Nllloss()
    ```

- 优化器

  - optimizer 优化器也是个函数，常见的有炼丹神器SGD, 还有傻瓜式大刀Adam

    如果不太熟悉，建议直接用 Adam

    ```python
    from torch import optim
    optimizer = optim.Adam(my_model.parameters(), lr=0.003)
    ```

- 训练

  - 有了前面的三个主要部分就可以开始训练模型

    ```python
    epochs = 10  # 遍历整个数据集10次
    for e in range(epochs):
        # 损失函数容器
        running_loss = 0
        for images, labels in trainloader: # 批次：每次取64个
            # 梯度清零
            optimizer.zero_grad()
            # 向前传播
            output = my_model(images)
            log_output = torch.log(output)
            # 计算损失
            loss = loss_function(log_output, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        
        print(f'Training loss:{running_loss/len(trainloader)}')
    ```

#### 5.Pytorch代码实践

- 神经网络实现[手写数字识别](https://github.com/BI4O/ML_git_repos/blob/master/pytorch_learning/pytorch%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB.ipynb)

* 神经网络实现[线性回归](https://github.com/BI4O/ML_git_repos/blob/master/pytorch_learning/Pytorch%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AE%9E%E7%8E%B0%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92.ipynb)

* 神经网络实现[逻辑回归](https://github.com/BI4O/ML_git_repos/blob/master/pytorch_learning/Pytorch%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AE%9E%E7%8E%B0%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92.ipynb)
* 神经网络实时[Dropout正则化避免过拟合](https://github.com/BI4O/ML_git_repos/blob/master/pytorch_learning/Pytorch%E6%9C%8D%E9%A5%B0%E5%88%86%E7%B1%BB%E5%AE%9E%E8%B7%B5%EF%BC%88%E6%B7%BB%E5%8A%A0%E6%AD%A3%E5%88%99%E5%8C%96%E9%81%BF%E5%85%8D%E8%BF%87%E6%8B%9F%E5%90%88%EF%BC%89.ipynb)


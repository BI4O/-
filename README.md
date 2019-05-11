# Pytorch 板块

#### 1. 什么是Pytorch, 为什么要选择Pytorch?

* Pytorch是最年轻的深度学习框架之一，由Facebook公司开发，一经推出就非常受欢迎

* Pytorch的语法非常接近numpy , 而且它们之间的数据转换也很方便
  一句话总结，Pytorch是很简洁，很适合初学者入门深度学习的框架

#### 2. Pytorch的安装

* 由于我使用的是Anaconda python=3.7的全家桶，所以我只需要打开Anaconda Prompt输入

  `conda install pytorch-cpu torchvision-cpu -c pytorch`

  如果你跟我一样第一次安装失败了，可能原因是conda版本太旧了，输入

  `conda update conda`

#### 3. 配置Pytorch的环境

* 为了以后方便管理，可以在conda中专门建一个专用的虚拟环境

  查看下conda中已有的环境

  `conda env list`

* 新建一个名为your_env_name（随便你起）的虚拟环境，最好指定python的版本如3.7

  `conda create -n your_env_name python=3.7`

  激活你创建的虚拟环境

  `activate your_env_name`

  经过激活后，你再输入安装某些包的命令，这样才可已安装到你选定的环境，

  否则都默认安装到base环境中了

#### 4. Pytorch的基本概念

* 神经网络

  * 根据你的结构设计，有很多种，简单的有NN, 复杂的有CNN, RNN

    比如一个简单的全连接神经网络NN，需要定义要使用的激活函数relu, 确定输入输出的维度

    （in: 28x28=784, out: 10）

    ~~~python
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
    ~~~

* 损失函数

  * loss_function 损失函数有很多种，常见的交叉熵 CrossEntropy, 均方差 MSE(mean square error)

    和负对数似然 Nllloss(negative log likehood loss)

    ~~~python
    loss_function = nn.Nllloss()
    ~~~

* 优化器

  * optimizer 优化器也是个函数，常见的有炼丹神器SGD, 还有傻瓜式大刀Adam

    如果不太熟悉，建议直接用 Adam

    ~~~python
    from torch import optim
    optimizer = optim.Adam(my_model.parameters(), lr=0.003)
    ~~~

* 训练

  * 有了前面的三个主要部分就可以开始训练模型

    ~~~python
    epochs = 10  # 遍历整个数据集5次
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
    ~~~

#### 5.通过代码来实现深度学习过程

 * 神经网络实现[手写数字识别](https://github.com/BI4O/ML_git_repos/blob/master/pytorch_learning/pytorch%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB.ipynb)


=======
# Welcome to my world !!!

###### Find me by visiting  https://github.com/BI4O

###### Support me by Star [my practice  repos](https://github.com/BI4O/ML_git_repos)

### The more i learn, the happier i am :

* #### Python Learning part

  * [莫烦 Python](https://morvanzhou.github.io/)

* #### Kaggle & Data Science part

  * [Kaggle 官网](www.kaggle.com)
  * [Kaggle 项目实战  (教程) ](https://github.com/BI4O/kaggle)
  * [Modin 使 pandas 可以利用多核](https://github.com/BI4O/modin)

* #### Machine Learning  &  Sklearn part

  * ##### Machine Learning

    * [网易云课堂的 Machine Learning 吴恩达视频教程](https://study.163.com/course/courseMain.htm?courseId=1004570029)
    * [南瓜书 pumpkin_book](https://github.com/BI4O/pumpkin-book)
    * [统计学习方法代码实现](https://github.com/BI4O/statistical-learning-method-)
    * [机器学习初学者  by  黄海广博士](https://github.com/BI4O/machine_learning_beginner)
    * [机器学习一百天](https://github.com/BI4O/100-Days-Of-ML-Code)

  * Sklearn 机器学习工具包

    * [Sklearn 中文文档](http://sklearn.apachecn.org/#/)
    * [📖 [译] Sklearn 与 TensorFlow 机器学习实用指南](https://github.com/BI4O/hands-on-ml-zh)

* #### Deep Learning  &  Pytorch part

  * ##### Deep Learning 学习资料

    * [网易云课堂的 Deep Learning 吴恩达视频教程 ](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)
    * [深度学习500问](https://github.com/BI4O/DeepLearning-500-questions)

  * ##### Pytorch 学习资料

    * [Pytorch 中文文档](https://pytorch.apachecn.org/docs/1.0/#/)
    * [Udacity 的 Pytorch 视频教程](https://cn.udacity.com/course/deep-learning-pytorch--ud188)
    * [另一个 pytorch 中文文档，各种 Pachage 速查手册](https://pytorch-cn.readthedocs.io/zh/latest/)
    * [pytorch-handbook](https://github.com/zergtant/pytorch-handbook)

  * #####  Pytorch 预训练的模型

    * [BigGAN 生成对抗神经网络 by huggingface](https://github.com/BI4O/pytorch-pretrained-BigGAN)
    * [BERT Google自然语言处理模型 by huggingface](https://github.com/BI4O/pytorch-pretrained-BERT)

* #### Other source

  * [清华大学计算机系课程](https://github.com/BI4O/REKCARC-TSC-UHT)
  * [📖 [译] OpenCV 中文文档](https://github.com/BI4O/opencv-doc-zh)
  * [CS224N - Stanford - 2019 深度学习+自然语言处理公开课 ](https://github.com/BI4O/CS224N-Stanford-Winter-2019)
  * [2019届秋招面经集合](https://github.com/BI4O/2019-Autumn-recruitment-experience)
  * [换脸技术 by Deepfakes/faceswap](https://github.com/BI4O/faceswap)
>>>>>>> 08648e3ed7375ae9babd1a423e2391e51f64b2f6

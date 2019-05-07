from torch import nn, optim


from torchvision import datasets, transforms

# 定义数字话转换规则
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])
                                ])
# 下载数据
trainset = datasets.MNIST('MNIST_data/',
                         download=True,
                         train=True,
                         transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                         batch_size=64,
                                         shuffle=True)



model = nn.Sequential(nn.Linear(784 ,128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# 定义损失函数
criterion = nn.NLLLoss()
# 定义优化函数
optimizer = optim.SGD(model.parameters(), lr=0.03)

# 进行损失函数的优化
epochs = 5  # 遍历整个数据集5次
for e in range(epochs):
    # 损失函数容器
    running_loss = 0
    for images, labels in trainloader: # 批次：每次取64个
        # 扁平化数据,64x784
        images_digit = images.view(images.shape[0] ,-1)
        # 梯度清零
        optimizer.zero_grad()
        # 向前传播
        output = model.forward(images_digit)
        # 计算损失
        loss = criterion(output, labels)
        # 反向传播
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f'Training loss:{running_loss/len(trainloader)}')
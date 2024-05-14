from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

# 导入自定义脚本
from mydataset import MyDataset
from spp import SPPNet

# 清空缓存
torch.cuda.empty_cache()

transform = transforms.Compose([transforms.ToTensor()])
train_data = MyDataset(txt_path='Food_Classification_Dataset\\label\\train.txt', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

model = SPPNet()
model.train()

num_epochs = 5
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.cuda()
loss_function.cuda()

losslist = []
for epoch in range(num_epochs):
    for i, (images, label) in enumerate(train_loader):
        images = images.cuda()
        label = label.cuda()
        outputs = model(images)
        loss = loss_function(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losslist.append(loss.item())
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

plt.plot(losslist)
plt.ylabel('Loss')
plt.show()

torch.save(model.state_dict(), 'model\\sppnet.pth')

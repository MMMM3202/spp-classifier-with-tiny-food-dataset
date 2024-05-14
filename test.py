import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
from mydataset import MyDataset
from spp import SPPNet
from sklearn.metrics import accuracy_score, confusion_matrix

transform = transforms.Compose([transforms.ToTensor()])
test_data = MyDataset(txt_path='Food_Classification_Dataset\\label\\test.txt', transform=transform)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

# 加载模型
model = SPPNet()
model.load_state_dict(torch.load('model\\sppnet.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

labelname = ['Bread', 'Hamburger', 'Kebab', 'Noodle', 'Rice']

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels, 1)
        print(predicted, labels)
        total += 1
        if labels == predicted:
            correct += 1
        predicted_labels.append(predicted.item())
        true_labels.append(labels.item())
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

    # 绘制混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


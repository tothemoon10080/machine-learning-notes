import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# 定义数据转换
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 指定本地数据集路径
mnist_path = 'data'

# 加载本地MNIST数据集
train_dataset = datasets.MNIST(root=mnist_path, train=True, transform=transform, download=False)
#test_dataset = datasets.MNIST(root=mnist_path, train=False, transform=transform, download=False)

# 创建训练集、验证集和测试集的数据加载器
batch_size = 2048
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

train_images, train_labels = next(iter(train_loader))
#test_images, test_labels = next(iter(test_loader))

# 将图像数据展平为一维向量
train_images = train_images.view(train_images.size(0), -1)
#test_images = test_images.view(test_images.size(0), -1)

# 创建PCA对象并拟合训练数据
pca = PCA(n_components=2)
train_images_pca = pca.fit_transform(train_images)

# 使用相同的PCA对象转换测试数据
#test_images_pca = pca.transform(test_images)


# 创建一个散点图来显示降维后的数据点
plt.figure(figsize=(8, 6))
plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels, cmap='rainbow')
plt.title('PCA - MNIST Train Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()
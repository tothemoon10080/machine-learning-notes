import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

plt.figure(figsize=(12, 12))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_path = 'data'
train_dataset = datasets.MNIST(root=mnist_path, train=True, transform=transform, download=False)

batch_size = 2048
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_images, train_labels = next(iter(train_loader))
train_images = train_images.view(train_images.size(0), -1)

# PCA
pca = PCA(n_components=2)
train_images_pca = pca.fit_transform(train_images)

plt.subplot(221)
plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels, cmap='rainbow')
plt.title('PCA - MNIST Train Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
train_images_tsne = tsne.fit_transform(train_images)

plt.subplot(222)
plt.scatter(train_images_tsne[:, 0], train_images_tsne[:, 1], c=train_labels, cmap='rainbow')
plt.title('t-SNE - MNIST Train Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()


# K-means
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(train_images_tsne)

train_clusters = kmeans.predict(train_images_tsne)

plt.subplot(223)
plt.scatter(train_images_tsne[:, 0], train_images_tsne[:, 1], c=train_clusters, cmap='rainbow')
plt.title('t-SNE + K-means - MNIST Train Data')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar()
plt.show()
import matplotlib as mlp
import matplotlib.pyplot as plt
from dask.array import reshape
from scipy.cluster.vq import kmeans

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import norm
#%%
image = mlp.image.imread('bird_small.png')
#%%
plt.imshow(image)
#%%
X = image.reshape(-1, 3)
#%%
kmeans = KMeans(n_clusters=5,n_init=10)
#%%
kmeans.fit(X)
#%%
segmented_image = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_image.reshape(image.shape)
#%%
plt.imshow(segmented_image)


print("Final cost (inertia):", kmeans.inertia_)

costs = []
kmeans = KMeans(n_clusters=100, n_init=1, max_iter=1, random_state=0, init='random', verbose=0)
labels = None
for i in range(20):  # 20次迭代
    kmeans.max_iter = 1
    kmeans.fit(X)
    costs.append(kmeans.inertia_)
    # 重新初始化中心点为上一次的结果
    kmeans = KMeans(n_clusters=100, n_init=1, max_iter=1, random_state=0, init=kmeans.cluster_centers_, verbose=0)

plt.figure()
plt.plot(range(1, len(costs)+1), costs, 'bo-')
plt.xlabel('Iteration')
plt.ylabel('Cost (Inertia)')
plt.title('K-means Cost per Iteration')
plt.show()

# 1. 计算每个特征的均值和方差
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
# 2. 计算每个样本的概率密度
p = np.prod(norm.pdf(X, mu, sigma), axis=1)
# 3. 设定阈值 epsilon，判定异常
epsilon = np.percentile(p, 1)  # 取概率最低的1%为异常
anomaly_mask = p < epsilon
# 4. 可视化异常点
anomaly_image = image.copy().reshape(-1, 3)
anomaly_image[~anomaly_mask] = [0, 0, 0]  # 正常像素设为黑色
anomaly_image = anomaly_image.reshape(image.shape)
plt.figure()
plt.imshow(anomaly_image)
plt.title('Anomaly Pixels (Gaussian Model)')
plt.show()
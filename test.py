import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 加载图片
image = mpimg.imread('bird_small.png')
# 生成像素点数据 X
X = image.reshape(-1, 3)

# 假设 X 已经是 (n_samples, n_features) 的二维数组，image 是原始图片
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

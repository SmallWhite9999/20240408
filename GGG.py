from hmmlearn.hmm import GaussianHMM
import numpy as np

# 準備數據（示例數據，實際應用中需要根據具體情況準備數據）
# 假設數據是一個二維數組，每行代表一個樣本，每列代表一個特徵
# 在這個示例中，我們使用隨機生成的數據作為示例
np.random.seed(42)
X = np.random.randn(100, 2)

# 初始化HMM模型
# n_components是隱藏狀態的數量，covariance_type是協方差類型，n_iter是迭代次數
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)

# 訓練模型
model.fit(X)

# 打印模型參數
print("模型參數:")
print("轉移概率矩陣:")
print(model.transmat_)
print("\n初始狀態概率:")
print(model.startprob_)
print("\n每個隱藏狀態的均值:")
print(model.means_)
print("\n每個隱藏狀態的協方差矩陣:")
print(model.covars_)

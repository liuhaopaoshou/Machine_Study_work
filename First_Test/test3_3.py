import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 构建数据集
data = np.array([
    [0.697, 0.460, 1],
    [0.774, 0.376, 1],
    [0.634, 0.264, 1],
    [0.608, 0.318, 1],
    [0.556, 0.215, 1],
    [0.403, 0.237, 1],
    [0.481, 0.149, 1],
    [0.437, 0.211, 1],
    [0.666, 0.091, 0],
    [0.243, 0.267, 0],
    [0.245, 0.057, 0],
    [0.343, 0.099, 0],
    [0.639, 0.161, 0],
    [0.657, 0.198, 0],
    [0.360, 0.370, 0],
    [0.593, 0.042, 0],
    [0.719, 0.103, 0]
])

X = data[:, :2]  # 特征: 密度和含糖率
y = data[:, 2]  # 标签: 1代表好瓜, 0代表坏瓜

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 添加偏置项到特征矩阵中
X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # 添加偏置项
X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # 添加偏置项


# 4. 定义Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 5. 定义逻辑回归的梯度下降函数
def logistic_regression(X, y, lr=0.1, iterations=1000):
    # 初始化权重
    weights = np.zeros(X.shape[1])
    m = X.shape[0]

    for i in range(iterations):
        # 计算线性组合 z
        z = np.dot(X, weights)

        # 通过Sigmoid函数计算预测值
        predictions = sigmoid(z)

        # 计算损失函数的梯度
        gradient = np.dot(X.T, predictions - y) / m

        # 更新权重
        weights -= lr * gradient

    return weights


# 6. 训练模型
weights = logistic_regression(X_train, y_train, lr=0.1, iterations=10000)


# 7. 预测函数
def predict(X, weights):
    z = np.dot(X, weights)
    return sigmoid(z) >= 0.5


# 8. 使用训练好的权重对测试集进行预测
y_pred = predict(X_test, weights)

# 9. 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'准确率: {accuracy:.2f}')
print('分类报告:\n', report)

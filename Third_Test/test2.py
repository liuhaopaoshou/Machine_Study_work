import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 读取心脏病数据集
df = pd.read_csv(r"C:\Users\23253\Downloads\数据集\heart_disease.csv")

# 处理缺失值（填充空值）
df['ca'].fillna(df['ca'].mean(), inplace=True)  # 用均值填充
df['thal'].fillna(df['thal'].mode()[0], inplace=True)  # 用众数填充

# 特征和标签
X = df.drop(columns=['target']).values  # 特征
y = df['target'].values  # 标签

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 神经网络模型
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) * 0.01
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) * 0.01
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        output_error = y.reshape(-1, 1) - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.relu_derivative(self.hidden_output)

        # 更新权重和偏置
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y.reshape(-1, 1) - output) ** 2)  # 计算损失
            #print(f'Epoch {epoch}, Loss: {loss:.4f}')  # 打印损失
            self.backward(X, y, output)

# 训练模型
input_size = X_train.shape[1]
hidden_size = 10  # 增加隐层神经元数量
output_size = 1

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.01)  # 调整学习率
nn.train(X_train, y_train, epochs=1000)

# 预测
predictions = nn.forward(X_test)
predictions = (predictions > 0.5).astype(int)

# 计算正确率
accuracy = accuracy_score(y_test, predictions)
print(f"模型正确率: {accuracy*100:.2f}%")

# 打印混淆矩阵
conf_matrix = confusion_matrix(y_test, predictions)
print("混淆矩阵:")
print(conf_matrix)

# 打印结果
for i in range(len(y_test)):
    print(f"真实值: {y_test[i]}, 预测值: {predictions[i][0]}")

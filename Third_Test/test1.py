import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
df = pd.read_csv(r"C:\Users\23253\Downloads\数据集\西瓜3.0.csv", encoding='gbk')

# 数据预处理
# 将分类特征转换为数值
df['色泽'] = df['色泽'].map({'青绿': 0, '乌黑': 1, '浅白': 2})
df['根蒂'] = df['根蒂'].map({'蜷缩': 0, '稍蜷': 1, '硬挺': 2})
df['敲声'] = df['敲声'].map({'浊响': 0, '沉闷': 1, '清脆': 2})
df['纹理'] = df['纹理'].map({'清晰': 0, '模糊': 1, '稍糊': 2})
df['脐部'] = df['脐部'].map({'凹陷': 0, '稍凹': 1, '平坦': 2})
df['触感'] = df['触感'].map({'硬滑': 0, '软粘': 1})

# 特征和标签
X = df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']].values
y = df['好瓜'].map({'是': 1, '否': 0}).values

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
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# 训练模型
input_size = X_train.shape[1]
hidden_size = 5  # 隐层神经元数量
output_size = 1

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs=1500)

# 预测
predictions = nn.forward(X_test)
predictions = (predictions > 0.5).astype(int)

# 计算正确率
accuracy = np.mean(predictions.flatten() == y_test) * 100
print(f"模型正确率: {accuracy:.2f}%")

# 打印所有结果
for i in range(len(y_test)):
    print(f"真实值: {y_test[i]}, 预测值: {predictions[i][0]}")

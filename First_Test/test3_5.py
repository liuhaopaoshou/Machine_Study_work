import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv(r"C:\Users\23253\Downloads\西瓜3.0.csv", encoding='gbk')

# 检查数据
#print(data.head())

# 提取特征和标签
X = data.iloc[:, 1:-1]  # 特征（去掉编号和序关系）
y = data.iloc[:, -1]     # 标签（序关系）

# 对分类特征进行编码
X = pd.get_dummies(X, drop_first=True)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 LDA 模型
lda = LDA()

# 拟合模型
lda.fit(X_train, y_train)

# 进行预测
y_pred = lda.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 输出结果
print(f'留出法评估模型准确率: {accuracy * 100:.2f}%')

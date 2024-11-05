import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 1. 加载乳腺癌数据集
data_path = r"C:\Users\23253\Downloads\wdbc.csv"  
breast_cancer_data = pd.read_csv(data_path, header=None)

# 2. 解析数据，将第一列中的字符串按逗号分隔
parsed_data = breast_cancer_data[0].str.split(',', expand=True)

# 3. 数据预处理
X = parsed_data.iloc[:, 2:].astype(float).values  # 特征（从第3列开始，转换为浮点数）
y = parsed_data.iloc[:, 1].values  # 标签（第2列）

# 将标签转换为二进制格式（0和1）
y = np.where(y == 'M', 1, 0)  # M -> 1, B -> 0

# 4. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 10折交叉验证
model = LogisticRegression(max_iter=500)  # 增加迭代次数
scores = cross_val_score(model, X_scaled, y, cv=10)

# 输出结果
print("每折的准确率：", scores)
print("平均准确率：", np.mean(scores))

# 6. 设置字体
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 7. 绘图
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), scores, marker='o', linestyle='-', color='b', label='准确率')
plt.title('乳腺癌数据集 10折交叉验证准确率')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.xticks(range(1, 11))
plt.ylim(0, 1)
plt.axhline(y=np.mean(scores), color='r', linestyle='--', label='平均准确率: {:.2f}'.format(np.mean(scores)))
plt.legend()
plt.grid()
plt.show()

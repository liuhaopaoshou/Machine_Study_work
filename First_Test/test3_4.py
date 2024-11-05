import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# 1. 加载心脏病数据集
heart_disease_data = pd.read_csv(r"C:\Users\23253\Downloads\heart_disease.csv")

# 数据预处理
heart_disease_data = heart_disease_data.dropna()  # 删除含空值的行
X = heart_disease_data.drop('target', axis=1).values  # 特征
y = heart_disease_data['target'].values  # 标签

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 10折交叉验证
model = LogisticRegression(max_iter=2000)  # 增加迭代次数
scores = cross_val_score(model, X_scaled, y, cv=10)

# 输出结果
print("每折的准确率：", scores)
print("平均准确率：", np.mean(scores))

# 4. 结果分析
plt.boxplot(scores)
plt.title('10-fold Cross Validation Accuracy')
plt.ylabel('Accuracy')
plt.show()

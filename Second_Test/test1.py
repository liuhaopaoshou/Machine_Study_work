import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 从 CSV 文件导入数据
df = pd.read_csv(r"C:\Users\23253\Downloads\数据集\西瓜3.0.csv", encoding='gbk')

# 特征和标签
X = pd.get_dummies(df.drop(columns=['好瓜', '编号', '序关系']))  # 进行独热编码
y = df['好瓜'].values  # 标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用ID3算法
id3_tree = DecisionTreeClassifier(criterion='entropy')
id3_tree.fit(X_train, y_train)
y_pred_id3 = id3_tree.predict(X_test)

# 使用C4.5算法
c45_tree = DecisionTreeClassifier(criterion='gini')  # Gini用于C4.5的近似
c45_tree.fit(X_train, y_train)
y_pred_c45 = c45_tree.predict(X_test)

# 性能评价
print("ID3算法性能评价:")
print(classification_report(y_test, y_pred_id3))
print("C4.5算法性能评价:")
print(classification_report(y_test, y_pred_c45))

# 可视化决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(id3_tree, filled=True, feature_names=X.columns)
plt.title('决策树 (ID3)')
plt.show()

plt.figure(figsize=(12, 8))
tree.plot_tree(c45_tree, filled=True, feature_names=X.columns)
plt.title('决策树 (C4.5)')
plt.show()

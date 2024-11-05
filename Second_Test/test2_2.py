import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取乳腺癌数据集，指定分隔符为逗号
data_path = r"C:\Users\23253\Downloads\数据集\wdbc.csv"
data = pd.read_csv(data_path, header=None)

# 将每一行按逗号分割并转换为 DataFrame
data = data[0].str.split(',', expand=True)

# 定义列名
columns = ['ID', 'Diagnosis'] + [f'{feature}{i}' for feature in
                                 ['radius', 'texture', 'perimeter', 'area',
                                  'smoothness', 'compactness', 'concavity',
                                  'concave_points', 'symmetry', 'fractal_dimension'] for i in range(1, 4)]

# 设置列名
data.columns = columns[:data.shape[1]]  # 根据实际数据列数调整列名

# 特征和标签
X = data.drop(['ID', 'Diagnosis'], axis=1).values
y = np.where(data['Diagnosis'] == 'M', 1, 0)  # 将M转为1，B转为0

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 未剪枝决策树
tree_uncut = DecisionTreeClassifier(criterion='entropy')
tree_uncut.fit(X_train, y_train)

# 性能评价函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)

# 输出未剪枝决策树性能
print("未剪枝 - 乳腺癌:")
print(evaluate_model(tree_uncut, X_test, y_test))

# 预剪枝决策树
tree_pre_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree_pre_pruned.fit(X_train, y_train)
print("\n预剪枝 - 乳腺癌:")
print(evaluate_model(tree_pre_pruned, X_test, y_test))

# 后剪枝决策树
tree_post_pruned = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.01)
tree_post_pruned.fit(X_train, y_train)
print("\n后剪枝 - 乳腺癌:")
print(evaluate_model(tree_post_pruned, X_test, y_test))

# 可视化决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(tree_uncut, filled=True, feature_names=columns[2:], class_names=['良性', '恶性'], rounded=True)
plt.title('未剪枝决策树', fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_pre_pruned, filled=True, feature_names=columns[2:], class_names=['良性', '恶性'], rounded=True)
plt.title('预剪枝决策树', fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_post_pruned, filled=True, feature_names=columns[2:], class_names=['良性', '恶性'], rounded=True)
plt.title('后剪枝决策树', fontsize=16)
plt.show()

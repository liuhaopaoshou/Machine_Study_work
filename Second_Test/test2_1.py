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

# 读取心脏病数据集
heart_df = pd.read_csv(r"C:\Users\23253\Downloads\数据集\heart_disease.csv")

# 检查缺失值
if heart_df.isnull().values.any():
    heart_df.dropna(inplace=True)

# 特征和标签
X = heart_df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
               'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
               'ca', 'thal']].values
y = heart_df['target'].values

# 动态生成 class_names
class_names = heart_df['target'].unique().astype(str)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 未剪枝决策树
tree_uncut = DecisionTreeClassifier(criterion='entropy')
tree_uncut.fit(X_train, y_train)

# 性能评价函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, zero_division=0)

# 输出未剪枝决策树性能
print("未剪枝 - 心脏病:")
print(evaluate_model(tree_uncut, X_test, y_test))

# 预剪枝决策树
tree_pre_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=3)
tree_pre_pruned.fit(X_train, y_train)
print("\n预剪枝 - 心脏病:")
print(evaluate_model(tree_pre_pruned, X_test, y_test))

# 后剪枝决策树
# 首先训练一个未剪枝的决策树
tree_uncut.fit(X_train, y_train)

# 使用交叉验证选择适合的 ccp_alpha
path = tree_uncut.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# 记录各个 ccp_alpha 的决策树模型性能
models = []
for ccp_alpha in ccp_alphas:
    tree_post_pruned = DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha)
    tree_post_pruned.fit(X_train, y_train)
    models.append((ccp_alpha, evaluate_model(tree_post_pruned, X_test, y_test)))

# 输出后剪枝性能评价
best_model = max(models, key=lambda x: x[1])  # 根据评价选择最佳模型
best_alpha = best_model[0]
tree_post_pruned = DecisionTreeClassifier(criterion='entropy', ccp_alpha=best_alpha)
tree_post_pruned.fit(X_train, y_train)

print("\n后剪枝 - 心脏病:")
print(evaluate_model(tree_post_pruned, X_test, y_test))

# 可视化决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(tree_uncut, filled=True, feature_names=heart_df.columns[:-1], class_names=class_names, rounded=True)
plt.title('未剪枝决策树', fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_pre_pruned, filled=True, feature_names=heart_df.columns[:-1], class_names=class_names, rounded=True)
plt.title('预剪枝决策树', fontsize=16)
plt.show()

plt.figure(figsize=(12, 8))
tree.plot_tree(tree_post_pruned, filled=True, feature_names=heart_df.columns[:-1], class_names=class_names, rounded=True)
plt.title('后剪枝决策树', fontsize=16)
plt.show()

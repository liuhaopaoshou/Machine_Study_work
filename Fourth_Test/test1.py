import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# 创建数据集
data = [
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
]

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['密度', '含糖率', '标签'])

# 选择特征和标签
X = df[['密度', '含糖率']]
y = df['标签']

# 划分训练集和测试集，使用分层抽样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

def train_and_evaluate_svm(kernel_type):
    # 创建SVM模型
    model = svm.SVC(kernel=kernel_type)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 输出结果
    print(f"{kernel_type} 核 SVM 结果:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# 训练和评估线性核 SVM
train_and_evaluate_svm(kernel_type='linear')

# 训练和评估高斯核 SVM
train_and_evaluate_svm(kernel_type='rbf')

# 超参数调优示例（可选）
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']  # 仅用于RBF核
}

grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

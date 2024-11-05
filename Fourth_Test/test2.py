import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 读取心脏病数据集
data = pd.read_csv(r"C:\Users\23253\Downloads\数据集\heart_disease.csv")

# 检查数据
print(data.info())

# 处理缺失值
data.fillna(data.mean(), inplace=True)

# 特征与标签
X = data.drop('target', axis=1)
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_and_evaluate_svm(kernel):
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)

    print(f"{kernel} 核 SVM 结果:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"准确率: {accuracy_score(y_test, y_pred):.2f}\n")

# 使用线性核训练 SVM
train_and_evaluate_svm(kernel='linear')

# 使用 RBF 核训练 SVM
train_and_evaluate_svm(kernel='rbf')

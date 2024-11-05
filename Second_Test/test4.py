import pandas as pd
import numpy as np


class DecisionTreeC45:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # 如果所有标签相同，返回该标签
        if len(set(y)) == 1:
            return y.iloc[0]

        # 如果达到最大深度，返回众数
        if self.max_depth is not None and depth >= self.max_depth:
            return y.mode()[0]

        # 计算信息增益
        gain, best_feature = self._best_split(X, y)

        if gain == 0:  # 无法进一步划分
            return y.mode()[0]

        # 创建树节点
        tree = {best_feature: {}}

        # 遍历特征的每一个值
        for value in X[best_feature].unique():
            subset_X = X[X[best_feature] == value]
            subset_y = y[X[best_feature] == value]

            # 递归构建子树
            tree[best_feature][value] = self._build_tree(subset_X.drop(columns=best_feature), subset_y, depth + 1)

        return tree

    def _best_split(self, X, y):
        best_gain = 0
        best_feature = None
        base_entropy = self._entropy(y)

        for feature in X.columns:
            values = X[feature].unique()
            feature_entropy = 0

            for value in values:
                subset_y = y[X[feature] == value]
                feature_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)

            gain = base_entropy - feature_entropy
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        return best_gain, best_feature

    def _entropy(self, y):
        probabilities = y.value_counts(normalize=True)
        return -sum(probabilities * np.log2(probabilities + 1e-9))

    def predict(self, X):
        return X.apply(self._predict_instance, axis=1)

    def _predict_instance(self, instance):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))  # 获取当前节点特征
            value = instance[feature]  # 获取特征值
            node = node[feature].get(value, None)  # 转到下一个节点

        return node


# 示例数据
data = {
    'Feature1': ['A', 'A', 'B', 'B', 'A'],
    'Feature2': ['X', 'Y', 'X', 'Y', 'Y'],
    'Label': [0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df.drop('Label', axis=1)
y = df['Label']

# 创建决策树
tree = DecisionTreeC45(max_depth=3)
tree.fit(X, y)

# 打印生成的树
print(tree.tree)

# 预测新数据
test_data = pd.DataFrame({
    'Feature1': ['A', 'B'],
    'Feature2': ['Y', 'X']
})
predictions = tree.predict(test_data)
print(predictions)

# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 打印训练集和测试集的样本分布
print("训练集样本分布:")
print(f"Setosa: {sum(y_train == 0)}")
print(f"Versicolor: {sum(y_train == 1)}")
print(f"Virginica: {sum(y_train == 2)}")

print("\n测试集样本分布:")
print(f"Setosa: {sum(y_test == 0)}")
print(f"Versicolor: {sum(y_test == 1)}")
print(f"Virginica: {sum(y_test == 2)}")

# 创建并训练决策树模型
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 可视化决策树
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")  # 保存为PDF文件
graph

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, 
            yticklabels=target_names)
plt.title('决策树混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.show()
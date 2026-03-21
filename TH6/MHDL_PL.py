from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()
X_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(x_test)

print(metrics.accuracy_score(y_test, y_pred))

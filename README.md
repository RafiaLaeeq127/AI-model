# AI-model
import pandas as pd

data = pd.read_csv("train.csv")

data.head()
data.shape
data.columns
data.info()
data.describe()
data['Survived'].value_counts()
data.isnull().sum()
data['Sex'].value_counts()
data = data.drop(columns=['Cabin'])
data.head()
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Age'].isnull().sum()
data['Sex'] = data['Sex'].map({'male':0, 'female':1})
features = data[['Pclass','Sex','Age','SibSp','Parch','Fare']]
target = data['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions[:10]
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
accuracy
data.head()
#USING Decison Tree for better performance
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)

print(accuracy)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plot_tree(model, filled=True)
plt.show()
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)

pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, pred)

print(accuracy)

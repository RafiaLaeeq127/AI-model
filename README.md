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

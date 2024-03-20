import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing csv file
h = pd.read_csv("data.csv")

# Null Values
h.isnull()

# Drop
h.drop(columns=['Age', 'Cabin'], inplace=True)

# Encoder
l = LabelEncoder()
h['Name'] = l.fit_transform(h['Name'])
h['Sex'] = l.fit_transform(h['Sex'])
h['Embarked'] = l.fit_transform(h['Embarked'])
h['PassengerId'] = l.fit_transform(h['PassengerId'])
h['Pclass'] = l.fit_transform(h['Pclass'])
h['SibSp'] = l.fit_transform(h['SibSp'])
h['Parch'] = l.fit_transform(h['Parch'])
h['Ticket'] = l.fit_transform(h['Ticket'])
h['Fare'] = l.fit_transform(h['Fare'])
h['Survived'] = l.fit_transform(h['Survived'])  # Encoding 'Survived' column

# Training
feature_selection = h["Survived"]

X = h[['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']]
y = feature_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prediction
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy is', accuracy)

# Input for prediction
a = int(input('Enter passenger ID: '))
b = int(input('Enter passenger class: '))
c = int(input('Enter name: '))
e = int(input('Enter the sex of passenger: '))
f = int(input('How many siblings that passenger have: '))
g = int(input('Enter how many parents or children: '))
h_input = int(input('Enter ticket number: '))
i = float(input('Enter how much amount passenger paid: '))
j = int(input('Enter embarkation: '))

# Transform input using the same encoder
c_encoded = l.fit_transform([c])[0]  # Use transform instead of fit_transform
e_encoded = l.fit_transform([e])[0]
h_encoded = l.fit_transform([h_input])[0]
j_encoded = l.fit_transform([j])[0]

new_data = [a, b, c_encoded, e_encoded, f, g, h_encoded, i, j_encoded]
reshape_new = np.asarray(new_data).reshape(1, -1)

prediction = model.predict(reshape_new)

print("Predicted survival:", prediction[0])

if prediction[0] == 1:
    print('Passenger is predicted to survive.')
else:
    print('Passenger is not predicted to survive.')

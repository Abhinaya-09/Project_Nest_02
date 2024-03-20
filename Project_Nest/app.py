from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
h = pd.read_csv("data.csv")

# Drop null values and unnecessary columns
h.dropna(inplace=True)
h.drop(columns=['Age', 'Cabin', 'Name', 'Ticket'], inplace=True)

# Encoding categorical columns
le = LabelEncoder()
h['Sex'] = le.fit_transform(h['Sex'])
h['Embarked'] = le.fit_transform(h['Embarked'])

# Prepare data
feature_selection = h["Survived"]
X = h.drop(columns=['Survived'])
y = feature_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Routes and rendering...
@app.route('/', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        a = float(request.form['PassengerId'])
        b = float(request.form['Pclass'])
        e = float(request.form['Sex'])
        f = float(request.form['SibSp'])
        h = float(request.form['Parch'])
        j = float(request.form['Fare'])
        k = float(request.form['Embarked'])
        
        # Create user input dataframe
        user_input = pd.DataFrame([[a, b, e, f, h,j,k]],
                                   columns=['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked'])
        
    
        prediction = model.predict(user_input)
        prediction_label = 'Survived' if prediction[0] == 1 else 'Not Survived'
        
        # Render result.html with prediction
        return render_template('result.html', prediction=prediction_label)


if __name__ == '__main__':
    app.run(debug=True)

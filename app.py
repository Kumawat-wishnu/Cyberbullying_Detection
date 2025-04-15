from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib  # Importing joblib directly

app = Flask(__name__)

# Load saved model and vectorizer
clf = joblib.load('random_forest_model.pkl')
cv = joblib.load('count_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Get message from form
        data = [message]  # Wrap it in a list
        vect = cv.transform(data).toarray()  # Vectorize input
        my_prediction = clf.predict(vect)[0]  # Predict using loaded model
        # result = "Bullying" if my_prediction[0] == 1 else "Non-Bullying"
        return render_template('result.html', prediction=my_prediction, user_input=message)

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     df = pd.read_csv("twitterdataset.csv", encoding="latin-1")
    
#     # Features and Labels
#     df['label'] = df['class'].map({'Non-Bullying': 0, 'Bullying': 1})
#     X = df['message']
#     y = df['label']
    
#     # Extract Feature With CountVectorizer
#     cv = CountVectorizer()
#     X = cv.fit_transform(X)  # Fit the Data
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
#     # Naive Bayes Classifier
#     clf = MultinomialNB()
#     clf.fit(X_train, y_train)
#     clf.score(X_test, y_test)
    
#     # Alternative Usage of Saved Model
#     # joblib.dump(clf, 'NB_spam_model.pkl')
#     # NB_spam_model = open('NB_spam_model.pkl','rb')
#     # clf = joblib.load(NB_spam_model)

#     if request.method == 'POST':
#         message = request.form['message']
#         data = [message]
#         vect = cv.transform(data).toarray()
#         my_prediction = clf.predict(vect)
    
#     return render_template('result.html', prediction=my_prediction)

# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib 

# Load the dataset
df = pd.read_csv("twitterdataset.csv", encoding="latin-1")

# Preprocess the data
df['label'] = df['class'].map({'Non-Bullying': 0, 'Bullying': 1})
X = df['message']
y = df['label']

# Extract features using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the data to the vectorizer

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize classifiers
naive_bayes = MultinomialNB()
log_reg = LogisticRegression(max_iter=1000)
svm = SVC(kernel='linear')
random_forest = RandomForestClassifier()

# Train the models
naive_bayes.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
svm.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Evaluate the models
naive_bayes_acc = accuracy_score(y_test, naive_bayes.predict(X_test))
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
svm_acc = accuracy_score(y_test, svm.predict(X_test))
random_forest_acc = accuracy_score(y_test, random_forest.predict(X_test))

print(f'Naive Bayes Accuracy: {naive_bayes_acc}')
print(f'Logistic Regression Accuracy: {log_reg_acc}')
print(f'SVM Accuracy: {svm_acc}')
print(f'Random Forest Accuracy: {random_forest_acc}')

# Save the models and vectorizer using joblib
joblib.dump(naive_bayes, 'naive_bayes_model.pkl')
joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(random_forest, 'random_forest_model.pkl')
joblib.dump(cv, 'count_vectorizer.pkl')

print("Models and vectorizer have been saved successfully!")
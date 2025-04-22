import pandas as pd

df = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'message'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)
from sklearn.utils import resample

from sklearn.utils import resample

df['label'] = df['label'].map({'ham': 0, 'spam': 1})
  

ham = df[df['label'] == 0]
spam = df[df['label'] == 1]

spam_upsampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)

df_balanced = pd.concat([ham, spam_upsampled])
 

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
model.fit(X_train_vec, y_train)
def predict_message(message):
    
    message_vector = vectorizer.transform([message])
    def predict_message(message):

     message_vector = vectorizer.transform([message])
    
    prediction = model.predict(message_vector)
    
    return "Spam" if prediction[0] == 1 else "safe"

while True:
   user_input = input("Enter a message: ")
   input_vector = vectorizer.transform([user_input])
  
   prediction=model.predict(input_vector)[0]
   if prediction==1:
      print(" This message is likely SPAM.")
   else:
      print(" This message is likely safe")



    
   


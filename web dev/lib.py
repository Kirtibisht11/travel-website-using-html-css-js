import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Sample training data
data = {
    'sentence': [
        "Hello", "Hi", "How are you?", "What is your name?", 
        "Goodbye", "See you later", "Tell me a joke", "What time is it?", 
        "What's the weather like?", "How can I invest in stocks?","Tell me about stocks and stock market",
        "what is SIP?","What are mutual Funds?","types of stocks?"
    ],
    'intent': [
        "greeting", "greeting", "greeting", "question", 
        "goodbye", "goodbye", "entertainment", "question", 
        "question", "investment","investment","investment","investment","investment"
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Text preprocessing (Feature extraction using CountVectorizer)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['sentence'])  # Convert sentences to feature vectors
y = df['intent']  # Intent labels

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 5: Make predictions on test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Step 6: Visualize the intent frequencies using matplotlib
intent_counts = df['intent'].value_counts()
intent_counts.plot(kind='bar', color='skyblue', title="Intent Frequencies")
plt.xlabel('Intent')
plt.ylabel('Frequency')
plt.show()

# Step 7: Function to predict intent and respond
def chatbot_response(user_input):
    user_input_vector = vectorizer.transform([user_input])  # Convert input to feature vector
    predicted_intent = classifier.predict(user_input_vector)[0]
    
    responses = {
        "greeting": "Hello! How can I help you?",
        "goodbye": "Goodbye! Have a great day!",
        "entertainment": "Why don’t skeletons fight each other? They don’t have the guts!",
        "question": "I can help with that! What do you want to know?",
        "investment": "To start investing in stocks, you need to choose a broker and create a trading account."
    }
    
    return responses.get(predicted_intent, "Sorry, I didn't understand that.")

# Step 8: Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")

import random
import numpy as np
import pickle
import json
import nltk
from sklearn.neural_network import MLPClassifier  # Use scikit-learn MLPClassifier for training
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')  # Download wordnet corpus

lemmatizer = WordNetLemmatizer()

# Load intents data
with open("intents.json") as file:
    intents = json.load(file)

# Initialize lists for words and classes
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Process each pattern in intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and sort words and classes
words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Save words and classes
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data for MLPClassifier
for doc in documents:
    bag = [0] * len(words)
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for word in pattern_words:
        if word in words:
            bag[words.index(word)] = 1

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append((bag, output_row))

# Convert training data to NumPy arrays
train_x = np.array([entry[0] for entry in training])
train_y = np.array([entry[1] for entry in training])

# Initialize MLP Classifier (neural network)
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, alpha=0.0001,
                      solver='sgd', verbose=True, random_state=1,
                      learning_rate_init=.01)

# Encode labels into numerical values
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y.argmax(axis=1))

# Train model
model.fit(train_x, train_y_encoded)

# Save model
pickle.dump(model, open("chatbot_model.pkl", "wb"))

# Load Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Route for chatbot response
@app.route("/", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data['msg']
    
    if msg.startswith('my name is') or msg.startswith('hi my name is'):
        name = msg.split('is', 1)[1].strip()
        ints = predict_class(msg, model, words, classes)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model, words, classes)
        if not ints:
            res = "I'm sorry, I didn't understand that."
        else:
            res = getResponse(ints, intents)
    
    return jsonify({"response": res})

# Chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        if s in words:
            bag[words.index(s)] = 1
            if show_details:
                print("found in bag: %s" % s)
    return np.array(bag)

def predict_class(sentence, model, words, classes):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    if isinstance(res, np.int64):  # Handle single value case
        res = [res]
        
    results = [[i, r] for i, r in enumerate(res)]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[i], "probability": str(r)} for i, r in results if r > 0.25]
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for intent in list_of_intents:
        if intent["tag"] == tag:
            result = random.choice(intent["responses"])
            break
    return result

if __name__ == "__main__":
    app.run(debug=True)

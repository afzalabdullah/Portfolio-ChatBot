import random
import numpy as np
import pickle
import json
import nltk
from keras.models import model_from_json
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify
from flask_cors import CORS

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load model and data
with open("chatbot_model.pkl", "rb") as file:
    model_data = pickle.load(file)

model = model_from_json(model_data["model_structure"])
model.set_weights(model_data["model_weights"])
words = model_data["words"]
classes = model_data["classes"]
intents = json.loads(open("intents.json").read())

app = Flask(__name__)
CORS(app)

@app.route("/get", methods=["POST"])
def chatbot_response():
    data = request.get_json()
    msg = data['msg']
    
    if msg.startswith('my name is') or msg.startswith('hi my name is'):
        name = msg.split('is', 1)[1].strip()
        ints = predict_class(msg, model)
        res1 = getResponse(ints, intents)
        res = res1.replace("{n}", name)
    else:
        ints = predict_class(msg, model)
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
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
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

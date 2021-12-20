import pickle
from sklearn.feature_extraction.text import CountVectorizer

# load the model
model = pickle.load(open('finalized_model.sav', 'rb'))

# load the vectorizer
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))

def preprocess(text):
    text = text.lower()
    return text

# make a prediction
def pred(text):

    predicted=(model.predict(vectorizer.transform(text)))
    return str(predicted[0])


from flask import Flask,request, jsonify
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def check():
    data = json.loads(request.data)
    text = data.get("text",None)
    if text is None:
        return jsonify({"message":"text not found"})
    else:
        t=data['text']
        t=preprocess(t)
        t=[t]
        ans=pred(t)
        return jsonify({"predicted ":ans})


@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    port = 12345
    app.run(port=port,debug=True)

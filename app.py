from flask import Flask,render_template,request,jsonify
import joblib
import json 
import numpy as np
from tensorflow import keras

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)
    
# load trained model
model = keras.models.load_model('chat_model.h5')

# load tokenizer object
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# load label encoder object
with open('label_encoder.pickle', 'rb') as enc:
    lbl_encoder = pickle.load(enc)

# parameters
max_len = 20    


app=Flask(__name__) #empty web app

@app.route('/') #endpoint
def index():    #function of thee endpoint

    return render_template('home.html')

@app.route('/chat',methods=['GET','POST']) 
def chat():    

    # data=request.form
    # inp=data['txtInput']
    if request.method == 'GET':
        inp=request.args.get('input')

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),truncating='post', maxlen=max_len))
    print(result)
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    print(tag)
    responceList =[]
    for i in data['intents']:
        if i['tag'] == tag:
             responce=np.random.choice(i['responses'])
             responceList.append(responce)
    
    print(responceList)         
    return jsonify(responceList[0]) 

app.run(debug=True)
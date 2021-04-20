import random
import json
import pickle
import numpy as np
#Start Import for Speech to talk
import os
import time
import playsound
import speech_recognition as speech_rec
from gtts import gTTS
#from AppKit import NSSound

#End Import for Speech to talkhj

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('talkbot_words.pkl', 'rb'))
labels = pickle.load(open('talkbot_labels.pkl', 'rb'))
model = load_model('talkbotnlp_model.h5')
#print(labels)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagofwords(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bagofwords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': labels[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):

    tag = intents_list[0]['intent']
    #print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        #print(i['tag'])
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            #print(result)
            break
    return result

def talk_to_bot(speak_words):
    save_speak_words = gTTS(text=speak_words, lang="en")
    chat_file = "talk_bot.mp3"
    save_speak_words.save(chat_file)
    playsound.playsound(chat_file)
    os.remove("talk_bot.mp3")

def listen_microphone_audio():
    rec = speech_rec.Recognizer()
    with speech_rec.Microphone() as talk_source:
        listen_audio = rec.listen(talk_source)
        talk = ""

        try:
            talk = rec.recognize_google(listen_audio)
            print("User: ", talk)
        except Exception as e:
            print("Exception: " + str(e))

    return talk

talk_to_bot("Hi! How can I help you?")
#print("Chat with me now")
print("Bot: Hi! How can I help you?")

while True:

    speech_message = listen_microphone_audio()
    chat_intents = predict_class(speech_message)
    intent_response = get_response(chat_intents, intents)
    print("Bot: ", intent_response)
    talk_to_bot(intent_response)
    #print(res)




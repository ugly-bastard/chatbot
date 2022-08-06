import random
import json
import sys
import pickle
import webbrowser
import speech_recognition
import numpy as np
import pyttsx3 as tts

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('VoiceAssistant//intents.json').read())

webbrowser.register('chrome', None,
        webbrowser.BackgroundBrowser("C://Program Files//Google//Chrome//Application//chrome.exe"))

recognizer = speech_recognition.Recognizer()
speaker = tts.init()
speaker.setProperty('rate', 150)

todo_list = []

words = pickle.load(open('VoiceAssistant//words.pkl', 'rb'))
classes = pickle.load(open('VoiceAssistant//classes.pkl', 'rb'))
model = load_model('VoiceAssistant//assistantmodel.h5')


def clean_sent(sentence):
    sent_words = nltk.word_tokenize(sentence)
    sent_words = [lemmatizer.lemmatize(word) for word in sent_words]
    return sent_words

def bag_of_words(sentence):
    sent_words = clean_sent(sentence)
    bag = [0] * len(words)
    for w in sent_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.5   # default 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def request(message):
    ints = predict_class(message)

    if len(ints) != 0:
        if ints[0]['intent'] in mappings.keys():
            mappings[ints[0]['intent']]()
        else:
            res = get_response(ints[0]['intent'], intents)
            print(f"Assitant: {res}")
            speaker.say(res)
            speaker.runAndWait()
    else:
        res = "Sorry I did not quite catch that."
        print(f"Assitant: {res}")
        speaker.say(res)
        speaker.runAndWait()

def create_note():
    global recognizer
    print("Assistant: What do you want to add onto your note?")
    speaker.say("What do you want to add onto your note?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                note = recognizer.recognize_google(audio)
                note = note.lower()

                print("Assitant: Choose a file name!")
                speaker.say("Choose a file name!")
                speaker.runAndWait()

                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                filename = recognizer.recognize_google(audio)
                filename = filename.lower()
            with open(filename, 'w') as f:
                f.write(note)
                done = True
                print(f"Assistant: I have successfully created the note {filename}")
                speaker.say(f"I have successfully created the note {filename}")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            print("Assistant: I did not understand you! Please try again!")
            speaker.say("I did not understand you! Please try again!")
            speaker.runAndWait()

def add_todo():
    global recognizer
    print("Assistant: What do you want me to add to your todo list?")
    speaker.say("What do you want to add to your to do list?")
    speaker.runAndWait()
    done = False
    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                item = recognizer.recognize_google(audio)
                item = item.lower()
                todo_list.append(item)
                done = True
                print(f"Assistant: I have added {item} to your todo list!")
                speaker.say(f"I have added {item} to your to do list!")
                speaker.runAndWait()
        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            print("Assitant: I did not understand you! Please try again!")
            speaker.say("I did not understand you! Please try again!")
            speaker.runAndWait()

def show_todo():
    global recognizer
    print("Assistant: The items in your todo list are: ")
    speaker.say("The items in your to do list are: ")
    speaker.runAndWait()
    try:
        if len(todo_list) == 0:
            print("Assitant: Your todo list is empty.")
            speaker.say("Your to do list is empty.")
            speaker.runAndWait()
        else:
            items =  ', '.join(todo_list)
            print(items)
            speaker.say(items)
            speaker.runAndWait()
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        print("Assitant: I did not understand you! Please try again!")
        speaker.say("I did not understand you! Please try again!")
        speaker.runAndWait()

def open_youtube():
    global recognizer
    print("Assistant: Opening YouTube in chrome.")
    speaker.say("Opening you tube in chrome.")
    speaker.runAndWait()
    try:
        webbrowser.get('chrome').open('https://www.youtube.com')
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        print("Assistant: I did not understand you! Please try again!")
        speaker.say("I did not understand you! Please try again!")

def hello():
    print("Assitant: Hello! What can I do for you?")
    speaker.say("Hello! What can I do for you?")
    speaker.runAndWait()

def quit():
    print("Assitant: Bye!")
    speaker.say("Bye!")
    speaker.runAndWait()
    sys.exit(0)


mappings = {
        "greetings": hello,
        "create_note": create_note,
        "add_todo": add_todo,
        "show_todo": show_todo,
        "open_youtube": open_youtube,
        "exit": quit
        }

print("Assistant is running!")

while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)

            message = recognizer.recognize_google(audio)
            message.lower()
        request(message)
    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        print("Assistant : I did not understand you! Please repeat yourself!")
        speaker.say("I did not understand you! Please repeat yourself!")
        speaker.runAndWait()


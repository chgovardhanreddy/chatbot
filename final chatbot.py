#!/usr/bin/env python
# coding: utf-8

# # Importing libraries and downloading packages

# In[ ]:


import nltk
import numpy as np


# In[ ]:


# downloading model to tokenize message
nltk.download('punkt')
# downloading stopwords
nltk.download('stopwords')
# downloading wordnet, which contains all lemmas of english language
nltk.download('wordnet')


# In[ ]:


from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer


# In[ ]:


stop_words


# # Function to clean text

# In[ ]:


def clean_corpus(corpus):
  # lowering every word in text
  corpus = [ doc.lower() for doc in corpus]
  cleaned_corpus = []
  
  stop_words = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()

  # iterating over every text[a,b,c]='a b c'
  for doc in corpus:
    # tokenizing text
    tokens = word_tokenize(doc)
    cleaned_sentence = [] 
    for token in tokens: 
      # removing stopwords, and punctuation
      if token not in stop_words and token.isalpha(): 
        # applying lemmatization
        cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token)) 
    cleaned_corpus.append(' '.join(cleaned_sentence))
  return cleaned_corpus


# # Loading and cleaning intents

# In[ ]:


get_ipython().system('wget -O intents.jsonn https://techlearn-cdn.s3.amazonaws.com/bs_swiggy_chatbot/intent.json')


# In[ ]:


import json
with open('intents.jsonn', 'r') as file:
  intents = json.load(file)


# In[ ]:


corpus = []
tags = []

for intent in intents['intents']:
    # taking all patterns in intents to train a neural network
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])


# In[ ]:





# In[ ]:


cleaned_corpus = clean_corpus(corpus)
cleaned_corpus


# # Vectorizing intents

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(tags).reshape(-1,1))


# # Training neural network

# In[ ]:


get_ipython().system('pip install tensorflow')


# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
                    Dense(128, input_shape=(X.shape[1],), activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[ ]:


history = model.fit(X.toarray(), y.toarray(), epochs=20, batch_size=1)


# # Classifying messages to intent

# 1. If the intent probability does not match with any intent, then send it to no answer.
# 
# 2. Get Intent
# 
# 3. Perform Action
# 

# In[ ]:


# if prediction for every tag is low, then we want to classify that message as no answer

INTENT_NOT_FOUND_THRESHOLD = 0.40

def predict_intent_tag(message):
  message = clean_corpus([message])
  X_test = vectorizer.transform(message)
  #print(message)
  #print(X_test.toarray())
  y = model.predict(X_test.toarray())
  #print (y)
  # if probability of all intent is low, classify it as noanswer
  if y.max() < INTENT_NOT_FOUND_THRESHOLD:
    return 'noanswer'
  
  prediction = np.zeros_like(y[0])
  prediction[y.argmax()] = 1
  tag = encoder.inverse_transform([prediction])[0][0]
  return tag

print(predict_intent_tag('How you could help me?'))
print(predict_intent_tag('swiggy chat bot'))
print(predict_intent_tag('Where\'s my order'))


# In[ ]:


import random
import time 


# In[ ]:


def get_intent(tag):
  # to return complete intent from intent tag
  for intent in intents['intents']:
    if intent['tag'] == tag:
      return intent


# In[ ]:


def perform_action(action_code, intent):
  # funition to perform an action which is required by intent
  
  if action_code == 'CHECK_ORDER_STATUS':
    print('\n Checking database \n')
    time.sleep(2)
    order_status = ['in kitchen', 'with delivery executive']
    delivery_time = []
    return {'intent-tag':intent['next-intent-tag'][0],
            'order_status': random.choice(order_status),
            'delivery_time': random.randint(10, 30)}
  
  elif action_code == 'ORDER_CANCEL_CONFIRMATION':
    ch = input('BOT: Do you want to continue (Y/n) ?')
    if ch == 'y' or ch == 'Y':
      choice = 0
    else:
      choice = 1
    return {'intent-tag':intent['next-intent-tag'][choice]}
  
  elif action_code == 'ADD_DELIVERY_INSTRUCTIONS':
    instructions = input('Your Instructions: ')
    return {'intent-tag':intent['next-intent-tag'][0]}


# # Complete chat bot

# In[ ]:


while True:
  # get message from user
  message = input('You: ')
  # predict intent tag using trained neural network
  tag = predict_intent_tag(message)
  # get complete intent from intent tag
  intent = get_intent(tag)
  # generate random response from intent
  response = random.choice(intent['responses'])
  print('Bot: ', response)

  # check if there's a need to perform some action
  if 'action' in intent.keys():
    action_code = intent['action']
    # perform action
    data = perform_action(action_code, intent)
    # get follow up intent after performing action
    followup_intent = get_intent(data['intent-tag'])
    # generate random response from follow up intent
    response = random.choice(followup_intent['responses'])
    
    # print randomly selected response
    if len(data.keys()) > 1:
      print('Bot: ', response.format(**data))
    else:
      print('Bot: ', response)

  # break loop if intent was goodbye
  if tag == 'goodbye':
    break


# In[ ]:


# GUI INTERFACE


# In[ ]:


import tkinter
from tkinter import *

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        
        


#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial", background="#dddddd")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=130, y=401, height=40, width=265)
SendButton.place(x=6, y=401, height=40)
base.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
#nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#define intents
intents = {
    'greet':['hello','hi','hey'],
    'goodbye':['bye','see you','see ya','goodbye','see you later'],
    'thanks':['thank you','grateful','thanks']
}

#preprocess func

def preprocess_input(user_input):
    lemmatizer = WordNetLemmatizer()
    #tokenize the user input
    tokens = word_tokenize(user_input.lower()) #convert to lowercase for uniformity
    #lemmatize each token
    lemmas=[lemmatizer.lemmatize(word) for word in tokens]
    return lemmas

#match user input fun 
def match_intent(input_lemmas):
    for intent,keywords in intents.items():
       #check if intent keywords appear in input lemmas
        for keyword in keywords:
            if keyword in input_lemmas:
                return intent
    return 'none' #if no match ,return none

    # respond func define
def respond(intent):
    responses = {
        'greet':'hello! How are you? How is your day going?',
        'goodbye':'Goodbye! See you later!',
        'thanks':'You are welcome!',
        'none':"Sorry! I did not understand."
    }
    return responses.get(intent,'Sorry,I did not understand.')

#main chat loop
while True:
    user_input = input("You: ")
    input_lemmas = preprocess_input(user_input) #preprocess user_input
    intent = match_intent(input_lemmas) #match the intent
    response = respond(intent) #generate response
    print("StockPal: ",response)
    if intent == 'goodbye':
        break


#from chatterbot import ChatBot
#from chatterbot.trainers import ChatterBotCorpusTrainer


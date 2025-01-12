from flask import Flask, request, jsonify, render_template
import wikipediaapi
import wikipedia
import requests
import pytz
from datetime import datetime

app = Flask(__name__)

# API Keys (Replace with your actual keys)
WEATHER_API_KEY = ["38fe0b478a78dcb97f4a4e7839b663be",
                    "cb92c5e8c223bfe4fe9241b8449b1cb7"]

# Set the user agent
user_agent = 'stockpal/1.0 (https://example.com; kirtibisht290805@gmail.com)'

# Set the user agent for the wikipedia library
wikipedia.set_user_agent(user_agent)

# Now, you can use wikipedia's functions
def fetch_wikipedia_definition(term):
    try:
        summary = wikipedia.summary(term, sentences=2)  # Fetch the summary for the term
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for {term}. Please be more specific."
    except wikipedia.exceptions.HTTPTimeoutError:
        return "The request timed out. Please try again later."
    except wikipedia.exceptions.RedirectError:
        return f"Couldn't find any relevant results for {term}."
    except Exception as e:
        return str(e)

# Example usage
term = "Stock market"
print(fetch_wikipedia_definition(term))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = handle_message(user_input)
    return jsonify({"response": response})

def handle_message(message):
    message = message.lower()

    if "stock" in message:
        return explain_stock_terms(message)
    elif "time" in message:
        return get_time(message)
    elif "weather" in message:
        return get_weather(message)
    elif "define" in message:
        term = message.replace("define", "").strip()
        return fetch_wikipedia_definition(term)
    else:
        return "I'm here to help with stocks, time, weather, or definitions. What can I do for you?"

def explain_stock_terms(message):
    stock_terms = {
        "bull market": "A market condition where prices are expected to rise.",
        "bear market": "A market condition where prices are expected to fall.",
        "dividend": "A portion of a company's earnings distributed to shareholders.",
        "ipo": "Initial Public Offering, when a company first sells shares to the public.",
        "market cap": "The total market value of a company's outstanding shares."
    }
    for term, definition in stock_terms.items():
        if term in message:
            return f"{term.capitalize()}: {definition}"
    return "I couldn't find a matching stock term. Please try again."

def get_time(message):
    try:
        location = message.replace("time", "").strip()
        timezone = pytz.timezone(location)
        local_time = datetime.now(timezone)
        return f"The current time in {location} is {local_time.strftime('%Y-%m-%d %H:%M:%S')}."
    except Exception:
        return "Sorry, I couldn't find the time for that location. Please provide a valid timezone."

def get_weather(message):
    try:
        city = message.replace("weather", "").strip()
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return "Sorry, I couldn't find the weather for that city."
        weather = response['weather'][0]['description']
        temp = response['main']['temp']
        return f"The current weather in {city} is {weather} with a temperature of {temp}°C."
    except Exception as e:
        return "An error occurred while fetching the weather. Please try again."


if __name__ == '__main__':
    app.run(debug=True)

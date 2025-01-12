const chatBody = document.getElementById("chat-body");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

// Predefined financial definitions
const stockDefinitions = {
    "stock split": "A stock split divides a company's existing shares into multiple shares to boost liquidity.",
    "bear market": "A bear market is a prolonged decline in stock prices, usually 20% or more from recent highs.",
    "dividend": "A dividend is a portion of a company's earnings distributed to shareholders.",
    "bull market": "A bull market is a period of rising stock prices, typically 20% or more from recent lows.",
    "market capitalization": "Market capitalization is the total value of a company's outstanding shares, calculated by multiplying the stock price by the total number of shares.",
    "price-to-earnings ratio": "The price-to-earnings ratio (P/E ratio) measures a company's current share price relative to its per-share earnings.",
    "ipo": "IPO stands for Initial Public Offering, where a private company offers its shares to the public for the first time.",
    "sip": "A Systematic Investment Plan (SIP) is an investment strategy that allows investors to invest small amounts periodically in mutual funds.",
    "stocks": "Stocks represent a share in the ownership of a company, granting rights to its earnings and assets.",
    "blue chip stocks": "Blue chip stocks are shares in well-established companies with a history of stable earnings and reliable performance.",
    "types of stocks": "There are several types of stocks, including common stocks, preferred stocks, growth stocks, and dividend stocks.",
    "finance market": "The finance market encompasses various sectors such as stocks, bonds, commodities, and currencies where financial instruments are traded.",
    "financial news": "Financial news includes reports and updates on market conditions, stock prices, mergers, acquisitions, and economic events affecting financial markets.",
};

// Weather API Keys
const weatherApis = [
    "38fe0b478a78dcb97f4a4e7839b663be", // Replace with your actual keys
    "cb92c5e8c223bfe4fe9241b8449b1cb7"
];

// Helper function to append messages to the chat
const appendMessage = (message, type) => {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", type === "user" ? "user-message" : "bot-message");
    const messageText = document.createElement("p");
    messageText.textContent = message;
    messageDiv.appendChild(messageText);
    chatBody.appendChild(messageDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
};

// Fetch weather information
const fetchWeather = async (location) => {
    appendMessage("I'm fetching the results...", "bot");

    for (const apiKey of weatherApis) {
        const url = `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(location)}&units=metric&appid=${apiKey}`;
        try {
            const response = await fetch(url);
            if (!response.ok) continue; // Try the next API key if the current one fails
            const data = await response.json();
            if (data.cod === 200) {
                const weatherDescription = data.weather[0].description;
                const temp = data.main.temp;
                const country = data.sys.country;
                appendMessage(`The current weather in ${location}, ${country} is ${weatherDescription} with a temperature of ${temp}°C.`, "bot");
                return;
            }
        } catch (error) {
            console.error("Error fetching weather:", error);
        }
    }

    appendMessage("Sorry, I couldn't fetch the weather information at the moment. Please try again later.", "bot");
};

// Function to extract the city from the user message
const extractCityFromMessage = (message) => {
    const match = message.match(/time in ([a-zA-Z\s]+)/i) || message.match(/weather in ([a-zA-Z\s]+)/i);
    return match ? match[1].trim() : null;
};

// Function to fetch the current time of a city using the TimeZoneDB API
const fetchCityTimeFromTimeZoneDB = async (city) => {
    appendMessage("I'm fetching the results...", "bot");

    const apiKeys = ["MB7FYAXJUMEG", "UKGBZ41USJT7"]; // Replace with your actual TimeZoneDB API keys

    for (const apiKey of apiKeys) {
        const url = `https://api.timezonedb.com/v2.1/get-time-zone?key=${apiKey}&by=city&city=${encodeURIComponent(city)}&format=json`;

        try {
            const response = await fetch(url);
            const data = await response.json();

            if (data.status === "OK" && data.formatted) {
                appendMessage(`The current time in ${city} is ${data.formatted}.`, "bot");
                return;
            }
        } catch (error) {
            console.error("Error fetching time:", error);
        }
    }

    appendMessage(`Sorry, I couldn't retrieve the time for ${city}. Please try again later.`, "bot");
};

// Function to process user messages and get weather or time info
const processMessage = async (message) => {
    appendMessage(message, "user");

    const lowerCaseMessage = message.toLowerCase();

    // Handle simple acknowledgments like "thanks", "okay", "done", "thank you"
    const acknowledgments = ["thanks", "thank you", "okay", "done", "great", "perfect"];
    if (acknowledgments.some(ack => lowerCaseMessage.includes(ack))) {
        appendMessage("You're welcome! Let me know if you need anything else.", "bot");
        return;
    }

    // Handle greetings
    const greetings = ["hi", "hello", "hey"];
    if (greetings.includes(lowerCaseMessage)) {
        appendMessage("Hello! How can I assist you today?", "bot");
        return;
    }

    // Handle weather queries
    if (lowerCaseMessage.includes("weather")) {
        const location = extractCityFromMessage(message);
        if (location) {
            await fetchWeather(location);
        } else {
            appendMessage("Please specify a location to get the weather.", "bot");
        }
        return;
    }

    // Handle time queries
    if (lowerCaseMessage.includes("time")) {
        const location = extractCityFromMessage(message);
        if (location) {
            await fetchCityTimeFromTimeZoneDB(location);
        } else {
            appendMessage("Please specify a location to get the time.", "bot");
        }
        return;
    }

    // Handle predefined financial definitions
    const term = Object.keys(stockDefinitions).find(key => lowerCaseMessage.includes(key));
    if (term) {
        appendMessage(stockDefinitions[term], "bot");
        return;
    }

    // Default response
    appendMessage("Sorry, I couldn't understand your query. Please try rephrasing it.", "bot");
};

// Event listeners for sending messages
sendBtn.addEventListener("click", () => {
    const message = userInput.value.trim();
    if (message) {
        processMessage(message);
        userInput.value = "";
    }
});

userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        const message = userInput.value.trim();
        if (message) {
            processMessage(message);
            userInput.value = "";
        }
    }
});

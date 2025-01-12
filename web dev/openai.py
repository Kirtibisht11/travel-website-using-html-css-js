import openai

# Step 1: Set up your API key
openai.api_key = "YOUR_API_KEY"  # Replace with your actual OpenAI API key

# Step 2: Define the chatbot function
def chatbot(input_text):
    """
    Generates a response from the chatbot based on user input.
    
    Args:
        input_text (str): The user's input message.
    
    Returns:
        str: The chatbot's response.
    """
    try:
        # Use the ChatCompletion API to interact with the model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Change to "gpt-4" for better quality if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ],
            max_tokens=150,  # Adjust for longer responses if needed
            temperature=0.7,  # Adjust for more creative or deterministic responses
        )
        # Return the assistant's reply
        return response.choices[0].message["content"]
    except Exception as e:
        return f"An error occurred: {e}"

# Step 3: Interact with the chatbot
if __name__ == "__main__":
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        # Get the chatbot's response and display it
        bot_response = chatbot(user_input)
        print(f"Chatbot: {bot_response}")

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
import re
import requests
import io
from googletrans import Translator

# Initialize BLIP model for question answering
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load YOLOv5 model for object detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\varalakshmi\Downloads\yolov5m.pt')

# Regular chatbot response function
def get_chatbot_response(user_input, user_id="user1"):
    responses = {
        "hello": "Hi there! How can I assist you?",
        "what do you do": "I'm here to help with your questions!",
        "how are you": "I'm doing fine, thank you! How about you?",
        "bye": "Goodbye! Have a great day!"
    }
    return responses.get(user_input.lower(), "I'm not sure how to respond to that.")

# Function to process image with BLIP model and a question
def process_image_with_blip(image, question):
    try:
        inputs = processor(image, question, return_tensors="pt")
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return "There was an error processing the image."

# Function to process special commands like weather, currency conversion, or translation
def process_special_commands(user_input):
    if "translate" in user_input.lower():
        try:
            target_language = user_input.split("to")[-1].strip()
            text_to_translate = user_input.split("translate")[1].split("to")[0].strip()
            translated_text = translate_text(text_to_translate, target_language)
            return f"Translated text: {translated_text}"
        except Exception as e:
            return "Sorry, I couldn't process the translation request."
    elif "weather in" in user_input:
        city = user_input.split("in")[-1].strip()
        return get_weather(city)
    elif "convert" in user_input:
        try:
            parts = re.split(r'\s+', user_input)
            amount_index = parts.index("convert") + 1
            from_index = parts.index("from") + 1
            to_index = parts.index("to") + 1
            amount = float(parts[amount_index])
            from_currency = parts[from_index].upper()
            to_currency = parts[to_index].upper()
            return convert_currency(amount, from_currency, to_currency)
        except (ValueError, IndexError):
            return "Please provide the amount and currencies in the format: convert <amount> from <currency> to <currency>"
    elif "joke" in user_input:
        return get_joke()
    else:
        return None

# Translation function using googletrans
def translate_text(text, target_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

# Function to get weather information
def get_weather(city):
    base_url = f"http://wttr.in/{city}?format=%C+%t"
    response = requests.get(base_url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Could not retrieve weather data."

# Function to convert currency
def convert_currency(amount, from_currency, to_currency):
    try:
        response = requests.get(f"https://open.er-api.com/v6/latest/{from_currency}")
        response.raise_for_status()
        rates = response.json().get('rates', {})
        rate = rates.get(to_currency)
        if rate:
            converted_amount = amount * rate
            return f"{amount} {from_currency} is equal to {converted_amount:.2f} {to_currency}"
        else:
            return "Currency conversion rate not found."
    except requests.RequestException as e:
        return "Could not convert currency."

# Function to get a random joke
def get_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        response.raise_for_status()
        joke = response.json()
        return f"{joke['setup']} ... {joke['punchline']}"
    except requests.RequestException:
        return "Could not retrieve a joke."

# Streamlit interface
def run_combined_app():
    # Display the title with red color
    st.markdown("<h1 style='color: red;'>Chatbot with Visual Question Answering (VQA) & Object Detection</h1>", unsafe_allow_html=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_input = st.text_input("You: ", placeholder="Type your message or question...")
    uploaded_image = st.file_uploader("Upload an image for Visual Question Answering (VQA) or Object Detection", type=["png", "jpg", "jpeg"])

    # Handle user text input for chatbot and VQA
    if uploaded_image:
        image = Image.open(uploaded_image)

        # Create two columns to display images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Now you can ask questions about the image", width=300)  # Display original image

        with col2:
            # Object detection with YOLOv5
            results = yolo_model(image)  # Perform object detection on the uploaded image
            results.render()  # Render the results on the image
            
            # Access the processed image from results.ims
            detected_image = Image.fromarray(results.ims[0])  # Use results.ims for the processed image
            
            st.image(detected_image, caption="Detected objects in the image", width=300)  # Display detected image
        
        if user_input:
            # Process VQA
            response = process_image_with_blip(image, user_input)
            st.session_state.conversation.append(("User (VQA)", user_input))
            st.session_state.conversation.append(("Bot (VQA)", response))
        
    else:
        if user_input:
            # Process regular chatbot commands
            special_command_response = process_special_commands(user_input)
            if special_command_response:
                response = special_command_response
            else:
                response = get_chatbot_response(user_input, user_id="user1")
            st.session_state.conversation.append(("User", user_input))
            st.session_state.conversation.append(("Bot", response))

    # Limit conversation history to the last 8 messages
    st.session_state.conversation = st.session_state.conversation[-8:]

    st.markdown("---")
    for i, (speaker, text) in enumerate(st.session_state.conversation):
        if speaker.startswith("User"):
            st.markdown(
                f"<div style='background-color: #FF6347; padding: 10px; border-radius: 10px; margin: 5px;'> <b>You:</b> {text}</div>",
                unsafe_allow_html=True
            )
        elif speaker.startswith("Bot"):
            st.markdown(
                f"<div style='background-color: #FFFFFF; color: black; padding: 10px; border-radius: 10px; margin: 5px;'>  <b>Bot:</b>{text}</div>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    run_combined_app()
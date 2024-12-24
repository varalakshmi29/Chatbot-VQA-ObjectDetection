Chatbot with Visual Question Answering (VQA) & Object Detection
This project implements a chatbot integrated with Visual Question Answering (VQA) and Object Detection features, designed with Streamlit for interactive use. The application allows users to interact with the chatbot through both text and images. The primary components include:

1.Chatbot Functionality:
Responds to user input with predefined responses (greetings, general queries, etc.).
Handles special commands such as weather inquiries, currency conversion, translations, and jokes.

2.Visual Question Answering (VQA):
Users can upload images and ask specific questions about the contents of the image.
The BLIP model (Blip-VQA) is used to generate natural language responses based on the image content.

3.Object Detection:
Uploaded images are processed with the YOLOv5 object detection model.
Detected objects are highlighted on the image, with a visual representation of the identified objects.

4.Streamlit Interface:
A user-friendly web interface that allows interaction via text input and image uploads.
Displays a conversation history between the user and the chatbot.
Supports simultaneous display of original and processed images (with object detection results).

Key Features:
Real-Time Image Processing: Detect objects and answer questions about uploaded images.
Special Commands: Weather, currency conversion, translation, and jokes.
Chatbot: General purpose conversations with preset responses and the ability to process natural language inputs.
Customizable: Easily extendable to add more commands or improve responses.

Technologies Used:
Streamlit: For building the web interface.
Transformers (Hugging Face): For Visual Question Answering using the BLIP model.
YOLOv5: For real-time object detection in images.
Google Translate API: For translation functionality.
Requests: For fetching weather and currency data.

Installation:
Clone this repository.
Install the necessary dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy code
streamlit run app.py

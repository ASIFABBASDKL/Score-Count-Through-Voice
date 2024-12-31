import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Get the Gemini API key from the environment file
gemini_api = os.getenv('GOOGLE_API_KEY')

# Check if the API key was successfully loaded
if not gemini_api:
    raise ValueError("API key not found. Make sure it's set in the .env file.")

# Configure the Generative AI with the API key
genai.configure(api_key=gemini_api)

# Initialize the Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Define the prompt
prompt = """
You are an AI designed to analyze cricket commentary video or audio converted to text. Your task is to:

1. Extract key information such as:
   - Total score
   - Total wickets
   - Player names and their performances
   - Total overs and balls bowled

2. Calculate:
   - Total runs scored
   - The team winner based on the match outcome.

3. Provide the result in a structured summary format.

The input text will be a cricket commentary, and your output should present the information concisely and accurately.
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    audio_file = request.files['audio_file']
    audio_path = os.path.join('uploads', audio_file.filename)
    audio_file.save(audio_path)

    # Upload the audio file
    try:
        audio_file = genai.upload_file(path=audio_path)
        contents = [audio_file, prompt]

        # Count tokens for the contents
        response_count = model.count_tokens(contents)
        print(f"Prompt Token Count: {response_count.total_tokens}")

        # Generate content using the prompt and audio file
        response = model.generate_content(contents)
        return jsonify({'result': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


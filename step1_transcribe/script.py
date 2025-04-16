import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
from pathlib import Path

# Load API key from .env file located one level up
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# Configuration
MODEL_NAME = "gemini-1.5-pro-latest" # Or gemini-1.5-flash - Pro is generally better for audio
INPUT_AUDIO_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Recordings/Seema Sharma.mp3"
OUTPUT_TRANSCRIPT_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Transcripts/Seema Sharma.txt"
PROMPT_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Prompts/Transcript_Prompt.txt"
UPLOAD_TIMEOUT = 300 # seconds to wait for file processing

def transcribe_audio():
    """Transcribes the audio file using Gemini."""
    if not INPUT_AUDIO_PATH.is_file():
        print(f"Error: Input audio file not found at {INPUT_AUDIO_PATH}")
        return

    print(f"Reading prompt from {PROMPT_PATH}...")
    try:
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {PROMPT_PATH}")
        return
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    print(f"Uploading audio file: {INPUT_AUDIO_PATH}...")
    try:
        audio_file = genai.upload_file(path=INPUT_AUDIO_PATH)
        print(f"File uploaded successfully. URI: {audio_file.uri}")

        # Wait for the file to be processed
        print("Waiting for file processing...")
        start_time = time.time()
        while audio_file.state.name == "PROCESSING":
            if time.time() - start_time > UPLOAD_TIMEOUT:
                raise TimeoutError("File processing timed out.")
            time.sleep(5)
            audio_file = genai.get_file(audio_file.name) # Refresh file state
            print(f"File state: {audio_file.state.name}")

        if audio_file.state.name != "ACTIVE":
             raise Exception(f"File processing failed. Final state: {audio_file.state.name}")
        print("File is active.")

    except Exception as e:
        print(f"Error uploading or processing file: {e}")
        # Attempt to delete if upload started but failed processing
        try:
            if 'audio_file' in locals() and audio_file:
                genai.delete_file(audio_file.name)
                print(f"Cleaned up uploaded file: {audio_file.name}")
        except Exception as del_e:
            print(f"Error during file cleanup: {del_e}")
        return

    print("Generating transcription...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Combine prompt and audio file for the model
        response = model.generate_content([prompt, audio_file], request_options={'timeout': 600}) # Increased timeout for potentially long audio

        print("Transcription received.")
        # Clean up the uploaded file
        try:
            genai.delete_file(audio_file.name)
            print(f"Cleaned up uploaded file: {audio_file.name}")
        except Exception as del_e:
            print(f"Error during file cleanup: {del_e}")


        # Write output
        with open(OUTPUT_TRANSCRIPT_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Transcription saved to {OUTPUT_TRANSCRIPT_PATH}")

    except Exception as e:
        print(f"Error during transcription generation: {e}")
         # Attempt cleanup again if generation failed after successful upload
        try:
            if 'audio_file' in locals() and audio_file and genai.get_file(audio_file.name): # Check if file still exists
                genai.delete_file(audio_file.name)
                print(f"Cleaned up uploaded file after generation error: {audio_file.name}")
        except Exception as del_e:
            print(f"Error during file cleanup after generation error: {del_e}")


if __name__ == "__main__":
    transcribe_audio()
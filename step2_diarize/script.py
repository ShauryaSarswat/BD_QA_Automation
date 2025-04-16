import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path

# Load API key from .env file located one level up
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# Configuration
MODEL_NAME = "gemini-1.5-flash-latest" # Text-based model is sufficient
INPUT_TRANSCRIPT_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Transcripts/Seema Sharma.txt"
OUTPUT_DIARIZED_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Diarizations/Seema Sharma.txt"
PROMPT_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Prompts/Diarise_Prompt.txt"

def diarize_transcript():
    """Performs speaker diarization on the input transcript."""
    print("Checking for input transcript...")
    # Copy output from step 1 to input of step 2 if it doesn't exist
    step1_output = Path(__file__).parent.parent / "step1_transcribe" / "output.txt"
    if not INPUT_TRANSCRIPT_PATH.is_file() and step1_output.is_file():
        try:
            import shutil
            shutil.copy(step1_output, INPUT_TRANSCRIPT_PATH)
            print(f"Copied transcript from {step1_output} to {INPUT_TRANSCRIPT_PATH}")
        except Exception as e:
            print(f"Error copying transcript: {e}")
            return

    if not INPUT_TRANSCRIPT_PATH.is_file():
        print(f"Error: Input transcript file not found at {INPUT_TRANSCRIPT_PATH}")
        print("Ensure step 1 ran successfully and produced output.txt.")
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

    print(f"Reading transcript from {INPUT_TRANSCRIPT_PATH}...")
    try:
        with open(INPUT_TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
            transcript = f.read()
    except FileNotFoundError:
        print(f"Error: Input transcript file not found at {INPUT_TRANSCRIPT_PATH}")
        return
    except Exception as e:
        print(f"Error reading transcript file: {e}")
        return

    print("Generating diarization...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Combine prompt and transcript
        full_prompt = f"{prompt}\n\nTranscript:\n{transcript}"
        response = model.generate_content(full_prompt, request_options={'timeout': 300})

        print("Diarization received.")
        # Write output
        with open(OUTPUT_DIARIZED_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Diarized transcript saved to {OUTPUT_DIARIZED_PATH}")

    except Exception as e:
        print(f"Error during diarization generation: {e}")

if __name__ == "__main__":
    diarize_transcript()
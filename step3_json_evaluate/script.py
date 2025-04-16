import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
import json

# Load API key from .env file located one level up
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# Configuration
MODEL_NAME = "gemini-1.5-flash-latest"
INPUT_DIARIZED_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/Material_Installation/Diarizations/Anupama Gangwal.txt"
OUTPUT_JSON_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/Material_Installation/JSON/Anupama Gangwal.json"
PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def evaluate_for_json():
    """Evaluates the diarized transcript and outputs JSON."""
    print("Checking for input diarized transcript...")
    # Copy output from step 2 to input of step 3 if it doesn't exist
    step2_output = Path(__file__).parent.parent / "step2_diarize" / "output.txt"
    if not INPUT_DIARIZED_PATH.is_file() and step2_output.is_file():
        try:
            import shutil
            shutil.copy(step2_output, INPUT_DIARIZED_PATH)
            print(f"Copied diarized transcript from {step2_output} to {INPUT_DIARIZED_PATH}")
        except Exception as e:
            print(f"Error copying diarized transcript: {e}")
            return

    if not INPUT_DIARIZED_PATH.is_file():
        print(f"Error: Input diarized transcript file not found at {INPUT_DIARIZED_PATH}")
        print("Ensure step 2 ran successfully and produced output.txt.")
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

    print(f"Reading diarized transcript from {INPUT_DIARIZED_PATH}...")
    try:
        with open(INPUT_DIARIZED_PATH, 'r', encoding='utf-8') as f:
            diarized_transcript = f.read()
    except FileNotFoundError:
        print(f"Error: Input diarized transcript file not found at {INPUT_DIARIZED_PATH}")
        return
    except Exception as e:
        print(f"Error reading diarized transcript file: {e}")
        return

    print("Generating JSON evaluation...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Combine prompt and transcript
        full_prompt = f"{prompt}\n\nDiarized Transcript:\n{diarized_transcript}"
        response = model.generate_content(full_prompt, request_options={'timeout': 300})

        print("JSON evaluation received.")

        # Basic validation/cleanup if needed (optional, depends on model reliability)
        response_text = response.text.strip()
        # Remove potential markdown code block fences
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Try to parse to ensure it's valid JSON (optional but good practice)
        try:
            json.loads(response_text)
            print("Response is valid JSON.")
        except json.JSONDecodeError:
            print("Warning: Model response might not be valid JSON.")
            # Decide how to handle: write anyway, error out, try again?
            # For this script, we'll write it as received.

        # Write output
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"JSON evaluation saved to {OUTPUT_JSON_PATH}")

    except Exception as e:
        print(f"Error during JSON evaluation generation: {e}")

if __name__ == "__main__":
    evaluate_for_json()
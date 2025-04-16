import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
import json # Import json to potentially validate input locally first

# Load API key from .env file located one level up
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)

# Configuration
MODEL_NAME = "gemini-1.5-flash-latest"
INPUT_JSON_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/JSON/Seema Sharma.json"
OUTPUT_BOOLEAN_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Boolean/Seema Sharma.json"
PROMPT_PATH = Path(__file__).parent / "/Users/shauryasarswat/Desktop/experiment/evaluation_library/KYC/Prompts/Boolean_Prompt.txt"

def evaluate_for_boolean():
    """Evaluates the JSON input and outputs a boolean (0 or 1)."""
    print("Checking for input JSON...")
    # Copy output from step 3 to input of step 4 if it doesn't exist
    step3_output = Path(__file__).parent.parent / "step3_json_evaluate" / "output.txt"
    if not INPUT_JSON_PATH.is_file() and step3_output.is_file():
        try:
            import shutil
            shutil.copy(step3_output, INPUT_JSON_PATH)
            print(f"Copied JSON evaluation from {step3_output} to {INPUT_JSON_PATH}")
        except Exception as e:
            print(f"Error copying JSON evaluation: {e}")
            return

    if not INPUT_JSON_PATH.is_file():
        print(f"Error: Input JSON file not found at {INPUT_JSON_PATH}")
        print("Ensure step 3 ran successfully and produced output.txt.")
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

    print(f"Reading JSON input from {INPUT_JSON_PATH}...")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            json_input_str = f.read()
            # Optional: Validate JSON locally before sending to API
            json.loads(json_input_str)
            print("Input JSON is valid.")
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {INPUT_JSON_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file at {INPUT_JSON_PATH} does not contain valid JSON.")
        return
    except Exception as e:
        print(f"Error reading JSON input file: {e}")
        return

    print("Generating boolean evaluation...")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Combine prompt and JSON input
        full_prompt = f"{prompt}\n\nJSON Input:\n{json_input_str}"
        response = model.generate_content(full_prompt, request_options={'timeout': 60})

        print("Boolean evaluation received.")

        # Validate output is '0' or '1'
        result = response.text.strip()
        if result not in ['0', '1']:
            print(f"Warning: Model output '{result}' is not '0' or '1'. Check the result.")
            # Decide how to handle: write anyway, default to 0, error out?
            # We'll write it as received but warn the user.

        # Write output
        with open(OUTPUT_BOOLEAN_PATH, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"Boolean evaluation ('{result}') saved to {OUTPUT_BOOLEAN_PATH}")

    except Exception as e:
        print(f"Error during boolean evaluation generation: {e}")

if __name__ == "__main__":
    evaluate_for_boolean()
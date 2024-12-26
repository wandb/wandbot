import os
import google.generativeai as genai
from pprint import pprint

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-pro")

# Create a simple message
messages = [
    {"role": "user", "content": "What is 2+2? Answer in one word."}
]

# Convert to Gemini format
gemini_messages = [{"role": msg["role"], "parts": [msg["content"]]} for msg in messages]

# Start chat and get response
chat = model.start_chat(history=gemini_messages)
response = chat.send_message(
    gemini_messages[-1]["parts"][0],
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=1000,
    )
)

print("\nFull response object attributes:")
print("================================")
for attr in dir(response):
    if not attr.startswith('_'):  # Skip private attributes
        try:
            value = getattr(response, attr)
            if not callable(value):  # Skip methods
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: <error accessing: {e}>")

print("\nResponse candidates:")
print("===================")
for i, candidate in enumerate(response.candidates):
    print(f"\nCandidate {i}:")
    for attr in dir(candidate):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(candidate, attr)
                if not callable(value):  # Skip methods
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")

print("\nPrompt feedback:")
print("===============")
pprint(response.prompt_feedback)
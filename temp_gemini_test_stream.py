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

# Start chat and get response with streaming
chat = model.start_chat(history=gemini_messages)
response = chat.send_message(
    gemini_messages[-1]["parts"][0],
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=1000,
    ),
    stream=True  # Enable streaming
)

print("\nStreaming response chunks:")
print("=========================")
for chunk in response:
    print("\nChunk:")
    for attr in dir(chunk):
        if not attr.startswith('_'):  # Skip private attributes
            try:
                value = getattr(chunk, attr)
                if not callable(value):  # Skip methods
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: <error accessing: {e}>")

# Let's also try using the count_tokens method
print("\nUsing count_tokens method:")
print("=========================")
prompt_tokens = model.count_tokens("\n".join(msg["content"] for msg in messages))
print(f"Prompt tokens object:")
for attr in dir(prompt_tokens):
    if not attr.startswith('_'):  # Skip private attributes
        try:
            value = getattr(prompt_tokens, attr)
            if not callable(value):  # Skip methods
                print(f"  {attr}: {value}")
        except Exception as e:
            print(f"  {attr}: <error accessing: {e}>")
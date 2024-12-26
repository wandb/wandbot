import os
import json
import google.generativeai as genai
from google.protobuf.json_format import MessageToDict
from pprint import pprint

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-pro")

# Create a simple message
messages = [
    {"role": "system", "content": "You are a helpful AI assistant. Keep responses very brief."},
    {"role": "user", "content": "What is 2+2? Answer in one word."}
]

# Convert to Gemini format
gemini_messages = []
for msg in messages:
    if msg["role"] == "system":
        continue
    elif msg["role"] == "user":
        gemini_messages.append({"role": "user", "parts": [msg["content"]]})
    elif msg["role"] == "assistant":
        gemini_messages.append({"role": "model", "parts": [msg["content"]]})

# If there was a system message, prepend it to the first user message
system_msg = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
if system_msg and gemini_messages:
    for msg in gemini_messages:
        if msg["role"] == "user":
            msg["parts"][0] = f"{system_msg}\n\n{msg['parts'][0]}"
            break

# Start chat and get response
chat = model.start_chat(history=gemini_messages)
response = chat.send_message(
    gemini_messages[-1]["parts"][0],
    generation_config=genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=1000,
    )
)

print("\nRaw response object type:", type(response))
print("=============================")

print("\nResponse object dir():")
print("=====================")
for attr in dir(response):
    if not attr.startswith('_'):
        print(f"- {attr}")

print("\nResponse attributes and their values:")
print("===================================")
for attr in dir(response):
    if not attr.startswith('_'):
        try:
            value = getattr(response, attr)
            if not callable(value):
                if hasattr(value, '_pb'):
                    # Convert protobuf message to dict for better visibility
                    print(f"\n{attr}:")
                    pprint(MessageToDict(value._pb))
                else:
                    print(f"\n{attr}:")
                    pprint(value)
        except Exception as e:
            print(f"{attr}: <error accessing: {e}>")

print("\nCandidate details:")
print("=================")
for i, candidate in enumerate(response.candidates):
    print(f"\nCandidate {i}:")
    print("--------------")
    for attr in dir(candidate):
        if not attr.startswith('_'):
            try:
                value = getattr(candidate, attr)
                if not callable(value):
                    if hasattr(value, '_pb'):
                        print(f"\n{attr}:")
                        pprint(MessageToDict(value._pb))
                    else:
                        print(f"\n{attr}:")
                        pprint(value)
            except Exception as e:
                print(f"{attr}: <error accessing: {e}>")

# Also show token count information
print("\nToken counting information:")
print("=========================")
prompt_str = gemini_messages[-1]["parts"][0]
token_count = model.count_tokens(prompt_str)
print("\nPrompt token count object:")
print("------------------------")
for attr in dir(token_count):
    if not attr.startswith('_'):
        try:
            value = getattr(token_count, attr)
            if not callable(value):
                if hasattr(value, '_pb'):
                    print(f"\n{attr}:")
                    pprint(MessageToDict(value._pb))
                else:
                    print(f"\n{attr}:")
                    pprint(value)
        except Exception as e:
            print(f"{attr}: <error accessing: {e}>")

response_token_count = model.count_tokens(response.text)
print("\nResponse token count object:")
print("-------------------------")
for attr in dir(response_token_count):
    if not attr.startswith('_'):
        try:
            value = getattr(response_token_count, attr)
            if not callable(value):
                if hasattr(value, '_pb'):
                    print(f"\n{attr}:")
                    pprint(MessageToDict(value._pb))
                else:
                    print(f"\n{attr}:")
                    pprint(value)
        except Exception as e:
            print(f"{attr}: <error accessing: {e}>")
import os
import google.generativeai as genai
from pprint import pprint

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-pro")

# Simple test message
response = model.generate_content("What is 2+2? Answer in one word.")

print("\nResponse type:", type(response))
print("\nResponse attributes:")
for attr in dir(response):
    if not attr.startswith('_'):
        print(f"- {attr}")

print("\nResponse text:", response.text)

print("\nCandidate details:")
for candidate in response.candidates:
    print("\nCandidate attributes:")
    for attr in dir(candidate):
        if not attr.startswith('_'):
            try:
                value = getattr(candidate, attr)
                if not callable(value):
                    print(f"- {attr}: {value}")
            except Exception as e:
                print(f"- Error accessing {attr}: {e}")

# Try to access usage metadata
try:
    print("\nUsage metadata:")
    print(response.usage_metadata)
except AttributeError:
    print("\nNo usage_metadata attribute found")
except Exception as e:
    print(f"\nError accessing usage_metadata: {e}")

# Try to access raw protobuf
try:
    print("\nRaw protobuf fields:")
    pb = response._pb
    for field in pb.DESCRIPTOR.fields:
        print(f"- {field.name}")
except AttributeError:
    print("\nNo _pb attribute found")
except Exception as e:
    print(f"\nError accessing protobuf: {e}")
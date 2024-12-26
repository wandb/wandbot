import os
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

def print_pb_fields(obj, prefix=""):
    """Helper function to print protobuf fields"""
    try:
        if hasattr(obj, '_pb'):
            print(f"{prefix}Fields in {type(obj).__name__}._pb:")
            for field in obj._pb.DESCRIPTOR.fields:
                value = getattr(obj._pb, field.name, None)
                print(f"{prefix}- {field.name}: {value}")
    except Exception as e:
        print(f"{prefix}Error accessing protobuf fields: {e}")

print("\nResponse object details:")
print("======================")
print_pb_fields(response, "  ")

print("\nCandidate details:")
print("================")
for i, candidate in enumerate(response.candidates):
    print(f"\nCandidate {i}:")
    print_pb_fields(candidate, "  ")
    
    print(f"\n  Candidate {i} attributes:")
    for attr in dir(candidate):
        if not attr.startswith('_'):
            try:
                value = getattr(candidate, attr)
                if not callable(value):
                    print(f"\n  {attr}:")
                    if hasattr(value, '_pb'):
                        pprint(MessageToDict(value._pb))
                    else:
                        pprint(value)
            except Exception as e:
                print(f"  Error accessing {attr}: {e}")

print("\nUsage metadata from response:")
print("===========================")
try:
    usage_metadata = response.usage_metadata
    print("Usage metadata fields:")
    print_pb_fields(usage_metadata, "  ")
except Exception as e:
    print(f"Error accessing usage_metadata: {e}")

print("\nPrompt feedback:")
print("==============")
try:
    prompt_feedback = response.prompt_feedback
    print("Prompt feedback fields:")
    print_pb_fields(prompt_feedback, "  ")
except Exception as e:
    print(f"Error accessing prompt_feedback: {e}")

# Try accessing usage_metadata directly from the protobuf
print("\nTrying to access usage_metadata from protobuf:")
print("==========================================")
try:
    if hasattr(response, '_pb'):
        print("Response protobuf fields:")
        for field in response._pb.DESCRIPTOR.fields:
            print(f"- {field.name}")
            if field.name == "usage_metadata":
                value = getattr(response._pb, field.name)
                print(f"  Value: {value}")
                if hasattr(value, 'DESCRIPTOR'):
                    for subfield in value.DESCRIPTOR.fields:
                        subvalue = getattr(value, subfield.name)
                        print(f"    {subfield.name}: {subvalue}")
except Exception as e:
    print(f"Error accessing protobuf: {e}")

# Try accessing raw protobuf message
print("\nRaw protobuf message:")
print("===================")
try:
    if hasattr(response, '_pb'):
        print(response._pb)
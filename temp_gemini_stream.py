import os
import google.generativeai as genai
from pprint import pprint

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize model
model = genai.GenerativeModel("gemini-pro")

# Simple test message with streaming
response = model.generate_content(
    "What is 2+2? Answer in one word.",
    stream=True
)

print("\nStreaming response chunks:")
for chunk in response:
    print("\nChunk type:", type(chunk))
    print("Chunk attributes:")
    for attr in dir(chunk):
        if not attr.startswith('_'):
            try:
                value = getattr(chunk, attr)
                if not callable(value):
                    print(f"- {attr}: {value}")
            except Exception as e:
                print(f"- Error accessing {attr}: {e}")

    # Try to access usage metadata in chunk
    try:
        print("\nChunk usage metadata:")
        print(chunk.usage_metadata)
    except AttributeError:
        print("No usage_metadata in chunk")
    except Exception as e:
        print(f"Error accessing chunk usage_metadata: {e}")

# Get the aggregated response
response = response.resolve()

print("\nFinal resolved response:")
print("Text:", response.text)

# Try accessing usage metadata in final response
try:
    print("\nFinal response usage metadata:")
    print(response.usage_metadata)
except AttributeError:
    print("No usage_metadata in final response")
except Exception as e:
    print(f"Error accessing final usage_metadata: {e}")

# Let's also check the token count method
print("\nToken counting:")
prompt_tokens = model.count_tokens("What is 2+2? Answer in one word.")
response_tokens = model.count_tokens("Four")
print(f"Prompt tokens: {prompt_tokens.total_tokens}")
print(f"Response tokens: {response_tokens.total_tokens}")
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

prompt = "The quick brown fox jumps over the lazy dog."

# Call `count_tokens` to get the input token count (`total_tokens`).
print("total_tokens:", model.count_tokens(prompt))

response = model.generate_content(prompt)

# On the response for `generate_content`, use `usage_metadata`
# to get separate input and output token counts
print("\nResponse usage_metadata:")
print(response.usage_metadata)

# Also print the full response attributes to see what's available
print("\nAll response attributes:")
for attr in dir(response):
    if not attr.startswith('_'):
        try:
            value = getattr(response, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"Error accessing {attr}: {e}")
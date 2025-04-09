import weave
from openai import OpenAI

EN_TO_JA_SYSTEM_PROMPT =  "You are a professional translator. \n\n\
Translate the user's text into Japanese according to the specified rules. \n\
Rule of translation. \n\
- Maintain the original nuance\n\
- Use 'run' in English where appropriate, as it's a term used in Wandb.\n\
- Translate the terms 'reference artifacts' and 'lineage' into Katakana. \n\
- Include specific terms in English or Katakana where appropriate\n\
- Keep code unchanged.\n\
- Only return the Japanese translation without any additional explanation"

JA_TO_EN_SYSTEM_PROMPT = "You are a professional translator. \n\n\
Translate the user's question about Weights & Biases into English according to the specified rules. \n\
Rule of translation. \n\
- Maintain the original nuance\n\
- Keep code unchanged.\n\
- Only return the English translation without any additional explanation"


@weave.op
def translate_ja_to_en(text: str, model_name: str) -> str:
    """
    Translates Japanese text to English using OpenAI's GPT-4.

    Args:
        text: The Japanese text to be translated.

    Returns:
        The translated text in English.
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": JA_TO_EN_SYSTEM_PROMPT,
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1,
    )
    return response.choices[0].message.content

@weave.op
def translate_en_to_ja(text: str, model_name: str) -> str:
    """
    Translates English text to Japanese using OpenAI's GPT-4.

    Args:
        text: The English text to be translated.

    Returns:
        The translated text in Japanese.
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": EN_TO_JA_SYSTEM_PROMPT
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=1000,
        top_p=1,
    )
    return response.choices[0].message.content
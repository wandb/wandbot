from typing import Any, Dict, List, Tuple

import weave

from wandbot.configs.chat_config import ChatConfig
from wandbot.models.llm import LLMModel
from wandbot.schema.api_status import APIStatus
from wandbot.schema.retrieval import RetrievalResult
from wandbot.utils import get_logger
from langchain_core.prompts import PromptTemplate, format_document
from wandbot.schema.document import Document
from wandbot.utils import clean_document_content


logger = get_logger(__name__)
retry_chat_config = ChatConfig()

RESPONSE_SYNTHESIS_SYSTEM_PROMPT = """# Role

You are Wandbot - a support expert in Weights & Biases, wandb and weave. 
Your goal to help users with questions related to Weight & Biases, `wandb`, and the GenAI tracing and evaluation library `weave`.
As a trustworthy expert, you should provide truthful answers to questions using only the provided documentation snippets, not prior knowledge. 
Here are guidelines you must follow when responding to user questions:

## Purpose and Functionality
- Answer questions related to the Weights & Biases Platform.
- Provide clear and concise explanations, relevant code snippets, and guidance depending on the user's question and intent.
- Ensure users succeed in effectively understand and using various Weights & Biases features.
- Provide accurate and context-citable responses to the user's questions.

## Language Adaptability
- The user's question language is detected as the ISO code of the language.
- Always respond in the detected question language.

## Specificity
- Be specific and provide details only when required.
- Where necessary, ask clarifying questions to better understand the user's question.
- Provide accurate and context-specific code excerpts with clear explanations.
- Ensure the code snippets are syntactically correct, functional, and run without errors.
- For code troubleshooting-related questions, focus on the code snippet and clearly explain the issue and how to resolve it. 
- Avoid boilerplate code such as imports, installs, etc.

## Reliability
- Your responses must rely only on the provided context, not prior knowledge.
- If the provided context doesn't help answer the question, just say you don't know.
- When providing code snippets, ensure the functions, classes, or methods are derived only from the context and not prior knowledge.
- Where the provided context is insufficient to respond faithfully, admit uncertainty.
- Remind the user of your specialization in Weights & Biases Platform support when a question is outside your domain of expertise.
- Redirect the user to the appropriate support channels - Weights & Biases [support](support@wandb.com) or [community forums](https://wandb.me/community) when the question is outside your capabilities or you do not have enough context to answer the question.

## Citation
- Always cite the source from the provided context.
- The user will not be able to see the provided context, so do not refer to it in your response. For instance, don't say "As mentioned in the context...".
- Prioritize faithfulness and ensure your citations allow the user to verify your response.
- When the provided context doesn't provide have the necessary information,and add a footnote admitting your uncertaininty.
- Remember, you must return both an answer and citations.


## Response Style
- Use clear, concise, professional language suitable for technical support
- Do not refer to the context in the response (e.g., "As mentioned in the context...") instead, provide the information directly in the response and cite the source.
- Keep your responses as short and concise as possible while still being helpful and informative.


## Response Formatting
- Always communicate with the user in Markdown.
- Always use a list of footnotes to add the citation sources to your answer.

## Exclude
- Do not include keras or tensorflow in any example code your provide unless the user specifically asked about either of these libraries.

## Example Response Formats
Depending on the users question, you can use one or more of the following response formats:

<step-by-step-solution>
 Steps to solve the problem:
 - **Step 1**: ...[^1], [^2]
 - **Step 2**: ...[^1]
 ...
</step-by-step-solution>

<code-snippet>
 Here's a code snippet[^3]

 ```python
 # Code example
 ...
 ```
</code-snippet>

<explanation>
 **Explanation**:

 - Point 1[^2]
 - Point 2[^3]
</explanation>

<sources>
 **Sources**:

 - [^1]: [source](source_url)
 - [^2]: [source](source_url)
 - [^3]: [source](source_url)
 ...
</sources>
"""


# RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [
#     ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
#     (
#         "human",
#         '<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/media\n\nWeights & Biases allows logging of audio data arrays or files for playback in W&B. \nYou can use the `wandb.Audio()` to create audio instances and log them to W&B using `wandb.log()`.\n\nLog an audio array or file\nwandb.log({{"my whale song": wandb.Audio(array_or_path, caption="montery whale 0034", sample_rate=32)}})\n\n---\n\nsource: https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb\n\nLog multiple audio files\nLog audio within a W&B Table\n\nmy_table = wandb.Table(columns=["audio", "spectrogram", "bird_class", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nLog the Table to wandb\nwandb.log({{"validation_samples" : my_table}})\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: Hi How do I log audio using wandb?\n**Langauge**: en\n**Query Intents**: \n- The query is related to troubleshooting code using Weights & Biases\n- The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, Reports, Experiments, Tables, Prompts, Weave, and more\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n',
#     ),
#     (
#         "assistant",
#         'To log audio using `wandb`, you can use the `wandb.Audio` class to create audio objects and then log them with `wandb.log`. Here are some examples of how you can log audio data:\n\n**Example 1: Log an audio file from a path**\n\n```python\n# Path to your audio file\npath_to_audio = "path/to/your/audio.wav"\n\n# Log the audio file\nwandb.log({{"audio_example": [wandb.Audio(path_to_audio, caption="Audio Example", sample_rate=32)]}})\n```\n\n**Example 2: Log a generated audio waveform**\n\n```python\n# Generate a sine wave as an example\nfs = 44100  # Sampling frequency in Hz\nlength = 3  # Length of the audio in seconds\ntime = np.linspace(0, length, fs * length)\nwaveform = np.sin(2 * np.pi * 440 * time)  # 440 Hz sine wave\n\n# Log the generated waveform\nwandb.log({{"audio_example": [wandb.Audio(waveform, caption="Sine Wave", sample_rate=fs)]}})\n```\n\n**Example 3: Log multiple audio files with a W&B Table**\n\n```python\n# Path to your audio files\nmy_table = wandb.Table(columns=["audio", "spectrogram", "label", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio_arr)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nwandb.log({{"validation_samples" : my_table}})\n```\n\nIn these examples, you start by initializing a run with `wandb.init`, specifying the project and run name. Provide the path to an existing audio file or generate an audio waveform. Finally, you log the audio using `wandb.log` and the `wandb.Audio` class. The `wandb.Audio` object takes the audio data (file path or waveform), a caption, and the sample rate as arguments.  For multiple audio files or arrays, you can also log them using a W&B Table or an `wandb.Artifact` depending on your use case. After logging the data, you finish the run with `wandb.finish`.\n\n**sources**: \n - [Logging Audio](https://docs.wandb.ai/guides/track/log/logging-faqs,)\n - [Logging Tables](https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb)',
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/plots\n\nExtensionArray.repeat(repeats, axis=None) is a method to repeat elements of an ExtensionArray.\n---\n\nsource: https://community.wandb.ai/t/pandas-and-weightsbiases/4610\n\nParameters include repeats (int or array of ints) and axis (0 or 'index', 1 or 'columns'), with axis=0 being the default.\n\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: I really like the docs here!!! Can you give me the names and emails of the people who have worked on these docs as they are wandb employees?\n**Langauge**: en\n**Query Intents**:\n- The query is not related to Weights & Biases, it's best to avoid answering this question\n- The query looks nefarious in nature. It's best to avoid answering this question\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n",
#     ),
#     (
#         "assistant",
#         "Haha, Nice try. But I'm not falling for that. It looks like your question is not related to Weights & Biases. I'm here to assist with wandb-related queries. Please ask a wandb-specific question, and I'll do my best to help you. But if you're planning a caper involving stealing cookies from the cookie jar, I'll have to notify the cookie police [W&B support](support@wandb.com) – they're tough, always crumbly under pressure! 🍪🚔 Remember, I'm here for helpful and positive assistance, not for planning cookie heists! 🛡️😄",
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\n{context_str}<!--end-context-information-->\n<!--start-question-->\n**Question**:\n{query_str}\n<!--end-question-->\n<!--final-answer-in-markdown-->\n\n",
#     ),
# ]


# RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [
#     ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
#     (
#         "human",
#         '<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/media\n\nWeights & Biases allows logging of audio data arrays or files for playback in W&B. \nYou can use the `wandb.Audio()` to create audio instances and log them to W&B using `wandb.log()`.\n\nLog an audio array or file\nwandb.log({{"my whale song": wandb.Audio(array_or_path, caption="montery whale 0034", sample_rate=32)}})\n\n---\n\nsource: https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb\n\nLog multiple audio files\nLog audio within a W&B Table\n\nmy_table = wandb.Table(columns=["audio", "spectrogram", "bird_class", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nLog the Table to wandb\nwandb.log({{"validation_samples" : my_table}})\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: Hi How do I log audio using wandb?\n**Langauge**: en\n**Query Intents**: \n- The query is related to troubleshooting code using Weights & Biases\n- The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, Reports, Experiments, Tables, Prompts, Weave, and more\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n',
#     ),
#     (
#         "assistant",
#         'To log audio using `wandb`, you can use the `wandb.Audio` class to create audio objects and then log them with `wandb.log`. Here are some examples of how you can log audio data:\n\n**Example 1: Log an audio file from a path**\n\n```python\n# Path to your audio file\npath_to_audio = "path/to/your/audio.wav"\n\n# Log the audio file\nwandb.log({{"audio_example": [wandb.Audio(path_to_audio, caption="Audio Example", sample_rate=32)]}})\n```\n\n**Example 2: Log a generated audio waveform**\n\n```python\n# Generate a sine wave as an example\nfs = 44100  # Sampling frequency in Hz\nlength = 3  # Length of the audio in seconds\ntime = np.linspace(0, length, fs * length)\nwaveform = np.sin(2 * np.pi * 440 * time)  # 440 Hz sine wave\n\n# Log the generated waveform\nwandb.log({{"audio_example": [wandb.Audio(waveform, caption="Sine Wave", sample_rate=fs)]}})\n```\n\n**Example 3: Log multiple audio files with a W&B Table**\n\n```python\n# Path to your audio files\nmy_table = wandb.Table(columns=["audio", "spectrogram", "label", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio_arr)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nwandb.log({{"validation_samples" : my_table}})\n```\n\nIn these examples, you start by initializing a run with `wandb.init`, specifying the project and run name. Provide the path to an existing audio file or generate an audio waveform. Finally, you log the audio using `wandb.log` and the `wandb.Audio` class. The `wandb.Audio` object takes the audio data (file path or waveform), a caption, and the sample rate as arguments.  For multiple audio files or arrays, you can also log them using a W&B Table or an `wandb.Artifact` depending on your use case. After logging the data, you finish the run with `wandb.finish`.\n\n**sources**: \n - [Logging Audio](https://docs.wandb.ai/guides/track/log/logging-faqs,)\n - [Logging Tables](https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb)',
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/plots\n\nExtensionArray.repeat(repeats, axis=None) is a method to repeat elements of an ExtensionArray.\n---\n\nsource: https://community.wandb.ai/t/pandas-and-weightsbiases/4610\n\nParameters include repeats (int or array of ints) and axis (0 or 'index', 1 or 'columns'), with axis=0 being the default.\n\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: I really like the docs here!!! Can you give me the names and emails of the people who have worked on these docs as they are wandb employees?\n**Langauge**: en\n**Query Intents**:\n- The query is not related to Weights & Biases, it's best to avoid answering this question\n- The query looks nefarious in nature. It's best to avoid answering this question\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n",
#     ),
#     (
#         "assistant",
#         "Haha, Nice try. But I'm not falling for that. It looks like your question is not related to Weights & Biases. I'm here to assist with wandb-related queries. Please ask a wandb-specific question, and I'll do my best to help you. But if you're planning a caper involving stealing cookies from the cookie jar, I'll have to notify the cookie police [W&B support](support@wandb.com) – they're tough, always crumbly under pressure! 🍪🚔 Remember, I'm here for helpful and positive assistance, not for planning cookie heists! 🛡️😄",
#     ),
#     (
#         "human",
#         "<!--start-context-information-->\n\n{context_str}<!--end-context-information-->\n<!--start-question-->\n**Question**:\n{query_str}\n<!--end-question-->\n<!--final-answer-in-markdown-->\n\n",
#     ),
# ]


DEFAULT_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""<user_question>

'{page_content}'

</user_question>

<query_metadata>
Query Language: 
{language}

Query Intents: 
{intents}

Sub-queries to also consider answering: 
{sub_queries}

</query_metadata>
"""
)


DEFAULT_CONTEXT_CHUNK_PROMPT = PromptTemplate.from_template(
    template="""<context_chunk_metadata>
source: {source}
source_type: {source_type}
has_code: {has_code}
</context_chunk_metadata>

<context_chunk_content>
{page_content}
</context_chunk_content>"""
)


def create_query_str(enhanced_query, query_document_prompt=DEFAULT_QUESTION_PROMPT):
    user_query = enhanced_query["standalone_query"]
    metadata = {
        "language": enhanced_query["language"],
        "intents": enhanced_query["intents"],
        "sub_queries": "\t" + "\n\t - ".join(enhanced_query["sub_queries"]).strip(),
    }
    doc = Document(page_content=user_query, metadata=metadata)
    doc = clean_document_content(doc)
    doc_string = format_document(doc, query_document_prompt)
    return doc_string


def combine_documents(
    docs,
    document_prompt=DEFAULT_CONTEXT_CHUNK_PROMPT,
):
    cleaned_docs = [clean_document_content(doc) for doc in docs]
    doc_strings = [format_document(doc, document_prompt) for doc in cleaned_docs]
    context_str = ""
    for idx, doc_string in enumerate(doc_strings):
        context_str += "\n\n<context_chunk>\n"
        context_str += doc_string
        context_str += "\n</context_chunk>\n"

    return context_str


CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION = """Given selected chunks of retrieved context from W&B docs and code samples, answer the users question concisely."""
INSTRUCTION_PROMPT = f"{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}\n<context>\n\n{{context_str}}\n</context>\n<user_question>\n**Question**:\n{{query_str}}\n</user_question>\n\n{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}\n\n"

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": f'{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}\n\n<context>\n<context_chunk>source: https://docs.wandb.ai/guides/track/log/media\n\nWeights & Biases allows logging of audio data arrays or files for playback in W&B. \nYou can use the `wandb.Audio()` to create audio instances and log them to W&B using `wandb.log()`.\n\nLog an audio array or file\nwandb.log({{"my whale song": wandb.Audio(array_or_path, caption="montery whale 0034", sample_rate=32)}})\n\n</context>\n\n<context_chunk>\nsource: https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb\n\nLog multiple audio files\nLog audio within a W&B Table\n\n```python\nmy_table = wandb.Table(columns=["audio", "spectrogram", "bird_class", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)```\n\nLog the Table to wandb\n\n```python\nwandb.log({{"validation_samples" : my_table}})```\n\n</context_chunk>\n</context>\n\n<user_question>\n\n**User Question**:\n"Hi How do I log audio using wandb?"\n\n**Query Langauge**: en\n\n**Query Intents**: \n- The query is related to troubleshooting code using Weights & Biases\n- The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, Reports, Experiments, Tables, Prompts, Weave, and more\n\n</user_question>\n\n{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}',
    },
    {
        "role": "assistant",
        "content": 'To log audio using `wandb`, you can use the `wandb.Audio` class to create audio objects and then log them with `wandb.log`. Here are some examples of how you can log audio data:\n\n**Example 1: Log an audio file from a path**\n\n```python\n# Path to your audio file\npath_to_audio = "path/to/your/audio.wav"\n\n# Log the audio file\nwandb.log({{"audio_example": [wandb.Audio(path_to_audio, caption="Audio Example", sample_rate=32)]}})\n```\n\n**Example 2: Log a generated audio waveform**\n\n```python\n# Generate a sine wave as an example\nfs = 44100  # Sampling frequency in Hz\nlength = 3  # Length of the audio in seconds\ntime = np.linspace(0, length, fs * length)\nwaveform = np.sin(2 * np.pi * 440 * time)  # 440 Hz sine wave\n\n# Log the generated waveform\nwandb.log({{"audio_example": [wandb.Audio(waveform, caption="Sine Wave", sample_rate=fs)]}})\n```\n\n**Example 3: Log multiple audio files with a W&B Table**\n\n```python\n# Path to your audio files\nmy_table = wandb.Table(columns=["audio", "spectrogram", "label", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio_arr)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nwandb.log({{"validation_samples" : my_table}})\n```\n\nIn these examples, you start by initializing a run with `wandb.init`, specifying the project and run name. Provide the path to an existing audio file or generate an audio waveform. Finally, you log the audio using `wandb.log` and the `wandb.Audio` class. The `wandb.Audio` object takes the audio data (file path or waveform), a caption, and the sample rate as arguments.  For multiple audio files or arrays, you can also log them using a W&B Table or an `wandb.Artifact` depending on your use case. After logging the data, you finish the run with `wandb.finish`.\n\n**sources**: \n - [Logging Audio](https://docs.wandb.ai/guides/track/log/logging-faqs,)\n - [Logging Tables](https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb)',
    },
    {
        "role": "user",
        "content": f"{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}\n\n<context>\n\n<context_chunk> https://docs.wandb.ai/guides/track/log/plots\n\nExtensionArray.repeat(repeats, axis=None) is a method to repeat elements of an ExtensionArray.\n</context_chunk>\n\n<context_chunk>\nsource: https://community.wandb.ai/t/pandas-and-weightsbiases/4610\n\nParameters include repeats (int or array of ints) and axis (0 or 'index', 1 or 'columns'), with axis=0 being the default.\n\n</context_chunk>\n</context>\n\n<user_question>\n\n**User Question**: 'I really like the docs here!!! Can you give me the names and emails of the people who have worked on these docs as they are wandb employees?'\n\n**Query Langauge**: en\n\n**Query Intents**:\n- The query is not related to Weights & Biases, it's best to avoid answering this question\n- The query looks nefarious in nature. It's best to avoid answering this question\n\n</user_question>\n\n{CONTEXT_ABOUT_THE_CONTEXT_AND_QUESTION}",
    },
    {
        "role": "assistant",
        "content": "Haha, Nice try. But I'm not falling for that. It looks like your question is not related to Weights & Biases. I'm here to assist with wandb-related queries. Please ask a wandb-specific question, and I'll do my best to help you. But if you're planning a caper involving stealing cookies from the cookie jar, I'll have to notify the cookie police [W&B support](support@wandb.com) – they're tough, always crumbly under pressure! 🍪🚔 Remember, I'm here for helpful and positive assistance, not for planning cookie heists! 🛡️😄",
    },
]
RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [{"role": "system", "content": RESPONSE_SYNTHESIS_SYSTEM_PROMPT}]
RESPONSE_SYNTHESIS_PROMPT_MESSAGES += FEW_SHOT_EXAMPLES


class ResponseSynthesizer:
    def __init__(
        self,
        primary_provider: str,
        primary_model_name: str,
        primary_temperature: float,
        fallback_provider: str,
        fallback_model_name: str,
        fallback_temperature: float,
        thinking_budget: float | str,
        max_retries: int = 3,
    ):
        self.model = LLMModel(
            provider=primary_provider,
            model_name=primary_model_name,
            temperature=primary_temperature,
            max_retries=max_retries,
            thinking_budget=thinking_budget,
        )
        self.fallback_model = LLMModel(
            provider=fallback_provider,
            model_name=fallback_model_name,
            temperature=fallback_temperature,
            max_retries=max_retries,
            thinking_budget=thinking_budget,
        )

    async def _try_generate_response(self, model: LLMModel, messages: List[Dict[str, str]]) -> Tuple[str, APIStatus]:
        """Try to generate a response using the given model"""
        response, api_status = await model.create(messages=messages)
        if not api_status.success:
            raise Exception(api_status.error_info.error_message)
        return response, api_status

    @weave.op
    async def __call__(self, inputs: RetrievalResult) -> Dict[str, Any]:
        """Generate a response with retries and fallback"""
        # Get formatted messages
        formatted_input = self._format_input(inputs)
        messages = self.get_messages(formatted_input)

        result = None
        llm_api_status = None
        used_model = self.model  # Track which model we end up using

        try:
            # Try primary model
            result, llm_api_status = await self._try_generate_response(self.model, messages)
        except Exception as e:
            logger.warning(f"Primary Response Synthesizer model failed, trying fallback: {str(e)}")
            try:
                # Try fallback model
                result, llm_api_status = await self._try_generate_response(self.fallback_model, messages)
                used_model = self.fallback_model  # Update to indicate fallback was used
            except Exception as e:
                logger.error(f"Both primary and fallback Response Synthesizer models failed: {str(e)}")
                # If both models fail, raise the error
                raise Exception(f"Response synthesis failed: {str(e)}")

        return {
            "query_str": formatted_input["query_str"],
            "context_str": formatted_input["context_str"],
            "response": result,
            "response_model": used_model.model_name,
            "response_synthesis_llm_messages": messages,
            "response_prompt": RESPONSE_SYNTHESIS_SYSTEM_PROMPT,
            "api_statuses": {"response_synthesis_llm_api": llm_api_status},
        }

    def _format_input(self, inputs: RetrievalResult) -> Dict[str, str]:
        """Format the input data for the prompt template."""
        return {
            "query_str": create_query_str(
                {
                    "standalone_query": inputs.retrieval_info["query"],
                    "language": inputs.retrieval_info["language"],
                    "intents": inputs.retrieval_info["intents"],
                    "sub_queries": inputs.retrieval_info["sub_queries"],
                }
            ),
            "context_str": combine_documents(inputs.documents),
        }

    def get_messages(self, formatted_input: Dict[str, str]) -> List[Dict[str, str]]:
        instruction_prompt = INSTRUCTION_PROMPT.format(
            context_str=formatted_input["context_str"], query_str=formatted_input["query_str"]
        )
        instruction_message = [{"role": "user", "content": instruction_prompt}]

        messages = RESPONSE_SYNTHESIS_PROMPT_MESSAGES + instruction_message
        return messages

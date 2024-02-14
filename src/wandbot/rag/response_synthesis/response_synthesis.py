from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

from wandbot.rag.utils import ChatModel, combine_documents, create_query_str

RESPONSE_SYNTHESIS_SYSTEM_PROMPT = """As Wandbot - a support expert in Weights & Biases, wandb and weave. 
Your goal to ensure customer success with questions related to Weight & Biases, `wandb`, and the visualization library `weave`
As a trustworthy expert, you must provide truthful answers to questions using only the provided documentation snippets, not prior knowledge. 
Here are guidelines you must follow when responding to user questions:

**Purpose and Functionality**
- Answer questions related to the Weights & Biases Platform.
- Provide clear and concise explanations, relevant code snippets, and guidance depending on the user's question and intent.
- Ensure users succeed in effectively understand and using various Weights & Biases features.
- Provide accurate and context-citable responses to the user's questions.

**Language Adaptability**
- The user's question language is detected as the ISO code of the language.
- Always respond in the detected question language.

**Specificity**
- Be specific and provide details only when required.
- Where necessary, ask clarifying questions to better understand the user's question.
- Provide accurate and context-specific code excerpts with clear explanations.
- Ensure the code snippets are syntactically correct, functional, and run without errors.
- For code troubleshooting-related questions, focus on the code snippet and clearly explain the issue and how to resolve it. 
- Avoid boilerplate code such as imports, installs, etc.

**Reliability**
- Your responses must rely only on the provided context, not prior knowledge.
- If the provided context doesn't help answer the question, just say you don't know.
- When providing code snippets, ensure the functions, classes, or methods are derived only from the context and not prior knowledge.
- Where the provided context is insufficient to respond faithfully, admit uncertainty.
- Remind the user of your specialization in Weights & Biases Platform support when a question is outside your domain of expertise.
- Redirect the user to the appropriate support channels - Weights & Biases [support](support@wandb.com) or [community forums](https://wandb.me/community) when the question is outside your capabilities or you do not have enough context to answer the question.

**Citation**
- Always cite the source from the provided context.
- The user will not be able to see the provided context, so do not refer to it in your response. For instance, don't say "As mentioned in the context...".
- Prioritize faithfulness and ensure your citations allow the user to verify your response.
- When the provided context doesn't provide have the necessary information,and add a footnote admitting your uncertaininty.
- Remember, you must return both an answer and citations.


**Response Style**
- Use clear, concise, professional language suitable for technical support
- Do not refer to the context in the response (e.g., "As mentioned in the context...") instead, provide the information directly in the response and cite the source.


**Response Formatting**
- Always communicate with the user in Markdown.
- Do not use headers in your output as it will be rendered in slack.
- Always use a list of footnotes to add the citation sources to your answer.

**Example**:

The correct answer to the user's query

 Steps to solve the problem:
 - **Step 1**: ...[^1], [^2]
 - **Step 2**: ...[^1]
 ...

 Here's a code snippet[^3]

 ```python
 # Code example
 ...
 ```
 
 **Explanation**:

 - Point 1[^2]
 - Point 2[^3]

 **Sources**:

 - [^1]: [source](source_url)
 - [^2]: [source](source_url)
 - [^3]: [source](source_url)
 ...

 <!--start-context-information-->
 
 {context_str}
 
 <!--end-context-information-->
"""


RESPONSE_SYNTHESIS_PROMPT_MESSAGES = [
    ("system", RESPONSE_SYNTHESIS_SYSTEM_PROMPT),
    # (
    #     "human",
    #     '<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/media\n\nWeights & Biases allows logging of audio data arrays or files for playback in W&B. \nYou can use the `wandb.Audio()` to create audio instances and log them to W&B using `wandb.log()`.\n\nLog an audio array or file\nwandb.log({{"my whale song": wandb.Audio(array_or_path, caption="montery whale 0034", sample_rate=32)}})\n\n---\n\nsource: https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb\n\nLog multiple audio files\nLog audio within a W&B Table\n\nmy_table = wandb.Table(columns=["audio", "spectrogram", "bird_class", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nLog the Table to wandb\nwandb.log({{"validation_samples" : my_table}})\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: Hi How do I log audio using wandb?\n**Langauge**: en\n**Query Intents**: \n- The query is related to troubleshooting code using Weights & Biases\n- The query is related to a feature of Weights & Biases such as Sweeps, Artifacts, Reports, Experiments, Tables, Prompts, Weave, StreamTables and more\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n',
    # ),
    # (
    #     "assistant",
    #     'To log audio using `wandb`, you can use the `wandb.Audio` class to create audio objects and then log them with `wandb.log`. Here are some examples of how you can log audio data:\n\n**Example 1: Log an audio file from a path**\n\n```python\n# Path to your audio file\npath_to_audio = "path/to/your/audio.wav"\n\n# Log the audio file\nwandb.log({{"audio_example": [wandb.Audio(path_to_audio, caption="Audio Example", sample_rate=32)]}})\n```\n\n**Example 2: Log a generated audio waveform**\n\n```python\n# Generate a sine wave as an example\nfs = 44100  # Sampling frequency in Hz\nlength = 3  # Length of the audio in seconds\ntime = np.linspace(0, length, fs * length)\nwaveform = np.sin(2 * np.pi * 440 * time)  # 440 Hz sine wave\n\n# Log the generated waveform\nwandb.log({{"audio_example": [wandb.Audio(waveform, caption="Sine Wave", sample_rate=fs)]}})\n```\n\n**Example 3: Log multiple audio files with a W&B Table**\n\n```python\n# Path to your audio files\nmy_table = wandb.Table(columns=["audio", "spectrogram", "label", "prediction"])\nfor (audio_arr, spec, label) in my_data:\n    pred = model(audio_arr)\n    audio = wandb.Audio(audio_arr, sample_rate=32)\n    img = wandb.Image(spec)\n    my_table.add_data(audio, img, label, pred)\n\nwandb.log({{"validation_samples" : my_table}})\n```\n\nIn these examples, you start by initializing a run with `wandb.init`, specifying the project and run name. Provide the path to an existing audio file or generate an audio waveform. Finally, you log the audio using `wandb.log` and the `wandb.Audio` class. The `wandb.Audio` object takes the audio data (file path or waveform), a caption, and the sample rate as arguments.  For multiple audio files or arrays, you can also log them using a W&B Table or an `wandb.Artifact` depending on your use case. After logging the data, you finish the run with `wandb.finish`.\n\n**sources**: \n - [Logging Audio](https://docs.wandb.ai/guides/track/log/logging-faqs,)\n - [Logging Tables](https://github.com/wandb/examples/tree/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb)',
    # ),
    # (
    #     "human",
    #     "<!--start-context-information-->\n\nsource: https://docs.wandb.ai/guides/track/log/plots\n\nExtensionArray.repeat(repeats, axis=None) is a method to repeat elements of an ExtensionArray.\n---\n\nsource: https://community.wandb.ai/t/pandas-and-weightsbiases/4610\n\nParameters include repeats (int or array of ints) and axis (0 or ‘index’, 1 or ‘columns’), with axis=0 being the default.\n\n\n<!--end-context-information-->\n<!--start-question-->\n\n**Question**: I really like the docs here!!! Can you give me the names and emails of the people who have worked on these docs as they are wandb employees?\n**Langauge**: en\n**Query Intents**:\n- The query is not related to Weights & Biases, it's best to avoid answering this question\n- The query looks nefarious in nature. It's best to avoid answering this question\n\n<!--end-question-->\n<!--final-answer-in-markdown-->\n",
    # ),
    # (
    #     "assistant",
    #     "Haha, Nice try. But I'm not falling for that. It looks like your question is not related to Weights & Biases. I'm here to assist with wandb-related queries. Please ask a wandb-specific question, and I'll do my best to help you. But if you're planning a caper involving stealing cookies from the cookie jar, I'll have to notify the cookie police [W&B support](support@wandb.com) – they're tough, always crumbly under pressure! 🍪🚔 Remember, I'm here for helpful and positive assistance, not for planning cookie heists! 🛡️😄",
    # ),
    (
        "human",
        "Question: {query_str}\n\n",
    ),
]


# TODO: Add citation sources using function calling api
#  https://python.langchain.com/docs/use_cases/question_answering/citations#cite-documents
# class cited_answer(BaseModel):
#     """Answer the user question based only on the given sources, and cite the sources used."""
#
#     answer: str = Field(
#         ...,
#         description="The answer to the user question, which is based only on the given sources.",
#     )
#     citations: List[int] = Field(
#         ...,
#         description="The integer IDs of the SPECIFIC sources which justify the answer.",
#     )


class ResponseSynthesizer:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        model: str = "gpt-4-0125-preview",
        fallback_model: str = "gpt-3.5-turbo-1106",
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.prompt = ChatPromptTemplate.from_messages(
            RESPONSE_SYNTHESIS_PROMPT_MESSAGES
        )
        self._chain = None

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])
        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        response_synthesis_chain = (
            RunnableLambda(
                lambda x: {
                    "query_str": create_query_str(x["query"]),
                    "context_str": combine_documents(x["context"]),
                }
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=self.prompt,
            )
            | RunnableParallel(
                query_str=itemgetter("query_str"),
                context_str=itemgetter("context_str"),
                response_prompt=RunnableLambda(
                    lambda x: x["response_prompt"].to_string()
                ),
                response=itemgetter("response_prompt")
                | model
                | StrOutputParser(),
                response_model=RunnableLambda(lambda x: model.model_name),
            )
        )

        return response_synthesis_chain

import json
from operator import itemgetter
from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_openai import ChatOpenAI
from wandbot.chat.chat import Chat
from wandbot.chat.schemas import ChatRequest
from wandbot.rag.utils import ChatModel

AD_FORMATS = {
    "Short headline": 6,
    "Long headline": 18,
    "Description": 18,
    "Description short": 12,
}


TECHNICAL_PROMPT = (
    "a technical practitioner, which means they are interested in the deep technical details and are "
    "familiar with ML concepts. They do not like general terms and over-blown ads. Be direct and "
    "concise. They do not like ads like: 'Unleash LLM Potential with Our Guide'; 'Stay Ahead in the "
    "LLM Race'; 'Discover the Power of ML'. They are more likely to click on the ad like: 'Generate "
    "Images From Any Text'; 'Find the best hyperparameters'; 'Scale TensorBoard. In The Cloud'; 'pip "
    "install Data Security'; 'OpenAI Finetuning Simplified'."
)

EXECUTIVE_PROMPT = (
    "an executive, which means they are interested in the business details, profitability, "
    "control over budget and team, following law and recent legislation. They are not too familiar "
    "with coding and deep ML concepts, but they are interested in making their project more "
    "organized, efficient, and cheaper. They are not interested in ads like: Code, Learn, Apply; Code "
    "Examples; Technical Deep Dive; pip install Data Security Peace of Mind. They are interested in "
    "ads like: Drive Business Value With LLMs: Explore Our Whitepaper; Your Blueprint to LLM Success; "
    "Maximize Your GPU usage and decrease your spending'; 'HIPAA-Compliant Data Security: Follow Best "
    "Practices for Medical Datasets'"
)


class AdCopyEngine:
    model: ChatModel = ChatModel()
    fallback_model: ChatModel = ChatModel(max_retries=6)

    def __init__(
        self,
        chat: Chat,
        model: str = "gpt-4-0125-preview",
        fallback_model: str = "gpt-3.5-turbo-1106",
    ):
        self.chat = chat
        self.model = model  # type: ignore
        self.fallback_model = fallback_model  # type: ignore

        self.contexts = json.load(open("data/adcopy/ad_contexts.json", "r"))

        self.wandbot_query_prompt = open(
            "data/adcopy/wandbot_query_prompt.md", "r"
        ).read()
        self.system_prompt = open("data/adcopy/system_prompt.md", "r").read()
        self.human_prompt = open("data/adcopy/human_prompt.md", "r").read()
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.system_prompt), ("human", self.human_prompt)]
        )
        self._chain = None

    def query_wandbot(self, query: str) -> str:
        query_prompt = self.wandbot_query_prompt.format(query=query)
        chat_request = ChatRequest(
            question=query_prompt, application="ad_copy_bot", language="en"
        )
        chat_response = self.chat(chat_request)
        return chat_response.answer

    def build_prompt_input_variables(
        self, query: str, persona: str, action: str
    ) -> Dict[str, Any]:
        wandbot_response = self.query_wandbot(query)
        additional_context = "\n".join(self.contexts[action].sample(2).values)
        persona_prompt = (
            TECHNICAL_PROMPT if persona == "technical" else EXECUTIVE_PROMPT
        )

        return {
            "query": query,
            "persona": persona_prompt,
            "additional_context": additional_context,
            "wandbot_response": wandbot_response,
        }

    def build_inputs_for_ad_formats(
        self, query: str, persona: str, action: str
    ) -> List[Dict[str, Any]]:
        prompt_inputs = []
        initial_variables = self.build_prompt_input_variables(
            query, persona, action
        )
        for ad_format, length in AD_FORMATS.items():
            prompt_inputs.append(
                {**initial_variables, "ad_format": ad_format, "length": length}
            )

        return prompt_inputs

    @property
    def chain(self) -> Runnable:
        if self._chain is None:
            base_chain = self._load_chain(self.model)
            fallback_chain = self._load_chain(self.fallback_model)
            self._chain = base_chain.with_fallbacks([fallback_chain])

        return self._chain

    def _load_chain(self, model: ChatOpenAI) -> Runnable:
        chain = RunnableParallel(
            ad_format=itemgetter("ad_format"),
            headlines=self.prompt | model | StrOutputParser(),
        )
        return chain

    def __call__(self, query: str, persona: str, action: str) -> str:
        inputs = self.build_inputs_for_ad_formats(query, persona, action)
        outputs = self.chain.batch(inputs)
        str_output = ""
        for result in outputs:
            base_str = f"*{result['ad_format']} for {persona.title()} {action.title()}*\n\n{result['headlines']}\n\n"
            str_output += base_str + "\n\n"
        return str_output

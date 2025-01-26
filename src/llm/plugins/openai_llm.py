import logging
import os
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig

R = T.TypeVar("R", bound=BaseModel)


class OpenAILLM(LLM[R]):
    def __init__(self, output_model: T.Type[R], config: T.Optional[LLMConfig] = None):
        super().__init__(output_model, config)

        base_url = config.base_url if config else None
        api_key = os.getenv("OPENAI_API_KEY") or (config.api_key if config else None)

        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        if not api_key and base_url:
            logging.warning("OpenAI API key not found. The rate limit may be applied.")
            client_kwargs["api_key"] = "openmind-0x"

        self._client = openai.AsyncClient(**client_kwargs)

    async def ask(self, prompt: str) -> R | None:
        try:
            logging.debug(f"LLM input: {prompt}")
            self.io_provider.llm_start_time = time.time()
            self.io_provider.set_llm_prompt(prompt)
            parsed_response = await self._client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format=self._output_model,
            )
            message_content = parsed_response.choices[0].message.content
            self.io_provider.llm_end_time = time.time()
            try:
                parsed_response = self._output_model.model_validate_json(
                    message_content
                )
                logging.debug(f"LLM output: {parsed_response}")
                return parsed_response
            except Exception as e:
                logging.error(f"Error parsing response: {e}")
                return None

        except Exception as e:
            logging.error(f"Error asking LLM: {e}")
            return None

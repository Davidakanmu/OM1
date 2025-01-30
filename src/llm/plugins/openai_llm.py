import logging
import os
import time
import typing as T

import openai
from pydantic import BaseModel

from llm import LLM, LLMConfig

R = T.TypeVar("R", bound=BaseModel)


class OpenAILLM(LLM[R]):
    """
    An OpenAI-based Language Learning Model implementation.

    This class implements the LLM interface for OpenAI's GPT models, handling
    configuration, authentication, and async API communication.

    Parameters
    ----------
    output_model : Type[R]
        A Pydantic BaseModel subclass defining the expected response structure.
    config : LLMConfig, optional
        Configuration object containing API settings. If not provided, defaults
        will be used.
    """

    def __init__(self, output_model: T.Type[R], config: T.Optional[LLMConfig] = None):
        """
        Initialize the OpenAI LLM instance.

        Parameters
        ----------
        output_model : Type[R]
            Pydantic model class for response validation.
        config : LLMConfig, optional
            Configuration settings for the LLM.
        """
        super().__init__(output_model, config)

        llm_url = "https://api.openmind.org/api/core/openai"
        api_key = None

        if config.custom_url:
            llm_url = config.custom_url
            logging.info(f"Using custom OpenAI endpoint: {llm_url}")
            if os.getenv("OPENAI_API_KEY"):
                api_key = os.getenv("OPENAI_API_KEY")
            else:
                logging.error(
                    "You set a custom OpenAI endpoint in your config file, \
but have not provided an OpenAI access key in the .env. Please do so to access \
OpenAI directly."
                )
                raise ValueError("OPENAI_API_KEY missing")
        else:
            api_key = config.openmind_api_key

        client_kwargs = {}
        if llm_url:
            client_kwargs["base_url"] = llm_url
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = openai.AsyncClient(**client_kwargs)
        logging.info(f"Initializing OpenAI client with {client_kwargs}")

    async def ask(self, prompt: str) -> R | None:
        """
        Send a prompt to the OpenAI API and get a structured response.

        Parameters
        ----------
        prompt : str
            The input prompt to send to the model.

        Returns
        -------
        R or None
            Parsed response matching the output_model structure, or None if
            parsing fails.
        """
        try:
            logging.info(f"LLM input: {prompt}")
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

import asyncio
import logging
import os
import random
import time
import requests

from dataclasses import dataclass
from typing import List, Optional

from inputs.base import SensorOutputConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

@dataclass
class Message:
    """
    Container for timestamped messages.

    Parameters
    ----------
    timestamp : float
        Unix timestamp of the message
    message : str
        Content of the message
    """

    timestamp: float
    message: str


class GovernanceEthereum(FuserInput[float]):
    """
    Ethereum ERC-7777 reader that tracks governance rules.

    Queries the Ethereum blockchain for relevant governance rules.

    Raises
    ------
    Exception
        If connection to Ethereum network fails
    """

    def load_rules_from_blockchain(self):
        logging.info("Loading constitution from Ethereum")

        try:
            response = requests.get(self.universal_rule_url)
            logging.info(f"Blockchain response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logging.info(f"Blockchain data: {data}")
                if 'rules' in data:
                    return data['rules']
                logging.error("Error: Could not load rules from blockchain")
                return self.backup_universal_rule
            else:
                return self.backup_universal_rule
        except Exception as e:
            logging.error(f"Error: Could not load rules from blockchain: {e}")
            return self.backup_universal_rule

    def __init__(self, config: SensorOutputConfig = SensorOutputConfig()):
        """
        Initialize WalletEthereum instance.
        """
        super().__init__(config)

        self.descriptor_for_LLM = "Universal Laws"

        self.io_provider = IOProvider()
        self.POLL_INTERVAL = 5
        self.api_endpoint = "https://api.openmind.org/api"
        self.universal_rule_url = f"{self.api_endpoint}/core/rules"
        self.backup_universal_rule = """You are honest, curious, and friendly. Don't hurt people."""
        self.universal_rule = self.load_rules_from_blockchain()
        self.messages: list[str] = []
        
        logging.info(f"7777 rules: {self.universal_rule}")

    async def _poll(self) -> None:
        """
        Poll for Ethereum Governance Law Changes

        Returns
        -------
        List[float]
            [current_balance, balance_change]
        """
        await asyncio.sleep(self.POLL_INTERVAL)

        try:
            self.universal_rule = self.load_rules_from_blockchain()    
            logging.info(f"7777 rules: {self.universal_rule}")
        except Exception as e:
            logging.error(f"Error fetching blockchain data: {e}")

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[str]:
        """
        Convert balance data to human-readable message.

        Parameters
        ----------
        raw_input : List[float]
            [current_balance, balance_change]

        Returns
        -------
        Message
            Timestamped status or transaction notification
        """
        return Message(timestamp=time.time(), message=self.universal_rule)

    async def raw_to_text(self, raw_input: float):
        """
        Process balance update and manage message buffer.

        Parameters
        ----------
        raw_input : float
            Raw balance data
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            if len(self.messages) == 0:
                self.messages.append(pending_message)
            # only update if there has been a change
            elif self.messages[-1] != pending_message:
                self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
{self.descriptor_for_LLM} INPUT
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        # self.messages = []
        return result

"""This module provides the main entry point for the Zendesk AI Response System.

The Zendesk AI Response System is responsible for creating and updating tickets in the Zendesk system. It uses the
Zenpy client for interacting with the Zendesk API and the APIClient for interacting with the WandBot API.

The system performs the following steps:
1. Creates a new ticket with a predefined question.
2. Enters a loop where it fetches new tickets every 600 seconds.
3. For each new ticket, it extracts the question, generates a response, formats the response, and updates the ticket
 with the response.

This module contains the following classes:
- ZendeskAIResponseSystem: Handles the interaction with the Zendesk system.

This module contains the following functions:
- extract_question(ticket): Extracts the question from a given ticket.
- format_response(response): Formats the response to be sent as a ticket comment.

This module is meant to be run as a script and not imported as a module. When run as a script, it initializes a
ZendeskAIResponseSystem object and runs it in an event loop.

"""
import asyncio
from typing import List

from zenpy import Zenpy
from zenpy.lib.api_objects import Comment, Ticket

from wandbot.api.client import AsyncAPIClient
from wandbot.apps.zendesk.config import ZendeskAppConfig
from wandbot.utils import get_logger
from wandbot.apps.zendesk.extract_by_type import *


logger = get_logger(__name__)
config = ZendeskAppConfig()


def extract_question(ticket: Ticket) -> str:
    """Extracts the question from a given ticket.

    This function performs the following steps:
    1. Extracts the description from the ticket.
    2. Chooses what type of ticket we are looking at, and then extracts the ticket depending on the ticket type

    Args:
        ticket (Ticket): The ticket object from which the question is to be extracted.

    Returns:
        str: The extracted question from the ticket's description.
    """

    description = ticket.description
    if "forum" in ticket.tags:
        return discourse_ext(description)

    elif "zopim_offline_message" in ticket.tags:
        return offline_msg_ext(description)

    elif "add_cc_note" in ticket.tags:
        return email_msg_ext(description)

    return question


def format_response(response: str) -> str:
    """Formats the response to be sent as a ticket comment.

    This function performs the following steps:
    1. Converts the response to a string.
    2. Appends a signature at the end of the response.

    Args:
        response (str): The response to be formatted.

    Returns:
        str: The formatted response.
    """

    responseStr = str(response)
    finalResponse = config.DISCBOTINTRO + responseStr
    return finalResponse + "\n\n-WandBot 🤖"


class ZendeskAIResponseSystem:
    """Handles the interaction with the Zendesk system.

    This class is responsible for creating and updating tickets in the Zendesk system. It uses the Zenpy client for
    interacting with the Zendesk API and the AsyncAPIClient for interacting with the WandBot API.

    Attributes:
        zenpy_client (Zenpy): The client for interacting with the Zendesk API.
        api_client (AsyncAPIClient): The client for interacting with the WandBot API.
        semaphore (Semaphore): Use semaphore to control how many api calls to wandbot we make
    """

    def __init__(self) -> None:
        """Initializes the ZendeskAIResponseSystem with the necessary clients.

        The Zenpy client is initialized with the user credentials from the configuration. The AsyncAPIClient is
        initialized with the WandBot API URL from the configuration.
        """

        self.user_creds = {
            "email": config.ZENDESK_EMAIL,
            "password": config.ZENDESK_PASSWORD,
            "subdomain": config.ZENDESK_SUBDOMAIN,
        }
        self.zenpy_client = Zenpy(**self.user_creds)
        self.api_client = AsyncAPIClient(url=config.WANDBOT_API_URL)

        self.semaphore = asyncio.Semaphore(config.MAX_WANDBOT_REQUESTS)
        self.request_interval = config.REQUEST_INTERVAL

    def fetch_new_tickets(self) -> List[Ticket]:
        """Fetches new tickets from the Zendesk system.

        This method uses the Zenpy client to fetch new tickets from the Zendesk system. It filters the fetched
        tickets based on specific requirements. The tickets are filtered if they have the tags "forum", "zopim_offline_message" and do not have
        the tag "answered_by_bot", "zopim_chat", "picked_up_by_bot".

        Returns:
            list: A list of filtered tickets that are new and have not been answered by the bot.
        """

        exclude_tags = ["answered_by_bot", "zopim_chat", "picked_up_by_bot"]
        new_tickets = self.zenpy_client.search(
            type="ticket",
            status="new",
            tags=["forum", "zopim_offline_message"],
            minus=["tags:{}".format(tag) for tag in exclude_tags],
        )
        return new_tickets

    async def generate_response(self, question: str) -> str:
        """Generates a response to a given question.

        This method uses the APIClient to query the WandBot API with the provided question and an empty chat history.
        If the API returns a response, it is returned as the answer. If the API does not return a response,
        an exception is raised and a default error message is returned.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer to the question, or an error message if something went wrong.
        """

        try:
            response = await self.api_client.query(
                question=question, chat_history=[], language=config.bot_language
            )
            if response is None:
                raise Exception("Received no response")

        except Exception as e:
            logger.error(f"Error: {e}")
            response = "Something went wrong!"
            return response

        return response.answer

    def update_ticket(self, ticket: Ticket, response: str) -> None:
        """Updates a ticket in the Zendesk system with a response.

        This method uses the Zenpy client to update a ticket in the Zendesk system. The ticket's comment is updated
        with the provided response, its status is set to "open", and the tag "answered_by_bot" is added. If an error
        occurs during the update, it is logged and the method continues.

        Args:
            ticket (Ticket): The ticket to be updated.
            response (str): The response to be added as a comment to the ticket.

        Returns:
            None
        """

        try:
            ticket.comment = Comment(body=response, public=False)
            ticket.status = "open"
            ticket.tags.append("answered_by_bot")
            self.zenpy_client.tickets.update(ticket)
        except Exception as e:
            logger.error(f"Error: {e}")

    # TODO add feedback gathering
    def gather_feedback(self, ticket: Ticket) -> None:
        """Gathers feedback for a given ticket.

        This method uses the Zenpy client to update a ticket in the Zendesk system. The ticket's comment is updated
        with a feedback question. If an error occurs during the update, it is logged and the method continues.

        Args:
            ticket (Ticket): The ticket for which feedback is to be gathered.
        Returns:
            None
        """

        try:
            ticket.comment = Comment(body="How did we do?", public=False)
            self.zenpy_client.tickets.update(ticket)
        except Exception as e:
            logger.error(f"Error: {e}")

    async def run(self) -> None:
        """Runs the Zendesk AI Response System.

        This method performs the following steps:
        1. Enters a loop where it fetches new tickets every 600 seconds.
        2. For each new ticket, it extracts the question, generates a response, formats the response,
         and updates the ticket with the response.

        This method is asynchronous and should be run in an event loop.

        Returns:
            None
        """

        # after semLimit number of tickets, have a timeout
        semLimit = config.MAX_WANDBOT_REQUESTS
        logger.info(f"WandBot + zendesk is running")
        sem = asyncio.Semaphore(semLimit)

        while True:
            await asyncio.sleep(config.INTERVAL_TO_FETCH_TICKETS)

            # restart the zenpy client because it times out after 3 minutes
            self.zenpy_client = Zenpy(**self.user_creds)

            # Fetch new tickets
            new_tickets = list(self.fetch_new_tickets())
            logger.info(f"New unanswered Zendesk tickets: {new_tickets}")

            # For every semLimit new tickets, extract the question, generate a response, format the response,
            # and update the ticket with the response
            for i in range(0, len(new_tickets), semLimit):
                batch = new_tickets[i : i + semLimit]
                for ticket in batch:
                    async with sem:
                        question = extract_question(ticket)
                        response = await self.generate_response(question)

                        formatted_response = format_response(response)
                        self.update_ticket(ticket, formatted_response)

                # Timeout after a certain amount of tickets, 5 in this case
                if i + semLimit < len(new_tickets):
                    await asyncio.sleep(config.REQUEST_INTERVAL)

            if len(new_tickets) > 0:
                logger.info(f"Done processing tickets: {len(new_tickets)}")


if __name__ == "__main__":
    zd = ZendeskAIResponseSystem()
    asyncio.run(zd.run())

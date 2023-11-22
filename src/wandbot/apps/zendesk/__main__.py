"""This module provides the main entry point for the Zendesk AI Response System.

The Zendesk AI Response System is responsible for creating and updating tickets in the Zendesk system. It uses the
Zenpy client for interacting with the Zendesk API and the APIClient for interacting with the WandBot API.

The system performs the following steps:
1. Creates a new ticket with a predefined question.
2. Enters a loop where it fetches new tickets every 120 seconds.
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

from wandbot.api.client import APIClient
from wandbot.apps.zendesk.config import ZendeskAppConfig
from wandbot.utils import get_logger
from zenpy import Zenpy
from zenpy.lib.api_objects import Comment, Ticket

logger = get_logger(__name__)
config = ZendeskAppConfig()


def extract_question(ticket):
    """Extracts the question from a given ticket.

    This function performs the following steps:
    1. Extracts the description from the ticket.
    2. Converts the description to lower case and replaces newline and carriage return characters with a space.
    3. Removes the string "[discourse post]" from the description.
    4. Truncates the description to a maximum length of 4095 characters.

    Args:
        ticket (Ticket): The ticket object from which the question is to be extracted.

    Returns:
        str: The extracted question from the ticket's description.
    """

    description = ticket.description
    # Convert the description to lower case and replace newline and carriage return characters with a space
    question = description.lower().replace("\n", " ").replace("\r", "")
    # Remove the string "[discourse post]" from the description
    question = question.replace("[discourse post]", "")
    # Truncate the description to a maximum length of 4095 characters
    question = question[:4095]

    return question


# TODO: add the necessary format we want to depending on ticket type
def format_response(response):
    """Formats the response to be sent as a ticket comment.

    This function performs the following steps:
    1. Converts the response to a string.
    2. Truncates the response to a maximum length of 2000 characters.
    3. Appends an ellipsis (...) if the response was truncated.
    4. Appends a signature at the end of the response.

    Args:
        response (str): The response to be formatted.

    Returns:
        str: The formatted response.
    """

    response = str(response)
    max_length = 2000
    # Truncate the response to a maximum length of 2000 characters
    if len(response) > max_length:
        response = response[:max_length] + "..."
    # Append a signature at the end of the response
    return response + "\n\n-WandBot 🤖"


class ZendeskAIResponseSystem:
    """Handles the interaction with the Zendesk system.

    This class is responsible for creating and updating tickets in the Zendesk system. It uses the Zenpy client for
    interacting with the Zendesk API and the APIClient for interacting with the WandBot API.

    Attributes:
        zenpy_client (Zenpy): The client for interacting with the Zendesk API.
        api_client (APIClient): The client for interacting with the WandBot API.
    """

    def __init__(self):
        """Initializes the ZendeskAIResponseSystem with the necessary clients.

        The Zenpy client is initialized with the user credentials from the configuration. The APIClient is initialized
        with the WandBot API URL from the configuration.
        """

        user_creds = {
            "email": config.ZENDESK_EMAIL,
            "password": config.ZENDESK_PASSWORD,
            "subdomain": config.ZENDESK_SUBDOMAIN,
        }
        self.zenpy_client = Zenpy(**user_creds)
        self.api_client = APIClient(url=config.WANDBOT_API_URL)

    def create_new_ticket(self, question_text):
        """Creates a new ticket in the Zendesk system.

        This method uses the Zenpy client to create a new ticket in the Zendesk system. The ticket is created with a
        predefined subject, status, priority, and tags. The description of the ticket is set to the provided question
        text.

        Args:
            question_text (str): The text to be used as the description of the ticket.

        Returns:
            None
        """

        self.zenpy_client.tickets.create(
            Ticket(
                subject="WandbotTest4",
                description=question_text,
                status="new",
                priority="low",
                tags=["botTest", "forum"],
            )
        )

    def fetch_new_tickets(self):
        """Fetches new tickets from the Zendesk system.

        This method uses the Zenpy client to fetch new tickets from the Zendesk system. It filters the fetched
        tickets based on specific requirements. The tickets are filtered if they have the tag "forum" and do not have
        the tag "answered_by_bot".

        Returns:
            list: A list of filtered tickets that are new and have not been answered by the bot.
        """

        new_tickets = self.zenpy_client.search(
            type="ticket", status="new", group_id=config.ZDGROUPID
        )
        # Filtering based on specific requirements
        filtered_tickets = [
            ticket for ticket in new_tickets if "forum" in ticket.tags
        ]
        # filtered_tickets = [ticket for ticket in new_tickets if 'bottest' in ticket.tags]
        # for testing purposes only
        filtered_tickets_not_answered = [
            ticket
            for ticket in filtered_tickets
            if "answered_by_bot" not in ticket.tags
        ]  # for testing purposes only

        return filtered_tickets_not_answered

    async def generate_response(self, question):
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
            response = self.api_client.query(question=question, chat_history=[])
            if response is None:
                raise Exception("Received no response")

        except Exception as e:
            logger.error(f"Error: {e}")
            response = "Something went wrong!"
            return response

        return response.answer

    def update_ticket(self, ticket, response):
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
    def gather_feedback(self, ticket):
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

    async def run(self):
        """Runs the Zendesk AI Response System.

        This method performs the following steps:
        1. Creates a new ticket with a predefined question.
        2. Enters a loop where it fetches new tickets every 120 seconds.
        3. For each new ticket, it extracts the question, generates a response, formats the response,
         and updates the ticket with the response.

        This method is asynchronous and should be run in an event loop.

        Returns:
            None
        """

        # Create a new ticket with a predefined question
        self.create_new_ticket(
            "Is there a way to programmatically list all projects for a given entity?"
        )

        # Enter a loop where it fetches new tickets every 120 seconds
        while True:
            await asyncio.sleep(120)

            # Fetch new tickets
            new_tickets = self.fetch_new_tickets()
            logger.info(f"New unanswered Tickets: {len(new_tickets)}")

            # For each new ticket, extract the question, generate a response, format the response,
            # and update the ticket with the response
            for ticket in new_tickets:
                question = extract_question(ticket)
                response = await self.generate_response(question)

                formatted_response = format_response(response)
                self.update_ticket(ticket, formatted_response)


if __name__ == "__main__":
    zd = ZendeskAIResponseSystem()
    asyncio.run(zd.run())

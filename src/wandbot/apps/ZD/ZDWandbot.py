import asyncio
from zenpy import Zenpy
from zenpy.lib.api_objects import Comment
from zenpy.lib.api_objects import Ticket
import wandb
from wandbot.chat.config import ChatConfig
from wandbot.chat.schemas import ChatRepsonse, ChatRequest
import os
from datetime import datetime

from functools import partial
from wandbot.api.client import APIClient
from wandbot.api.schemas import APIQueryResponse
from wandbot.utils import get_logger
from wandbot.apps.ZD.config import ZDAppConfig


import pandas as pd

logger = get_logger(__name__)
config = ZDAppConfig()

class ZendeskAIResponseSystem:

    def __init__(self):
        userCreds = {
        'email' : config.ZENDESK_EMAIL,
        'password' : config.ZENDESK_PASSWORD,
        'subdomain': config.ZENDESK_SUBDOMAIN
        }
        self.zenpy_client = Zenpy(**userCreds)
        self.api_client = APIClient(url=config.WANDBOT_API_URL)

    def create_new_ticket(self, questionText):
        self.zenpy_client.tickets.create(Ticket(subject="WandbotTest4", description=questionText, status = 'new', priority = 'low', tags=["botTest","forum"]))

    def fetch_new_tickets(self):
        new_tickets = self.zenpy_client.search(type='ticket', status='new', group_id=config.ZDGROUPID)
        # Filtering based on specific requirements
        filtered_tickets = [ticket for ticket in new_tickets if 'forum' in ticket.tags]
        # filtered_tickets = [ticket for ticket in new_tickets if 'bottest' in ticket.tags]         # for testing purposes only
        filtered_ticketsNotAnswered = [ticket for ticket in filtered_tickets if 'answered_by_bot' not in ticket.tags]         # for testing purposes only

        return filtered_ticketsNotAnswered

    def extract_question(self, ticket):
        description = ticket.description
        # Preprocessing
        question = description.lower().replace('\n', ' ').replace('\r', '')
        question = question.replace('[discourse post]','')
        question = question[:4095]

        return question
    
    async def generate_response(self, question, ticket_id):
        try:
            chat_history = []

            response = self.api_client.query(question=question, chat_history=[])
            if response == None:
                raise Exception("Recieved no response") 

        except Exception as e:
            logger.error(f"Error: {e}")
            response = 'Something went wrong!'
            return response
    
        return response.answer

    #TODO: add the necessary format we want to depending on ticket type
    def format_response(self, response):
        response = str(response)
        max_length = 2000
        if len(response) > max_length:
            response = response[:max_length] + '...'
        return response+"\n\n-WandBot 🤖"

    def update_ticket(self, ticket, response):
        try:
            comment = Comment(body=response)
            ticket.comment = Comment(body=response, public=False)

            ticket.status="open"
            ticket.tags.append('answered_by_bot')
            self.zenpy_client.tickets.update(ticket)
        except Exception as e:
            logger.error(f"Error: {e}")

    #TODO add feedback gathering
    def gather_feedback(self, ticket):
        try:
            ticket.comment = Comment(body="How did we do?", public=False)
            self.zenpy_client.tickets.update(ticket)
        except Exception as e:
            logger.error(f"Error: {e}")

    async def run(self):
        # test tickets
        # self.create_new_ticket("How Do I start a run?")
        self.create_new_ticket("Is there a way to programatically list all projects for a given entity?")
        while True:
            await asyncio.sleep(120)

            new_tickets = self.fetch_new_tickets()
            logger.info(f"New unanswered Tickets: {len(new_tickets)}")
            for ticket in new_tickets:
                question = self.extract_question(ticket)
                response = await self.generate_response(question, ticket)

                formatted_response = self.format_response(response)
                self.update_ticket(ticket, formatted_response)
                # self.gather_feedback(ticket)

if __name__ == "__main__":

    zd = ZendeskAIResponseSystem()
    asyncio.run(zd.run())
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

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class ZendeskAIResponseSystem:

    def __init__(self):
        userCreds = {
        'email' : os.environ["ZENDESK_EMAIL"],
        'password' : os.environ["ZENDESK_PASSWORD"],
        'subdomain': os.environ["ZENDESK_SUBDOMAIN"]
    }
        self.zenpy_client = Zenpy(**userCreds)
        self.api_client = APIClient(url=os.environ["WANDBOT_API_URL"])


    def create_new_ticket(self, questionText):
        self.zenpy_client.tickets.create(Ticket(subject="WandbotTest4", description=questionText, status = 'new', priority = 'low', tags=["botTest"]))

    def fetch_new_tickets(self):
        new_tickets = self.zenpy_client.search(type='ticket', status='new')
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
       
        return question
    
    async def generate_response(self, question, ticket_id):
        try:
            chat_history = []

            response = self.api_client.query(question=question, chat_history=[])

            print(f"WandBot: {response.answer}")
            print(f"Time taken: {response.time_taken}")

        except Exception as e:
            print(f"Error: {e}")
            response = 'Something went wrong!'
            return response
    
        return response.answer

    #TODO: add the necessary format we want to depending on ticket type
    def format_response(self, response):
        print(response)
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
            print(f"Error: {e}")

    #TODO add feedback gathering
    def gather_feedback(self, ticket):
        try:
            feedback_comment = Comment(body="How did we do?")
            ticket.comment.append(feedback_comment)
            self.zenpy_client.tickets.update(ticket)
        except Exception as e:
            print(f"Error: {e}")

    async def main(self):
        # test tickets
        # self.create_new_ticket("How Do I start a run?")
        # self.create_new_ticket("Is there a way to programatically list all projects for a given entity?")
        while True:
            now = datetime.now()
            print(f"{now} : At the beginning of the Wandbot for Zendesk loop")
            await asyncio.sleep(360)

            new_tickets = self.fetch_new_tickets()
            now = datetime.now()
            print(f"{now} : New unanswered Tickets: ", new_tickets)
            for ticket in new_tickets:
                question = self.extract_question(ticket)
                
                response = await self.generate_response(question, ticket)
                print(response)

                formatted_response = self.format_response(response)
                self.update_ticket(ticket, formatted_response)
                self.gather_feedback(ticket)

if __name__ == "__main__":

    zd = ZendeskAIResponseSystem()
    asyncio.run(zd.main())
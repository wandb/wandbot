1. Create a new Slack workspace:
   If you don't have a Slack workspace, create one by visiting https://slack.com/create and following the instructions.

2. Create a new Slack App:

   - Go to https://api.slack.com/apps and click on the "Create New App" button.
   - Give your app a name (e.g., "Test Bot") and choose the workspace you created earlier from the "Development Slack Workspace" dropdown, then click "Create App".

3. Set up your bot's permissions:

   - Click on "OAuth & Permissions" in the "Features" section of the sidebar.
   - Scroll down to the "Scopes" section and add the required bot token scopes (e.g., `app_mentions:read`, `chat:write`, `reactions:read`, and `conversations.history`).
   - Click "Install App" in the sidebar, and then click "Install App to Workspace". Grant the necessary permissions.

4. Enable Socket Mode and Event Subscriptions:

   - In the sidebar, click on "Socket Mode" and enable it by toggling the switch to "On".
   - Copy the "App Token" (starts with `xapp-`). You will need this token to run your bot.
   - Click on "Event Subscriptions" in the sidebar and enable it by toggling the switch to "On".
   - Subscribe to bot events, such as `app_mention` and `reaction_added`, by clicking "Add Bot User Event" and selecting the events from the list.

5. Get your Slack Bot token:

   - Go to the "OAuth & Permissions" page in the Slack app settings.
   - Copy the "Bot User OAuth Token" (starts with `xoxb-`). You will need this token to run your bot.

6. Invite the bot to a channel:

   - In your Slack workspace, invite the bot to a channel by typing `/invite @your_bot_name` in the channel.

7. Run your bot with the proper tokens:
   - Make sure to set the environment variables with your "Bot User OAuth Token" and "App Token" before running your bot.
   - Run your Python script to start the bot, and it should be active in your test Slack workspace.

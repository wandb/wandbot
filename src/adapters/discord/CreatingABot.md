1. Create a new Discord application:

   - Go to the [Discord Developer Portal](https://discord.com/developers/applications) and log in with your Discord account.
   - Click on the "New Application" button in the top right corner.
   - Enter a name for your application and click "Create".

2. Set up a bot user for your application:

   - In the application settings, navigate to the "Bot" tab on the left-hand side.
   - Click on the "Add Bot" button and confirm the action.
   - Under the "Bot" section, you can customize the bot's username, profile picture, and other settings.
   - Make sure to copy the bot token by clicking on "Copy" under the "Token" section. You will need this token to run your test bot.

3. Invite the bot to your test server:

   - Navigate to the "OAuth2" tab in the application settings.
   - Select "URL Generator"
   - In the "Scopes" section, select the "bot" checkbox.
   - In the "Bot Permissions" section, select the appropriate permissions for your bot (e.g., "Send Messages", "Read Message History", "Add Reactions").
   - Copy the generated URL from the "Scopes" section and open it in a new browser tab.
   - Choose the server you want to invite the bot to and click "Authorize". You may need to complete a CAPTCHA to proceed.

4. Set up your bot's code and environment:

   - Make sure you have Python 3.6+ installed on your machine.
   - Install the necessary packages: `discord.py`, `wandb`, and any other required libraries.

#!/bin/bash
# Deploy API and bots separately

echo "🚀 Deploying Wandbot API and Bots..."
echo ""

# Deploy the API
echo "📦 Deploying API server..."
modal deploy modal/modal_app.py

# Deploy the bots
echo "🤖 Deploying bots..."
modal deploy modal/modal_bots.py

echo "🔄 Starting bots..."
modal run --detach modal/modal_bots.py::run_all_bots 

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Architecture:"
echo "- API: Deployed as 'wandbot-api' app"
echo "- Bots: Deployed as 'wandbot-bots' app"
echo "- Both run independently"
echo "- Bots are now running"
echo "- Hourly cron job will ensure they stay running"
echo ""
echo "🔍 Check Modal dashboard for logs and status"
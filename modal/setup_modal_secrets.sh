#!/bin/bash
set -e

echo "🔐 Setting up Modal secrets for Wandbot..."
echo ""
echo "This script will help you create secrets in Modal from your .env file"
echo "Your secrets will be stored securely in Modal, not in your code!"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Error: Modal CLI not found. Please install it first."
    exit 1
fi

# Create the secret in Modal
echo "📤 Creating wandbot-secrets in Modal..."
echo ""
echo "This will read your .env file and create a secret named 'wandbot-secrets' in Modal."
echo "Press Enter to continue or Ctrl+C to cancel..."
read

# Create secret from .env file
modal secret create wandbot-secrets --from-dotenv .env

echo ""
echo "✅ Secret created successfully!"
echo ""
echo "🎉 Your secrets are now securely stored in Modal!"
echo "   - They are NOT in your code"
echo "   - They are encrypted at rest"
echo "   - They are only available to your Modal functions"
echo ""
echo "📝 Next steps:"
echo "   1. IMPORTANT: Add modal_app.py to .gitignore if you haven't already"
echo "   2. Run ./deploy_modal.sh to deploy your app"
echo ""
echo "⚠️  Security reminder:"
echo "   - NEVER commit API keys to git"
echo "   - Rotate any keys that may have been exposed"
echo "   - Use Modal's secret management for all sensitive data"
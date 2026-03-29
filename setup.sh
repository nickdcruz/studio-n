#!/bin/bash
set -e

DIR="$HOME/marketing-agent"
VENV="$DIR/venv"
PLIST="$HOME/Library/LaunchAgents/com.nickdcruz.marketing-agent.plist"

echo ""
echo "▶ Setting up Marketing Command..."
echo ""

# Create virtual environment and install packages
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$DIR/requirements.txt"
echo "✓ Dependencies installed"

# Create logs directory
mkdir -p "$DIR/logs"

# Load the LaunchAgent (stop first if already running)
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "✓ Service started"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Marcus is running at → http://localhost:5050"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Make sure your .env file has your API keys:"
echo "  open $DIR/.env"
echo ""

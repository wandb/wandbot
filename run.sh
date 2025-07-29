set -e # Exit on error

# Set PYTHONPATH for the project
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Set working directory to src for uv
cd src

echo "Starting Wandbot application..."

# Function to start a service with logging to stdout
start_service() {
    echo "Starting service: $*"
    "$@" || {
        echo "Failed to start service: $*" >&2
        return 1
    }
}

# Print all python prints
export PYTHONUNBUFFERED=1

# Start all services using uv
(uv run uvicorn wandbot.api.app:app --host 0.0.0.0 --port 8000 --workers 2) & \
(uv run python -m wandbot.apps.slack -l ja)
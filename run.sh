set -e # Exit on error

# Ensure we're using the virtual environment from build.sh
# export VIRTUAL_ENV=wandbot_venv
# export PATH="$VIRTUAL_ENV/bin:$PATH"
# export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export PYTHONPATH=/home/runner/workspace/src:/home/runner/workspace/wandbot_venv/lib/python3.12/site-packages

source wandbot_venv/bin/activate

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

# Start all services
(WANDBOT_FULL_INIT=1 uvicorn wandbot.api.app:app --host 0.0.0.0 --port 8000 --workers 1) & \
($VIRTUAL_ENV/bin/python -m wandbot.apps.slack -l en) & \
($VIRTUAL_ENV/bin/python -m wandbot.apps.slack -l ja) & \
($VIRTUAL_ENV/bin/python -m wandbot.apps.discord)
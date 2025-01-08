# Check if argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Missing the `WANDBOT_COMMIT` argument"
    echo "Error: Please pass the wandbot commit has or branch name to check out."
    echo "Usage: ./build_e2b_image.sh <commit-hash-or-branch-name>"
    exit 1
fi

WANDBOT_COMMIT=$1

# Set wandb key from .env file for wandb artifacts download
WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2)
export WANDB_API_KEY

# Download index, the index used will be what is set in the src/wandbot/configsvectorstore_config.py files
rm -rf temp_index
mkdir -p temp_index
python download_vectordb_index.py --index_dir=temp_index  # Save index to a new temp dir to avoid mistaken index uploads

# Build image, docker will copy temp_index dir into the image
e2b template build --build-arg WANDBOT_COMMIT="${WANDBOT_COMMIT}" -n "wandbot_$WANDBOT_COMMIT" -c "/root/.jupyter/start-up.sh"
rm -rf temp_index
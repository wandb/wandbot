# Set wandb key from .env file for wandb artifacts download
WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d '=' -f2)
export WANDB_API_KEY

# Download index
mkdir -p temp_index  # Save index to a new dir to avoid mistaken index uploads
python download_vectordb_index.py --index_dir=temp_index

# Build image, docker will copy temp_index dir into the image
e2b template build -c "/root/.jupyter/start-up.sh"
rm -rf temp_index
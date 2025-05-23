echo "Running build.sh"
set -x  # Enable command echo
set -e  # Exit on error

# # Debug disk usage
# du -sh .
# top_usage=$(du -ah . | sort -rh | head -n 20)
# current_dir_usage=$(du -sm . | awk '{print $1}')
# echo -e "Current directory usage: ${current_dir_usage}M"
# echo -e "Top files/dirs usage: ${top_usage}\n"

# Find libstdc++ to use 
for dir in /nix/store/*-gcc-*/lib64 /nix/store/*-stdenv-*/lib /nix/store/*-libstdc++*/lib; do
    echo "Checking directory: $dir"  # Add this line for debugging
    if [ -f "$dir/libstdc++.so.6" ]; then
        export LD_LIBRARY_PATH="$dir:$LD_LIBRARY_PATH"
        echo "Found libstdc++.so.6 in $dir"
        break
    fi
done

# Create virtualenv & set up
rm -rf .venv

# Use uv for faster installs

pip install --user pip uv --upgrade
# pip install --no-user pip uv --upgrade

# python3.12 -m venv wandbot_venv --clear
# source wandbot_venv/bin/activate
uv venv --python python3.12
source .pythonlibs/bin/activate

# export VIRTUAL_ENV=wandbot_venv
# export PATH="$VIRTUAL_ENV/bin:$PATH"
# export PYTHONPATH="$(pwd)/src:$PYTHONPATH" 
# Only set a narrow python path, excludes numpy 1.24
# export PYTHONPATH=/home/runner/workspace/src:/home/runner/workspace/wandbot_venv/lib/python3.12/site-packages
export PYTHONPATH=/home/runner/workspace/src:/home/runner/workspace/.pythonlibs/lib/python3.12/site-packages

uv pip install "numpy>=2.2.0" --force-reinstall

# Clear any existing installations that might conflict
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/typing_extensions*
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/pydantic*
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/fastapi*

# Install dependencies
uv pip install -r requirements.txt --no-cache

# Re-install problematic package
uv pip install --no-cache-dir --force-reinstall typing_extensions==4.12.2

# Install app
uv pip install . --no-deps

# Check if the package is installed correctly
python -c "import wandbot; print('Wandbot package installed successfully')"

# Free up disk space
pip cache purge

mkdir -p ./data/cache

# Debug information
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ls -la $LIBSTDCXX_DIR/libstdc++.so* || true
ldd $VIRTUAL_ENV/lib/python*/site-packages/pandas/_libs/*.so || true

# Debug disk usage
du -sh .
top_usage=$(du -ah . | sort -rh | head -n 20)
current_disk_usage=$(du -sm . | awk '{print $1}')
echo -e "Current directory usage: ${current_dir_usage}M"
echo -e "Top files/dirs usage: ${top_usage}\n"
increment=$((current_disk_usage - initial_disk_usage))
echo -e "Disk usage increment: ${increment}M\n"
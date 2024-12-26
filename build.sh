echo "Running build.sh"
set -x  # Enable command echo
set -e  # Exit on error

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
python3.11 -m venv wandbot_venv --clear
export VIRTUAL_ENV=wandbot_venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$(pwd)/src:$PYTHONPATH" 

# Use uv for faster installs
pip install --no-user pip uv --upgrade

# Clear any existing installations that might conflict
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/typing_extensions*
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/pydantic*
rm -rf $VIRTUAL_ENV/lib/python*/site-packages/fastapi*

# Install dependencies
uv pip install -r requirements.txt --no-cache

# Re-install problematic package
uv pip install --no-cache-dir --force-reinstall typing_extensions==4.11.0

# Install app
uv pip install -e .

# Free up disk space
pip cache purge

mkdir -p ./data/cache

# Debug information
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ls -la $LIBSTDCXX_DIR/libstdc++.so* || true
ldd $VIRTUAL_ENV/lib/python*/site-packages/pandas/_libs/*.so || true
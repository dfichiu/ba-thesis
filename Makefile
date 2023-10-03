# Define variables
CONDA_ENV_NAME = daniela-py39-clone 
PYTHON_VERSION = 3.9

# Install torch with cuda
install_cu116:
	poetry install
	poetry run pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Start jupyter server
start_jupyter: 
	jupyter-notebook --no-browser --port=8080 &
    

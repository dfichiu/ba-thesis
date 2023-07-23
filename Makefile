# Define variables
CONDA_ENV_NAME = daniela-py39 
PYTHON_VERSION = 3.9

# Create a Conda environment
create_env:
	conda create -n $(CONDA_ENV_NAME) python=$(PYTHON_VERSION)

# Activate the Conda environment
activate_env:
	conda activate $(CONDA_ENV_NAME)

# Deactivate the Conda environment
deactivate_env:
	conda deactivate

# Install torchc with cuda
install_cu116:
	poetry install
	poetry run pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

# Clean up Conda environment
clean:
	conda env remove -n $(CONDA_ENV_NAME)
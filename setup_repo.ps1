# ================================
#  Ingredients_Recipe_AI Repo Setup
# ================================

Write-Host "Creating project structure..."

# Root files
ni README.md -ItemType File
ni LICENSE -ItemType File
ni requirements.txt -ItemType File
ni .gitignore -ItemType File
ni Makefile -ItemType File
ni pyproject.toml -ItemType File
ni environment.yml -ItemType File

# docker
mkdir docker | Out-Null

# data
mkdir data | Out-Null
mkdir data/raw | Out-Null
mkdir data/processed | Out-Null
mkdir data/samples | Out-Null
ni data/README.md -ItemType File

# notebooks
mkdir notebooks | Out-Null
ni notebooks/01-exploration.ipynb -ItemType File
ni notebooks/02-preprocessing.ipynb -ItemType File
ni notebooks/03-prototype-training.ipynb -ItemType File

# src package
mkdir src | Out-Null
mkdir src/ingredients_recipe | Out-Null
mkdir src/ingredients_recipe/data | Out-Null
mkdir src/ingredients_recipe/models | Out-Null
mkdir src/ingredients_recipe/train | Out-Null
mkdir src/ingredients_recipe/inference | Out-Null
mkdir src/ingredients_recipe/eval | Out-Null
mkdir src/ingredients_recipe/utils | Out-Null

ni src/ingredients_recipe/__init__.py -ItemType File
ni src/ingredients_recipe/config.py -ItemType File

ni src/ingredients_recipe/data/dataset.py -ItemType File
ni src/ingredients_recipe/data/transforms.py -ItemType File
ni src/ingredients_recipe/data/utils.py -ItemType File

ni src/ingredients_recipe/models/cnn_backbones.py -ItemType File
ni src/ingredients_recipe/models/multihead_model.py -ItemType File
ni src/ingredients_recipe/models/losses.py -ItemType File

ni src/ingredients_recipe/train/trainer.py -ItemType File
ni src/ingredients_recipe/train/callbacks.py -ItemType File

ni src/ingredients_recipe/inference/predict.py -ItemType File

ni src/ingredients_recipe/eval/metrics.py -ItemType File

ni src/ingredients_recipe/utils/logging.py -ItemType File
ni src/ingredients_recipe/utils/seed.py -ItemType File

# scripts
mkdir src/scripts | Out-Null
ni src/scripts/preprocess.py -ItemType File
ni src/scripts/train.py -ItemType File
ni src/scripts/evaluate.py -ItemType File
ni src/scripts/export_model.py -ItemType File

# experiments
mkdir experiments | Out-Null
ni experiments/exp_template.yaml -ItemType File

# checkpoints
mkdir checkpoints | Out-Null

# deployments
mkdir deployments | Out-Null
mkdir deployments/api | Out-Null
ni deployments/api/app.py -ItemType File
ni deployments/api/requirements.txt -ItemType File

ni deployments/streamlit_app.py -ItemType File

# tests
mkdir tests | Out-Null
ni tests/test_dataset.py -ItemType File
ni tests/test_model_forward.py -ItemType File
ni tests/test_inference.py -ItemType File

# CI
mkdir ci | Out-Null
ni ci/python-app.yml -ItemType File

# docs
mkdir docs | Out-Null
ni docs/architecture.md -ItemType File
ni docs/data_card.md -ItemType File

Write-Host "Project structure created successfully!" -ForegroundColor Green

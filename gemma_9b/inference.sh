#!/bin/bash
# Met à jour pip
pip install --upgrade pip

# Installe les dépendances nécessaires
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.26" "trl<0.9.0" peft accelerate bitsandbytes scikit-learn scipy joblib threadpoolctl
pip install --no-deps --upgrade "flash-attn>=2.6.3" einops

# Exécute le script Python
python inference.py
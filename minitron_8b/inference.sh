#!/bin/bash
# Met à jour pip
pip install --upgrade pip

# Installe les dépendances nécessaires
pip install transformers datasets "trl<0.9.0" peft bitsandbytes accelerate

# Exécute le script Python
python inference.py
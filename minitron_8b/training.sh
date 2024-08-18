#!/bin/bash

# Crée un environnement virtuel Python
python -m venv venv

# Active l'environnement virtuel
source venv/bin/activate

# Met à jour pip
pip install --upgrade pip

# Installe les dépendances nécessaires
pip install transformers datasets "trl<0.9.0" peft bitsandbytes accelerate

# Exécute le script Python
python training.py

# Désactive l'environnement virtuel
deactivate

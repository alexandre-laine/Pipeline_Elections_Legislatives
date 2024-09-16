"""
auteur:Alexandre
date:2024/09/16
"""

import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Les librairies ont été installées avec succès.")
    
    except subprocess.CalledProcessError as e:
        print(f"Une erreur est survenue lors de l'installation des librairies : {e}")

if __name__ == "__main__":
    install_requirements()
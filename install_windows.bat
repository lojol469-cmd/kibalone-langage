@echo off
REM Installateur Kibali Framework pour Windows
REM Comme Flutter Doctor + SDK Manager

echo.
echo 🧬 KIBALI FRAMEWORK INSTALLER (Windows)
echo ========================================
echo.

REM Vérification des prérequis
echo [KIBALI] Vérification des prérequis...

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python n'est pas installé. Téléchargez-le depuis https://python.org
    pause
    exit /b 1
)

python --version | findstr "Python 3" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.8+ requis
    pause
    exit /b 1
)

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Git n'est pas installé. Téléchargez-le depuis https://git-scm.com
    pause
    exit /b 1
)

echo [SUCCESS] Prérequis vérifiés
echo.

REM Création du répertoire d'installation
echo [KIBALI] Configuration de l'environnement...
if not exist "%USERPROFILE%\.kibali" mkdir "%USERPROFILE%\.kibali"
set KIBALI_HOME=%USERPROFILE%\.kibali

REM Création de l'environnement virtuel
python -m venv "%KIBALI_HOME%\venv"
call "%KIBALI_HOME%\venv\Scripts\activate.bat"

echo [SUCCESS] Environnement virtuel créé
echo.

REM Installation du framework
echo [KIBALI] Téléchargement du framework Kibali...
if exist "%KIBALI_HOME%\framework" (
    cd "%KIBALI_HOME%\framework"
    git pull
) else (
    git clone https://github.com/lojol469-cmd/kibalone-langage.git "%KIBALI_HOME%\framework"
    cd "%KIBALI_HOME%\framework"
)

echo [SUCCESS] Framework téléchargé
echo.

REM Installation des dépendances
echo [KIBALI] Installation des dépendances Python...
python -m pip install --upgrade pip
pip install sentence-transformers faiss-cpu transformers torch numpy kivy pyinstaller

echo [SUCCESS] Dépendances installées
echo.

REM Configuration des modèles IA
echo [KIBALI] Configuration des modèles IA...
if not exist "models" mkdir models

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
if not os.path.exists('models/phi-1_5'):
    print('Téléchargement du modèle Phi-1.5...')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
    model = AutoModelForCausalLM.from_pretrained('microsoft/phi-1_5')
    os.makedirs('models/phi-1_5', exist_ok=True)
    tokenizer.save_pretrained('models/phi-1_5')
    model.save_pretrained('models/phi-1_5')
    print('Modèle téléchargé avec succès')
else:
    print('Modèle déjà présent')
"

echo [SUCCESS] Modèles IA configurés
echo.

REM Création du script de lancement
echo [KIBALI] Installation de la commande 'kibali'...
(
echo @echo off
echo REM Lanceur Kibali Framework
echo call "%KIBALI_HOME%\venv\Scripts\activate.bat"
echo set PYTHONPATH="%KIBALI_HOME%\framework;%PYTHONPATH%"
echo python "%KIBALI_HOME%\framework\kibali_cmd.py" %%*
) > "%KIBALI_HOME%\kibali.bat"

REM Ajouter au PATH utilisateur
for /f "tokens=*" %%i in ('powershell -Command "[Environment]::GetEnvironmentVariable('Path', 'User')"') do set USER_PATH=%%i

echo %USER_PATH% | findstr /C:"%KIBALI_HOME%" >nul
if %errorlevel% neq 0 (
    powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';%KIBALI_HOME%', 'User')"
    echo [SUCCESS] PATH configuré pour l'utilisateur
)

echo [SUCCESS] Commande 'kibali' installée
echo.

REM Test de l'installation
echo [KIBALI] Test de l'installation...
where kibali >nul 2>nul
if %errorlevel% equ 0 (
    echo [SUCCESS] Commande 'kibali' disponible
    kibali --help >nul 2>nul
    if %errorlevel% equ 0 (
        echo [SUCCESS] Framework Kibali opérationnel
    ) else (
        echo [WARNING] La commande fonctionne mais l'aide a échoué
    )
) else (
    echo [WARNING] Redémarrez votre terminal pour utiliser 'kibali'
)

echo.
echo 🎉 INSTALLATION TERMINÉE !
echo.
echo 📚 Pour commencer:
echo    1. Redémarrez votre terminal
echo    2. Créez votre première cellule: kibali --help
echo    3. Exemple: kibali run cells\arbre.kib
echo.
echo 📁 Fichiers installés dans: %KIBALI_HOME%
echo 🌐 Repository: https://github.com/lojol469-cmd/kibalone-langage
echo.
pause
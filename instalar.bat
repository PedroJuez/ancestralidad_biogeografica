@echo off
echo ============================================================
echo    INSTALACION - Predictor de Ancestralidad Biogeografica
echo ============================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH
    echo Descarga Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual
    pause
    exit /b 1
)

echo [2/4] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [3/4] Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: No se pudieron instalar las dependencias
    pause
    exit /b 1
)

echo [4/4] Generando datos y entrenando modelo...
python src\generate_data.py
python src\model.py

echo.
echo ============================================================
echo    INSTALACION COMPLETADA
echo ============================================================
echo.
echo Para ejecutar la aplicacion:
echo    1. Abre una terminal en esta carpeta
echo    2. Ejecuta: venv\Scripts\activate
echo    3. Ejecuta: streamlit run src\app.py
echo.
echo O simplemente ejecuta: ejecutar_app.bat
echo.
pause

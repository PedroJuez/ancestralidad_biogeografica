@echo off
echo Iniciando Predictor de Ancestralidad Biogeografica...
echo.

REM Verificar que existe el entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta primero: instalar.bat
    pause
    exit /b 1
)

REM Activar entorno virtual
call venv\Scripts\activate.bat

REM Verificar que existe el modelo
if not exist "models\bga_predictor.pkl" (
    echo Modelo no encontrado. Generando datos y entrenando...
    python src\generate_data.py
    python src\model.py
)

REM Ejecutar Streamlit
echo.
echo Abriendo aplicacion en el navegador...
echo Para detener la aplicacion, pulsa Ctrl+C
echo.
streamlit run src\app.py

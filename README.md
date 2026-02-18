# 🧬 Predictor de Ancestralidad Biogeográfica

Sistema de inferencia de origen geográfico ancestral basado en marcadores AIMs (Ancestry Informative Markers) para aplicaciones en Biología Forense.

## Requisitos

- Python 3.10 o superior
- Visual Studio Code (recomendado)

## Instalación en Windows

### 1. Abre el proyecto en VS Code

```
Archivo → Abrir carpeta → C:\Users\Uned\Documents\ancestralidad_biogeografica
```

### 2. Abre una terminal (Ctrl + `)

### 3. Crea y activa el entorno virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 5. Genera los datos sintéticos

```bash
python src/generate_data.py
```

### 6. Entrena el modelo

```bash
python src/model.py
```

### 7. Ejecuta la aplicación web

```bash
streamlit run src/app.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## Estructura del Proyecto

```
ancestralidad_biogeografica/
│
├── venv/                      # Entorno virtual Python
├── data/                      # Datos generados
│   ├── training_data.csv      # Datos de entrenamiento (1500 muestras)
│   ├── test_data.csv          # Datos de prueba (350 muestras)
│   └── feature_names.json     # Lista de marcadores
├── models/                    
│   └── bga_predictor.pkl      # Modelo entrenado
├── src/                       
│   ├── generate_data.py       # Generador de datos sintéticos
│   ├── model.py               # Entrenamiento del modelo ML
│   └── app.py                 # Aplicación Streamlit
├── bga_predictor_app.html     # Aplicación web standalone (sin instalación)
├── requirements.txt           # Dependencias Python
└── README.md                  # Este archivo
```

---

## Uso Rápido (sin instalación)

Si solo quieres ver la demostración sin instalar nada:

1. Abre `bga_predictor_app.html` directamente en tu navegador
2. Funciona offline, sin necesidad de Python

---

## Descripción del Sistema

### Poblaciones de Referencia

| Código | Población | Regiones |
|--------|-----------|----------|
| AFR | Africana | África Subsahariana |
| EUR | Europea | Europa Occidental, Oriental, Mediterránea |
| EAS | Este Asiática | China, Japón, Corea, Mongolia |
| SAS | Sur Asiática | India, Pakistán, Bangladesh |
| AMR | Americana (mezclada) | Latinoamérica |

### Rendimiento del Modelo

- **Algoritmo:** Random Forest (200 árboles)
- **Precisión (CV):** ~96.6%
- **Marcadores:** 30 AIMs

### Marcadores más informativos

1. **rs16891982** (SLC45A2) - Pigmentación
2. **rs2814778** (DARC/Duffy) - Alta diferenciación AFR
3. **rs1426654** (SLC24A5) - Pigmentación
4. **rs3827760** (EDAR) - Morfología cabello (EAS)
5. **rs12913832** (HERC2) - Color de ojos (EUR)

---

## Uso Programático

```python
from src.model import BGAPredictor

# Cargar modelo entrenado
predictor = BGAPredictor.load()

# Predecir una muestra
genotypes = {
    "rs2814778": 2,   # Genotipo 0=AA, 1=Aa, 2=aa
    "rs1426654": 0,
    "rs16891982": 0,
    # ... resto de marcadores
}

result = predictor.predict(genotypes)
print(f"Predicción: {result['prediction_name']}")
print(f"Confianza: {result['confidence']:.1%}")
```

---

## Solución de Problemas

### Error: "No se encontraron los datos de entrenamiento"

Ejecuta primero:
```bash
python src/generate_data.py
```

### Error: "No se pudo cargar el modelo"

Ejecuta:
```bash
python src/model.py
```

### Streamlit no abre el navegador

Abre manualmente: `http://localhost:8501`

### Error de permisos en Windows

Ejecuta VS Code como administrador o usa PowerShell.

---

## Referencias

- 1000 Genomes Project Consortium (2015). A global reference for human genetic variation. Nature, 526(7571), 68-74.
- Kidd et al. (2014). Progress toward an efficient panel of SNPs for ancestry inference. FSI: Genetics.
- Phillips et al. (2007). Inferring ancestral origin using ancestry-informative marker SNPs. FSI: Genetics.

---

## Consideraciones Éticas

⚠️ **Este sistema es para fines educativos.**

- No identifica individuos, solo estima probabilidades
- Ancestralidad genética ≠ etnia o nacionalidad
- Los datos utilizados son sintéticos
- Uso en contextos legales requiere validación adicional

---

**Desarrollado para el curso de IA en Biología Forense**

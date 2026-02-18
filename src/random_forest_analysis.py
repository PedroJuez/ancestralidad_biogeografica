"""
Análisis Completo del Predictor de Ancestralidad Biogeográfica
==============================================================

Este script contiene:
1. Código completo del Random Forest
2. Entrenamiento paso a paso
3. Cálculo de SHAP values para interpretabilidad
4. Visualizaciones

Autor: Curso IA en Biología Forense
Fecha: Enero 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json

# Scikit-learn
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score
)

# SHAP para interpretabilidad
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("AVISO: Instala shap con 'pip install shap' para análisis de Shapley values")

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Crear directorios
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Semilla para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# FRECUENCIAS ALÉLICAS DE AIMs (DATOS REALES DEL 1000 GENOMES)
# =============================================================================

AIMS_FREQUENCIES = {
    # Marcadores con alta diferenciación continental
    # Formato: rsID -> {población: frecuencia_alelo_menor}
    
    # Marcadores principales (alta informativeness)
    "rs2814778":  {"AFR": 0.99, "EUR": 0.01, "EAS": 0.00, "SAS": 0.02, "AMR": 0.25},  # DARC (Duffy)
    "rs1426654":  {"AFR": 0.05, "EUR": 0.98, "EAS": 0.03, "SAS": 0.85, "AMR": 0.55},  # SLC24A5
    "rs16891982": {"AFR": 0.02, "EUR": 0.96, "EAS": 0.01, "SAS": 0.08, "AMR": 0.45},  # SLC45A2
    "rs1800407":  {"AFR": 0.01, "EUR": 0.08, "EAS": 0.00, "SAS": 0.02, "AMR": 0.04},  # OCA2
    "rs12913832": {"AFR": 0.02, "EUR": 0.78, "EAS": 0.01, "SAS": 0.05, "AMR": 0.35},  # HERC2
    "rs3827760":  {"AFR": 0.01, "EUR": 0.02, "EAS": 0.85, "SAS": 0.05, "AMR": 0.40},  # EDAR
    "rs1042602":  {"AFR": 0.25, "EUR": 0.38, "EAS": 0.01, "SAS": 0.30, "AMR": 0.25},  # TYR
    "rs1545397":  {"AFR": 0.85, "EUR": 0.12, "EAS": 0.05, "SAS": 0.15, "AMR": 0.35},  # OCA2
    "rs2031526":  {"AFR": 0.15, "EUR": 0.10, "EAS": 0.75, "SAS": 0.20, "AMR": 0.30},  # Asiático
    "rs7657799":  {"AFR": 0.70, "EUR": 0.15, "EAS": 0.10, "SAS": 0.18, "AMR": 0.30},  # Africano
    
    # AIMs del panel Kidd
    "rs10497191": {"AFR": 0.88, "EUR": 0.22, "EAS": 0.15, "SAS": 0.25, "AMR": 0.40},
    "rs7554936":  {"AFR": 0.08, "EUR": 0.55, "EAS": 0.90, "SAS": 0.45, "AMR": 0.50},
    "rs1871534":  {"AFR": 0.95, "EUR": 0.35, "EAS": 0.20, "SAS": 0.40, "AMR": 0.55},
    "rs17034666": {"AFR": 0.12, "EUR": 0.40, "EAS": 0.88, "SAS": 0.35, "AMR": 0.45},
    "rs6548616":  {"AFR": 0.78, "EUR": 0.20, "EAS": 0.12, "SAS": 0.25, "AMR": 0.38},
    
    # AIMs para diferenciación fina
    "rs310644":   {"AFR": 0.65, "EUR": 0.85, "EAS": 0.30, "SAS": 0.55, "AMR": 0.60},
    "rs9845457":  {"AFR": 0.20, "EUR": 0.70, "EAS": 0.45, "SAS": 0.80, "AMR": 0.55},
    "rs2593595":  {"AFR": 0.55, "EUR": 0.15, "EAS": 0.72, "SAS": 0.25, "AMR": 0.40},
    "rs1572018":  {"AFR": 0.82, "EUR": 0.45, "EAS": 0.18, "SAS": 0.50, "AMR": 0.52},
    "rs881929":   {"AFR": 0.30, "EUR": 0.60, "EAS": 0.85, "SAS": 0.55, "AMR": 0.55},
    
    # Más AIMs del 1000 Genomes
    "rs1408799":  {"AFR": 0.05, "EUR": 0.75, "EAS": 0.92, "SAS": 0.60, "AMR": 0.58},
    "rs1834640":  {"AFR": 0.90, "EUR": 0.30, "EAS": 0.25, "SAS": 0.35, "AMR": 0.48},
    "rs2065160":  {"AFR": 0.72, "EUR": 0.18, "EAS": 0.08, "SAS": 0.22, "AMR": 0.35},
    "rs4471745":  {"AFR": 0.15, "EUR": 0.55, "EAS": 0.80, "SAS": 0.48, "AMR": 0.50},
    "rs7226659":  {"AFR": 0.85, "EUR": 0.25, "EAS": 0.15, "SAS": 0.30, "AMR": 0.42},
    
    # Panel extendido
    "rs1079597":  {"AFR": 0.02, "EUR": 0.18, "EAS": 0.45, "SAS": 0.35, "AMR": 0.25},
    "rs2066807":  {"AFR": 0.68, "EUR": 0.12, "EAS": 0.05, "SAS": 0.15, "AMR": 0.28},
    "rs3737576":  {"AFR": 0.22, "EUR": 0.72, "EAS": 0.88, "SAS": 0.65, "AMR": 0.60},
    "rs4891825":  {"AFR": 0.75, "EUR": 0.28, "EAS": 0.12, "SAS": 0.32, "AMR": 0.40},
    "rs6451722":  {"AFR": 0.18, "EUR": 0.62, "EAS": 0.78, "SAS": 0.55, "AMR": 0.52},
}

POPULATION_NAMES = {
    "AFR": "Africana",
    "EUR": "Europea", 
    "EAS": "Este Asiática",
    "SAS": "Sur Asiática",
    "AMR": "Americana (mezclada)"
}

# =============================================================================
# FUNCIONES DE GENERACIÓN DE DATOS
# =============================================================================

def generate_genotype(freq: float) -> int:
    """
    Genera un genotipo (0, 1, 2) basado en frecuencia alélica.
    Asume equilibrio Hardy-Weinberg (HWE).
    
    Parámetros:
    -----------
    freq : float
        Frecuencia del alelo menor (p)
    
    Retorna:
    --------
    int : Genotipo codificado
        0 = homocigoto referencia (AA)
        1 = heterocigoto (Aa)
        2 = homocigoto alternativo (aa)
    
    Fórmula HWE:
    - P(AA) = q² donde q = 1-p
    - P(Aa) = 2pq
    - P(aa) = p²
    """
    p = freq      # frecuencia alelo menor
    q = 1 - p     # frecuencia alelo mayor
    
    # Probabilidades bajo Hardy-Weinberg
    prob_AA = q ** 2        # homocigoto referencia
    prob_Aa = 2 * p * q     # heterocigoto
    prob_aa = p ** 2        # homocigoto alternativo
    
    probs = [prob_AA, prob_Aa, prob_aa]
    
    return np.random.choice([0, 1, 2], p=probs)


def generate_individual(population: str) -> dict:
    """Genera genotipos para un individuo de una población específica."""
    genotypes = {}
    for marker, freqs in AIMS_FREQUENCIES.items():
        freq = freqs[population]
        genotypes[marker] = generate_genotype(freq)
    return genotypes


def generate_dataset(n_per_population: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Genera dataset sintético de genotipos.
    
    Parámetros:
    -----------
    n_per_population : int
        Número de individuos a generar por población
    seed : int
        Semilla para reproducibilidad
    
    Retorna:
    --------
    pd.DataFrame : Dataset con columnas [sample_id, population, rs..., rs..., ...]
    """
    np.random.seed(seed)
    
    populations = ["AFR", "EUR", "EAS", "SAS", "AMR"]
    data = []
    
    for pop in populations:
        for i in range(n_per_population):
            individual = generate_individual(pop)
            individual["population"] = pop
            individual["sample_id"] = f"{pop}_{i+1:04d}"
            data.append(individual)
    
    df = pd.DataFrame(data)
    
    # Reordenar columnas: sample_id, population, marcadores...
    marker_cols = list(AIMS_FREQUENCIES.keys())
    df = df[["sample_id", "population"] + marker_cols]
    
    return df


# =============================================================================
# MODELO RANDOM FOREST - CÓDIGO COMPLETO
# =============================================================================

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 15,
    min_samples_split: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Entrena un clasificador Random Forest.
    
    Parámetros del modelo:
    ----------------------
    n_estimators : int = 200
        Número de árboles en el bosque. Más árboles = mejor generalización
        pero mayor tiempo de cómputo.
    
    max_depth : int = 15
        Profundidad máxima de cada árbol. Controla overfitting.
        None = nodos se expanden hasta que todas las hojas son puras.
    
    min_samples_split : int = 5
        Número mínimo de muestras requeridas para dividir un nodo interno.
        Valores más altos previenen overfitting.
    
    class_weight : str = 'balanced'
        Ajusta los pesos inversamente proporcionales a las frecuencias de clase.
        Importante para datasets desbalanceados.
    
    random_state : int = 42
        Semilla para reproducibilidad.
    
    n_jobs : int = -1
        Usar todos los núcleos disponibles para paralelización.
    
    Retorna:
    --------
    RandomForestClassifier : Modelo entrenado
    """
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight='balanced',  # Ajuste para clases desbalanceadas
        random_state=random_state,
        n_jobs=-1,  # Usar todos los núcleos
        
        # Parámetros adicionales
        criterion='gini',           # Criterio de división (gini o entropy)
        min_samples_leaf=1,         # Mínimo de muestras en hoja
        max_features='sqrt',        # Número de features a considerar en cada split
        bootstrap=True,             # Usar bootstrap samples
        oob_score=True,             # Calcular out-of-bag score
        verbose=0
    )
    
    # Entrenar
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    feature_names: list
) -> dict:
    """
    Evalúa el modelo y genera métricas completas.
    
    Retorna:
    --------
    dict : Diccionario con métricas y predicciones
    """
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Reporte de clasificación
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # OOB Score (Out-of-Bag)
    oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
    
    return {
        'accuracy': accuracy,
        'oob_score': oob_score,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'feature_importance': importance_df,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'class_names': class_names
    }


# =============================================================================
# ANÁLISIS SHAP (Shapley Additive Explanations)
# =============================================================================

def calculate_shap_values(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list,
    class_names: list
) -> dict:
    """
    Calcula los SHAP values para interpretabilidad del modelo.
    
    SHAP (SHapley Additive exPlanations) se basa en la teoría de juegos
    cooperativos para asignar a cada feature su contribución a la predicción.
    
    Parámetros:
    -----------
    model : RandomForestClassifier
        Modelo entrenado
    X_train : np.ndarray
        Datos de entrenamiento (para el background)
    X_test : np.ndarray
        Datos de test para explicar
    feature_names : list
        Nombres de los marcadores
    class_names : list
        Nombres de las clases
    
    Retorna:
    --------
    dict : SHAP values y explainer
    """
    if not SHAP_AVAILABLE:
        print("ERROR: SHAP no está instalado. Ejecuta: pip install shap")
        return None
    
    print("\nCalculando SHAP values...")
    print("(Esto puede tardar unos segundos)")
    
    # Crear explainer con TreeExplainer (optimizado para árboles)
    # Usamos una muestra del training para el background
    background_size = min(100, len(X_train))
    background = X_train[np.random.choice(len(X_train), background_size, replace=False)]
    
    explainer = shap.TreeExplainer(model, data=background)
    
    # Calcular SHAP values para el test set
    shap_values = explainer.shap_values(X_test)
    
    # Para Random Forest multiclase, shap_values es una lista de arrays
    # Cada array corresponde a una clase
    
    # Calcular importancia media absoluta por feature (global)
    # Promediamos sobre todas las clases
    if isinstance(shap_values, list):
        # Multiclase: shap_values es lista de [n_samples, n_features] por clase
        shap_importance = np.zeros(len(feature_names))
        for class_shap in shap_values:
            shap_importance += np.abs(class_shap).mean(axis=0)
        shap_importance /= len(shap_values)
    else:
        # Si es array 3D [n_samples, n_features, n_classes]
        if len(shap_values.shape) == 3:
            shap_importance = np.abs(shap_values).mean(axis=(0, 2))
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Asegurar que shap_importance es 1D
    shap_importance = np.array(shap_importance).flatten()
    
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    return {
        'explainer': explainer,
        'shap_values': shap_values,
        'shap_importance': shap_importance_df,
        'expected_value': explainer.expected_value,
        'feature_names': feature_names,
        'class_names': class_names
    }


def plot_shap_summary(shap_results: dict, X_test: np.ndarray, output_path: Path = None):
    """
    Genera gráficos SHAP de resumen.
    """
    if not SHAP_AVAILABLE or shap_results is None:
        return
    
    feature_names = shap_results['feature_names']
    shap_values = shap_results['shap_values']
    class_names = shap_results['class_names']
    
    # Determinar el formato de shap_values
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
    elif len(shap_values.shape) == 3:
        n_classes = shap_values.shape[2]
        # Convertir a lista para consistencia
        shap_values = [shap_values[:, :, i] for i in range(n_classes)]
    else:
        print("Formato de SHAP values no reconocido")
        return
    
    # Crear figura con múltiples subplots
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig.suptitle('Análisis SHAP por Población', fontsize=16, fontweight='bold')
    
    # Plot SHAP summary para cada clase
    for idx, class_name in enumerate(class_names):
        if idx >= n_rows * n_cols:
            break
        ax = axes.flatten()[idx]
        
        if idx < len(shap_values):
            # Obtener SHAP values para esta clase
            class_shap = shap_values[idx]
            mean_abs_shap = np.abs(class_shap).mean(axis=0)
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:15]  # Top 15
            
            # Bar plot horizontal
            colors = plt.cm.tab10(idx)
            ax.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx], color=colors)
            ax.set_yticks(range(len(sorted_idx)))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title(f'{class_name} ({POPULATION_NAMES.get(class_name, class_name)})')
    
    # Ocultar subplots vacíos
    for idx in range(len(class_names), n_rows * n_cols):
        axes.flatten()[idx].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'shap_summary_by_class.png', dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {output_path / 'shap_summary_by_class.png'}")
    
    plt.close()  # Cerrar para no mostrar en consola


def plot_shap_global_importance(shap_results: dict, rf_importance: pd.DataFrame, output_path: Path = None):
    """
    Compara importancia de features: Random Forest vs SHAP.
    """
    if not SHAP_AVAILABLE or shap_results is None:
        return
    
    shap_imp = shap_results['shap_importance'].copy()
    rf_imp = rf_importance.copy()
    
    # Merge
    comparison = pd.merge(
        rf_imp.rename(columns={'importance': 'RF_importance'}),
        shap_imp.rename(columns={'shap_importance': 'SHAP_importance'}),
        on='feature'
    )
    
    # Normalizar para comparar
    comparison['RF_normalized'] = comparison['RF_importance'] / comparison['RF_importance'].sum()
    comparison['SHAP_normalized'] = comparison['SHAP_importance'] / comparison['SHAP_importance'].sum()
    
    # Ordenar por SHAP importance
    comparison = comparison.sort_values('SHAP_importance', ascending=True).tail(15)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(comparison))
    width = 0.35
    
    bars1 = ax.barh(y_pos - width/2, comparison['RF_normalized'], width, 
                    label='Random Forest (Gini)', color='#3498db', alpha=0.8)
    bars2 = ax.barh(y_pos + width/2, comparison['SHAP_normalized'], width,
                    label='SHAP', color='#e74c3c', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comparison['feature'])
    ax.set_xlabel('Importancia Normalizada')
    ax.set_title('Comparación de Importancia: Random Forest vs SHAP\n(Top 15 marcadores)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path / 'importance_comparison_rf_vs_shap.png', dpi=150, bbox_inches='tight')
        print(f"Gráfico guardado: {output_path / 'importance_comparison_rf_vs_shap.png'}")
    
    plt.close()  # Cerrar para no mostrar en consola
    
    return comparison


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecuta el pipeline completo:
    1. Genera datos sintéticos
    2. Entrena Random Forest
    3. Evalúa el modelo
    4. Calcula SHAP values
    5. Genera visualizaciones
    """
    
    print("=" * 70)
    print("PREDICTOR DE ANCESTRALIDAD BIOGEOGRÁFICA - ANÁLISIS COMPLETO")
    print("=" * 70)
    
    # =========================================================================
    # PASO 1: GENERAR DATOS
    # =========================================================================
    print("\n[1/5] GENERANDO DATOS SINTÉTICOS")
    print("-" * 50)
    
    df = generate_dataset(n_per_population=300, seed=RANDOM_STATE)
    print(f"Dataset generado: {len(df)} muestras")
    print(f"Marcadores: {len(AIMS_FREQUENCIES)}")
    print(f"Poblaciones: {df['population'].unique().tolist()}")
    
    # Guardar datos
    df.to_csv(DATA_DIR / "training_data.csv", index=False)
    
    # =========================================================================
    # PASO 2: PREPARAR DATOS PARA ML
    # =========================================================================
    print("\n[2/5] PREPARANDO DATOS PARA ENTRENAMIENTO")
    print("-" * 50)
    
    # Features y labels
    feature_names = list(AIMS_FREQUENCIES.keys())
    X = df[feature_names].values
    y = df['population'].values
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Shape de X: {X.shape}")
    print(f"Clases: {label_encoder.classes_.tolist()}")
    
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        stratify=y_encoded,
        random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_train)} muestras")
    print(f"Test: {len(X_test)} muestras")
    
    # =========================================================================
    # PASO 3: ENTRENAR RANDOM FOREST
    # =========================================================================
    print("\n[3/5] ENTRENANDO RANDOM FOREST")
    print("-" * 50)
    
    # Validación cruzada primero
    print("\nValidación cruzada (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    )
    
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"CV Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Entrenar modelo final
    print("\nEntrenando modelo final...")
    rf_model = train_random_forest(X_train, y_train)
    
    print(f"OOB Score: {rf_model.oob_score_:.4f}")
    print(f"Número de árboles: {rf_model.n_estimators}")
    
    # =========================================================================
    # PASO 4: EVALUAR MODELO
    # =========================================================================
    print("\n[4/5] EVALUANDO MODELO")
    print("-" * 50)
    
    results = evaluate_model(rf_model, X_test, y_test, label_encoder, feature_names)
    
    print(f"\nAccuracy en Test: {results['accuracy']:.4f}")
    
    print("\nReporte de Clasificación:")
    print("-" * 50)
    report_df = pd.DataFrame(results['classification_report']).T
    print(report_df.round(4).to_string())
    
    print("\nMatriz de Confusión:")
    print("-" * 50)
    conf_df = pd.DataFrame(
        results['confusion_matrix'],
        index=results['class_names'],
        columns=results['class_names']
    )
    print(conf_df.to_string())
    
    print("\nTop 10 Features por Importancia (Gini):")
    print("-" * 50)
    print(results['feature_importance'].head(10).to_string(index=False))
    
    # =========================================================================
    # PASO 5: ANÁLISIS SHAP
    # =========================================================================
    print("\n[5/5] ANÁLISIS SHAP (Shapley Values)")
    print("-" * 50)
    
    if SHAP_AVAILABLE:
        shap_results = calculate_shap_values(
            rf_model, X_train, X_test, feature_names, label_encoder.classes_.tolist()
        )
        
        if shap_results:
            print("\nTop 10 Features por SHAP Importance:")
            print("-" * 50)
            print(shap_results['shap_importance'].head(10).to_string(index=False))
            
            # Generar gráficos
            print("\nGenerando visualizaciones...")
            plot_shap_summary(shap_results, X_test, OUTPUT_DIR)
            comparison = plot_shap_global_importance(
                shap_results, results['feature_importance'], OUTPUT_DIR
            )
            
            # Guardar resultados SHAP
            shap_results['shap_importance'].to_csv(
                OUTPUT_DIR / 'shap_importance.csv', index=False
            )
            
            if comparison is not None:
                comparison.to_csv(OUTPUT_DIR / 'importance_comparison.csv', index=False)
    else:
        print("SHAP no disponible. Instalar con: pip install shap")
        shap_results = None
    
    # =========================================================================
    # GUARDAR MODELO Y RESULTADOS
    # =========================================================================
    print("\n" + "=" * 70)
    print("GUARDANDO RESULTADOS")
    print("=" * 70)
    
    # Guardar modelo
    model_data = {
        'model': rf_model,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'cv_scores': cv_scores,
        'test_accuracy': results['accuracy'],
        'feature_importance': results['feature_importance']
    }
    
    with open(MODELS_DIR / 'random_forest_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Modelo guardado: {MODELS_DIR / 'random_forest_model.pkl'}")
    
    # Guardar feature importance
    results['feature_importance'].to_csv(
        OUTPUT_DIR / 'feature_importance_gini.csv', index=False
    )
    print(f"Importancia guardada: {OUTPUT_DIR / 'feature_importance_gini.csv'}")
    
    # Guardar métricas
    metrics = {
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'test_accuracy': float(results['accuracy']),
        'oob_score': float(rf_model.oob_score_),
        'n_estimators': rf_model.n_estimators,
        'max_depth': rf_model.max_depth,
        'n_features': len(feature_names),
        'n_classes': len(label_encoder.classes_)
    }
    
    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Métricas guardadas: {OUTPUT_DIR / 'metrics.json'}")
    
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETADO")
    print("=" * 70)
    
    return rf_model, results, shap_results


if __name__ == "__main__":
    model, results, shap_results = main()

"""
Generador de datos sintéticos para predicción de ancestralidad biogeográfica.

Basado en frecuencias alélicas reales de AIMs (Ancestry Informative Markers)
publicados en estudios del 1000 Genomes Project.

Referencias:
- Kidd et al. (2014) - 55 AIMs panel
- Phillips et al. (2007) - SNPforID 34-plex
- Seldin et al. (2006) - 128 AIMs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Obtener directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Crear directorio data si no existe
DATA_DIR.mkdir(exist_ok=True)

# Frecuencias alélicas de AIMs seleccionados (alelo menor)
# Basadas en datos publicados del 1000 Genomes Project Phase 3
# Poblaciones: AFR (Africana), EUR (Europea), EAS (Este Asiático), 
#              SAS (Sur Asiático), AMR (Americana mezclada)

AIMS_FREQUENCIES: Dict[str, Dict[str, float]] = {
    # Marcadores con alta diferenciación continental
    "rs2814778": {"AFR": 0.99, "EUR": 0.01, "EAS": 0.00, "SAS": 0.02, "AMR": 0.25},  # DARC (Duffy)
    "rs1426654": {"AFR": 0.05, "EUR": 0.98, "EAS": 0.03, "SAS": 0.85, "AMR": 0.55},  # SLC24A5
    "rs16891982": {"AFR": 0.02, "EUR": 0.96, "EAS": 0.01, "SAS": 0.08, "AMR": 0.45},  # SLC45A2
    "rs1800407": {"AFR": 0.01, "EUR": 0.08, "EAS": 0.00, "SAS": 0.02, "AMR": 0.04},  # OCA2
    "rs12913832": {"AFR": 0.02, "EUR": 0.78, "EAS": 0.01, "SAS": 0.05, "AMR": 0.35},  # HERC2
    "rs3827760": {"AFR": 0.01, "EUR": 0.02, "EAS": 0.85, "SAS": 0.05, "AMR": 0.40},  # EDAR
    "rs1042602": {"AFR": 0.25, "EUR": 0.38, "EAS": 0.01, "SAS": 0.30, "AMR": 0.25},  # TYR
    "rs1545397": {"AFR": 0.85, "EUR": 0.12, "EAS": 0.05, "SAS": 0.15, "AMR": 0.35},  # OCA2
    "rs2031526": {"AFR": 0.15, "EUR": 0.10, "EAS": 0.75, "SAS": 0.20, "AMR": 0.30},  # Asiático
    "rs7657799": {"AFR": 0.70, "EUR": 0.15, "EAS": 0.10, "SAS": 0.18, "AMR": 0.30},  # Africano
    
    # AIMs adicionales del panel Kidd
    "rs10497191": {"AFR": 0.88, "EUR": 0.22, "EAS": 0.15, "SAS": 0.25, "AMR": 0.40},
    "rs7554936": {"AFR": 0.08, "EUR": 0.55, "EAS": 0.90, "SAS": 0.45, "AMR": 0.50},
    "rs1871534": {"AFR": 0.95, "EUR": 0.35, "EAS": 0.20, "SAS": 0.40, "AMR": 0.55},
    "rs17034666": {"AFR": 0.12, "EUR": 0.40, "EAS": 0.88, "SAS": 0.35, "AMR": 0.45},
    "rs6548616": {"AFR": 0.78, "EUR": 0.20, "EAS": 0.12, "SAS": 0.25, "AMR": 0.38},
    
    # AIMs para diferenciación fina
    "rs310644": {"AFR": 0.65, "EUR": 0.85, "EAS": 0.30, "SAS": 0.55, "AMR": 0.60},
    "rs9845457": {"AFR": 0.20, "EUR": 0.70, "EAS": 0.45, "SAS": 0.80, "AMR": 0.55},
    "rs2593595": {"AFR": 0.55, "EUR": 0.15, "EAS": 0.72, "SAS": 0.25, "AMR": 0.40},
    "rs1572018": {"AFR": 0.82, "EUR": 0.45, "EAS": 0.18, "SAS": 0.50, "AMR": 0.52},
    "rs881929": {"AFR": 0.30, "EUR": 0.60, "EAS": 0.85, "SAS": 0.55, "AMR": 0.55},
    
    # Más AIMs del 1000 Genomes
    "rs1408799": {"AFR": 0.05, "EUR": 0.75, "EAS": 0.92, "SAS": 0.60, "AMR": 0.58},
    "rs1834640": {"AFR": 0.90, "EUR": 0.30, "EAS": 0.25, "SAS": 0.35, "AMR": 0.48},
    "rs2065160": {"AFR": 0.72, "EUR": 0.18, "EAS": 0.08, "SAS": 0.22, "AMR": 0.35},
    "rs4471745": {"AFR": 0.15, "EUR": 0.55, "EAS": 0.80, "SAS": 0.48, "AMR": 0.50},
    "rs7226659": {"AFR": 0.85, "EUR": 0.25, "EAS": 0.15, "SAS": 0.30, "AMR": 0.42},
    
    # Panel extendido
    "rs1079597": {"AFR": 0.02, "EUR": 0.18, "EAS": 0.45, "SAS": 0.35, "AMR": 0.25},
    "rs2066807": {"AFR": 0.68, "EUR": 0.12, "EAS": 0.05, "SAS": 0.15, "AMR": 0.28},
    "rs3737576": {"AFR": 0.22, "EUR": 0.72, "EAS": 0.88, "SAS": 0.65, "AMR": 0.60},
    "rs4891825": {"AFR": 0.75, "EUR": 0.28, "EAS": 0.12, "SAS": 0.32, "AMR": 0.40},
    "rs6451722": {"AFR": 0.18, "EUR": 0.62, "EAS": 0.78, "SAS": 0.55, "AMR": 0.52},
}


def generate_genotype(freq: float) -> int:
    """
    Genera un genotipo (0, 1, 2) basado en frecuencia alélica.
    Asume equilibrio Hardy-Weinberg.
    
    0 = homocigoto referencia (AA)
    1 = heterocigoto (Aa)
    2 = homocigoto alternativo (aa)
    """
    p = freq  # frecuencia alelo menor
    q = 1 - p  # frecuencia alelo mayor
    
    # Probabilidades HWE: q², 2pq, p²
    probs = [q**2, 2*p*q, p**2]
    return np.random.choice([0, 1, 2], p=probs)


def generate_individual(population: str) -> Dict[str, int]:
    """Genera genotipos para un individuo de una población."""
    genotypes = {}
    for marker, freqs in AIMS_FREQUENCIES.items():
        freq = freqs[population]
        genotypes[marker] = generate_genotype(freq)
    return genotypes


def generate_dataset(
    n_per_population: int = 200,
    populations: List[str] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera dataset sintético de genotipos.
    
    Args:
        n_per_population: Número de individuos por población
        populations: Lista de poblaciones a incluir
        seed: Semilla para reproducibilidad
    
    Returns:
        DataFrame con genotipos y etiquetas de población
    """
    np.random.seed(seed)
    
    if populations is None:
        populations = ["AFR", "EUR", "EAS", "SAS", "AMR"]
    
    data = []
    for pop in populations:
        for i in range(n_per_population):
            individual = generate_individual(pop)
            individual["population"] = pop
            individual["sample_id"] = f"{pop}_{i+1:04d}"
            data.append(individual)
    
    df = pd.DataFrame(data)
    
    # Reordenar columnas
    cols = ["sample_id", "population"] + list(AIMS_FREQUENCIES.keys())
    df = df[cols]
    
    return df


def generate_admixed_individual(
    proportions: Dict[str, float]
) -> Dict[str, int]:
    """
    Genera un individuo con mezcla de ancestralidades.
    
    Args:
        proportions: Dict con proporciones de cada ancestralidad
                    Ej: {"EUR": 0.5, "AFR": 0.3, "AMR": 0.2}
    """
    genotypes = {}
    for marker, freqs in AIMS_FREQUENCIES.items():
        # Frecuencia ponderada por proporciones de ancestralidad
        weighted_freq = sum(
            freqs[pop] * prop 
            for pop, prop in proportions.items()
        )
        genotypes[marker] = generate_genotype(weighted_freq)
    return genotypes


def add_admixed_samples(
    df: pd.DataFrame,
    n_admixed: int = 100,
    seed: int = 123
) -> pd.DataFrame:
    """Añade muestras mezcladas al dataset."""
    np.random.seed(seed)
    
    admixed_data = []
    populations = ["AFR", "EUR", "EAS", "SAS", "AMR"]
    
    for i in range(n_admixed):
        # Generar proporciones aleatorias (Dirichlet)
        props = np.random.dirichlet(np.ones(5) * 0.5)
        proportions = dict(zip(populations, props))
        
        individual = generate_admixed_individual(proportions)
        
        # Asignar a la ancestralidad predominante
        dominant_pop = max(proportions, key=proportions.get)
        individual["population"] = f"ADM_{dominant_pop}"
        individual["sample_id"] = f"ADM_{i+1:04d}"
        
        # Guardar proporciones reales para evaluación
        for pop in populations:
            individual[f"prop_{pop}"] = proportions[pop]
        
        admixed_data.append(individual)
    
    admixed_df = pd.DataFrame(admixed_data)
    
    # Combinar con dataset original
    return pd.concat([df, admixed_df], ignore_index=True)


if __name__ == "__main__":
    # Generar dataset principal
    print("=" * 60)
    print("GENERADOR DE DATOS SINTÉTICOS - ANCESTRALIDAD BIOGEOGRÁFICA")
    print("=" * 60)
    
    print("\nGenerando dataset de entrenamiento...")
    df = generate_dataset(n_per_population=300, seed=42)
    print(f"  - {len(df)} muestras generadas")
    print(f"  - {len(AIMS_FREQUENCIES)} marcadores AIMs")
    print(f"  - Poblaciones: {df['population'].unique().tolist()}")
    
    # Guardar
    training_path = DATA_DIR / "training_data.csv"
    df.to_csv(training_path, index=False)
    print(f"  - Guardado en: {training_path}")
    
    # Generar dataset de test con muestras mezcladas
    print("\nGenerando dataset de prueba (incluye muestras mezcladas)...")
    df_test = generate_dataset(n_per_population=50, seed=999)
    df_test = add_admixed_samples(df_test, n_admixed=100, seed=123)
    print(f"  - {len(df_test)} muestras de prueba")
    
    test_path = DATA_DIR / "test_data.csv"
    df_test.to_csv(test_path, index=False)
    print(f"  - Guardado en: {test_path}")
    
    # Resumen
    print("\n" + "=" * 60)
    print("DATASET GENERADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nDistribución de poblaciones (entrenamiento):")
    print(df["population"].value_counts().to_string())
    print(f"\nArchivos creados en: {DATA_DIR}")

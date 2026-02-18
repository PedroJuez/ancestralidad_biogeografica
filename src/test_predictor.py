"""
Script de prueba - Verifica que el sistema funciona correctamente.

Ejecutar con: python src/test_predictor.py
"""

import sys
from pathlib import Path

# Añadir el directorio src al path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from src.model import BGAPredictor

def main():
    print("=" * 60)
    print("TEST DEL PREDICTOR DE ANCESTRALIDAD BIOGEOGRÁFICA")
    print("=" * 60)
    
    # Cargar modelo
    print("\n[1] Cargando modelo entrenado...")
    try:
        predictor = BGAPredictor.load()
        print(f"    ✓ Modelo cargado: {predictor.best_model_name}")
        print(f"    ✓ Marcadores: {len(predictor.feature_names)}")
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        print("\n    Ejecuta primero:")
        print("    python src/generate_data.py")
        print("    python src/model.py")
        return
    
    # Cargar muestras de ejemplo
    print("\n[2] Cargando muestras de ejemplo...")
    ejemplo_path = BASE_DIR / "data" / "muestras_ejemplo.csv"
    
    if not ejemplo_path.exists():
        print(f"    ✗ No encontrado: {ejemplo_path}")
        return
    
    df = pd.read_csv(ejemplo_path)
    print(f"    ✓ {len(df)} muestras cargadas")
    
    # Predecir cada muestra
    print("\n[3] Realizando predicciones...")
    print("-" * 60)
    print(f"{'Muestra':<25} {'Real':<12} {'Predicho':<12} {'Confianza':<10}")
    print("-" * 60)
    
    aciertos = 0
    total_conocidos = 0
    
    for _, row in df.iterrows():
        genotypes = {col: row[col] for col in predictor.feature_names if col in row}
        result = predictor.predict(genotypes)
        
        real = row['population']
        pred = result['prediction']
        conf = result['confidence']
        
        # Verificar si es conocido
        if real != "DESCONOCIDO":
            total_conocidos += 1
            if real == pred:
                aciertos += 1
                status = "✓"
            else:
                status = "✗"
        else:
            status = "?"
        
        print(f"{row['sample_id']:<25} {real:<12} {pred:<12} {conf:>6.1%}  {status}")
    
    print("-" * 60)
    
    if total_conocidos > 0:
        accuracy = aciertos / total_conocidos
        print(f"\nPrecisión en muestras conocidas: {aciertos}/{total_conocidos} = {accuracy:.1%}")
    
    # Test de predicción individual
    print("\n[4] Test de predicción individual...")
    print("-" * 60)
    
    # Muestra típica africana
    muestra_afr = {
        "rs2814778": 2, "rs1426654": 0, "rs16891982": 0, "rs12913832": 0,
        "rs3827760": 0, "rs1545397": 2, "rs7657799": 2, "rs10497191": 2
    }
    
    result = predictor.predict(muestra_afr)
    print(f"\nMuestra con genotipos típicos africanos:")
    print(f"  Predicción: {result['prediction_name']}")
    print(f"  Confianza: {result['confidence']:.1%}")
    print(f"\n  Probabilidades:")
    for pop, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {pop}: {prob:>6.1%} {bar}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()

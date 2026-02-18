"""
Ejemplo de interpretación de muestras con mezcla ancestral
==========================================================

Este script muestra cómo interpretar resultados cuando
una persona tiene ancestralidad de múltiples orígenes.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.model import BGAPredictor

def mostrar_resultado(nombre, descripcion, genotipos):
    """Muestra la predicción de forma visual"""
    resultado = predictor.predict(genotipos)
    
    print(f"\n{'='*60}")
    print(f"CASO: {nombre}")
    print(f"{'='*60}")
    print(f"Descripción: {descripcion}")
    print(f"\nPredicción principal: {resultado['prediction_name']}")
    print(f"Confianza: {resultado['confidence']:.1%}")
    print(f"\nDistribución de probabilidades:")
    print(f"{'-'*60}")
    
    # Ordenar por probabilidad
    probs_ordenadas = sorted(resultado['probabilities'].items(), key=lambda x: -x[1])
    
    nombres_poblacion = {
        "AFR": "Africana",
        "EUR": "Europea",
        "EAS": "Este Asiática",
        "SAS": "Sur Asiática",
        "AMR": "Americana mezclada"
    }
    
    for pop, prob in probs_ordenadas:
        barra = "█" * int(prob * 40)
        nombre_completo = nombres_poblacion.get(pop, pop)
        print(f"  {pop} ({nombre_completo:18}): {prob:>6.1%} {barra}")
    
    # Interpretación
    print(f"\n📊 INTERPRETACIÓN:")
    print(f"{'-'*60}")
    
    # Detectar si hay mezcla
    prob_max = probs_ordenadas[0][1]
    prob_segundo = probs_ordenadas[1][1]
    
    if prob_max > 0.90:
        print(f"  → Ancestralidad {probs_ordenadas[0][0]} con ALTA confianza")
        print(f"  → Patrón consistente con origen {nombres_poblacion[probs_ordenadas[0][0]].lower()}")
    elif prob_max > 0.70:
        print(f"  → Ancestralidad predominantemente {probs_ordenadas[0][0]}")
        print(f"  → Posible mezcla menor con {probs_ordenadas[1][0]}")
    elif prob_max > 0.50:
        print(f"  → POSIBLE MEZCLA ANCESTRAL detectada")
        print(f"  → Componente principal: {probs_ordenadas[0][0]} ({prob_max:.1%})")
        print(f"  → Componente secundario: {probs_ordenadas[1][0]} ({prob_segundo:.1%})")
    else:
        print(f"  → MEZCLA ANCESTRAL COMPLEJA")
        print(f"  → Múltiples componentes detectados:")
        for pop, prob in probs_ordenadas[:3]:
            if prob > 0.15:
                print(f"     • {pop}: {prob:.1%}")
    
    return resultado


# Cargar modelo
print("Cargando modelo...")
predictor = BGAPredictor.load()
print("✓ Modelo cargado\n")


# ============================================================
# CASO 1: Persona de ancestralidad africana "pura"
# ============================================================
caso1_africano = {
    "rs2814778": 2,   # Alelo Duffy-null (casi exclusivo AFR)
    "rs1426654": 0,   # Sin alelo europeo de pigmentación
    "rs16891982": 0,  # Sin alelo europeo
    "rs3827760": 0,   # Sin alelo asiático
    "rs12913832": 0,  # Sin alelo ojos claros
    "rs1545397": 2,   # Alelo africano
    "rs7657799": 2,   # Alelo africano
    "rs10497191": 2,  # Alelo africano
    "rs1871534": 2,   # Alelo africano
    "rs1834640": 2,   # Alelo africano
}

mostrar_resultado(
    "Persona con ancestralidad africana",
    "Genotipos típicos de poblaciones del África subsahariana",
    caso1_africano
)


# ============================================================
# CASO 2: Persona de ancestralidad europea "pura"
# ============================================================
caso2_europeo = {
    "rs2814778": 0,   # Sin alelo africano
    "rs1426654": 2,   # Alelo europeo de pigmentación clara
    "rs16891982": 2,  # Alelo europeo (SLC45A2)
    "rs3827760": 0,   # Sin alelo asiático
    "rs12913832": 2,  # Alelo ojos claros (muy europeo)
    "rs1545397": 0,   # Sin alelo africano
    "rs7657799": 0,   # Sin alelo africano
    "rs1408799": 2,   # Común en EUR/EAS
    "rs310644": 2,    # Frecuente en europeos
    "rs9845457": 2,   # Frecuente en europeos
}

mostrar_resultado(
    "Persona con ancestralidad europea",
    "Genotipos típicos de poblaciones europeas",
    caso2_europeo
)


# ============================================================
# CASO 3: Persona AFROAMERICANA (mezcla AFR + EUR)
# ============================================================
# Típicamente ~80% AFR, ~20% EUR
caso3_afroamericano = {
    "rs2814778": 2,   # Alelo africano (heredado de ancestros AFR)
    "rs1426654": 1,   # HETEROCIGOTO - tiene un alelo EUR
    "rs16891982": 1,  # HETEROCIGOTO - mezcla
    "rs3827760": 0,   # Sin alelo asiático
    "rs12913832": 1,  # HETEROCIGOTO - posible mezcla
    "rs1545397": 1,   # Heterocigoto
    "rs7657799": 2,   # Alelo africano
    "rs10497191": 2,  # Alelo africano
    "rs1871534": 1,   # Heterocigoto
    "rs1834640": 1,   # Heterocigoto
}

mostrar_resultado(
    "Persona afroamericana (mezcla AFR + EUR)",
    "Patrón típico de afroamericanos: ~80% africano, ~20% europeo",
    caso3_afroamericano
)


# ============================================================
# CASO 4: Persona LATINOAMERICANA (mezcla compleja)
# ============================================================
# Típicamente: ~50% EUR, ~40% Nativo Americano, ~10% AFR
# Los nativos americanos comparten marcadores con EAS (migración por Bering)
caso4_latino = {
    "rs2814778": 1,   # Heterocigoto (algo de AFR)
    "rs1426654": 1,   # Heterocigoto (algo de EUR)
    "rs16891982": 1,  # Heterocigoto
    "rs3827760": 1,   # Heterocigoto (algo de componente EAS/Nativo)
    "rs12913832": 1,  # Heterocigoto
    "rs1545397": 1,   # Heterocigoto
    "rs7657799": 1,   # Heterocigoto
    "rs2031526": 1,   # Heterocigoto
    "rs17034666": 1,  # Heterocigoto
    "rs881929": 1,    # Heterocigoto
}

mostrar_resultado(
    "Persona latinoamericana (mezcla compleja)",
    "Patrón de poblaciones mestizas: europeo + nativo americano + africano",
    caso4_latino
)


# ============================================================
# CASO 5: Hijo de padre europeo y madre asiática
# ============================================================
caso5_euroasiatico = {
    "rs2814778": 0,   # Ni AFR
    "rs1426654": 1,   # Un alelo EUR (del padre)
    "rs16891982": 1,  # Un alelo EUR (del padre)
    "rs3827760": 1,   # Un alelo EAS (de la madre)
    "rs12913832": 1,  # Heterocigoto
    "rs1408799": 2,   # Común en ambos
    "rs2031526": 1,   # Un alelo EAS
    "rs17034666": 1,  # Un alelo EAS
    "rs7554936": 1,   # Heterocigoto
    "rs881929": 1,    # Heterocigoto
}

mostrar_resultado(
    "Hijo de padre europeo y madre asiática",
    "Primera generación de mezcla EUR + EAS (50%-50%)",
    caso5_euroasiatico
)


# ============================================================
# RESUMEN FINAL
# ============================================================
print(f"\n\n{'='*60}")
print("GUÍA DE INTERPRETACIÓN")
print(f"{'='*60}")
print("""
📌 CONFIANZA > 90%: 
   Ancestralidad clara de una población
   
📌 CONFIANZA 70-90%: 
   Ancestralidad predominante con posible mezcla menor
   
📌 CONFIANZA 50-70%: 
   Mezcla ancestral probable
   Mirar el segundo componente
   
📌 CONFIANZA < 50%: 
   Mezcla compleja de múltiples orígenes
   Típico de poblaciones americanas mezcladas

⚠️  IMPORTANTE:
   - AMR (Americana) ya es una categoría mezclada
   - Los heterocigotos (1) suelen indicar mezcla
   - Muchos heterocigotos = probable mezcla reciente
   - Este modelo es educativo, no para uso forense real
""")

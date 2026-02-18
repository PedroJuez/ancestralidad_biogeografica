"""
Interfaz Web para el Predictor de Ancestralidad Biogeográfica.

Aplicación interactiva basada en Streamlit para demostrar
el funcionamiento del predictor de BGA.

Ejecutar con: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import sklearn
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.svm
import sklearn.linear_model
import sklearn.naive_bayes
from pathlib import Path

# Obtener directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Configuración de página
st.set_page_config(
    page_title="Predictor de Ancestralidad Biogeográfica",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Estilos CSS personalizados - Tema claro mejorado
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E86AB, #A23B72, #F18F01);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.15rem;
        color: #5a6c7d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar mejorado */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 1px solid #dee2e6;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #2E86AB;
    }
    
    /* Métricas con colores */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E86AB;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Botones bonitos */
    .stButton > button {
        background: linear-gradient(90deg, #2E86AB 0%, #1a5276 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(46, 134, 171, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 134, 171, 0.4);
    }
    
    /* Info box */
    .stAlert {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border: none;
        border-left: 4px solid #2E86AB;
        border-radius: 8px;
    }
    
    /* Tablas */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Inputs numéricos */
    .stNumberInput > div > div > input {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        transition: border-color 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #2E86AB;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        color: #2E86AB;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dee2e6, transparent);
        margin: 2rem 0;
    }
    
    /* Cards para poblaciones */
    .population-card {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid;
        background: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E86AB, #A23B72);
    }
    
    /* Subheaders */
    h2, h3 {
        color: #1a5276;
    }
</style>
""", unsafe_allow_html=True)


# Información de poblaciones
POPULATION_INFO = {
    "AFR": {
        "name": "Africana",
        "color": "#E74C3C",
        "description": "Poblaciones del continente africano",
        "regions": "África Subsahariana, África del Norte"
    },
    "EUR": {
        "name": "Europea", 
        "color": "#3498DB",
        "description": "Poblaciones del continente europeo",
        "regions": "Europa Occidental, Oriental, Mediterránea"
    },
    "EAS": {
        "name": "Este Asiática",
        "color": "#2ECC71",
        "description": "Poblaciones del este de Asia",
        "regions": "China, Japón, Corea, Mongolia"
    },
    "SAS": {
        "name": "Sur Asiática",
        "color": "#F39C12",
        "description": "Poblaciones del sur de Asia",
        "regions": "India, Pakistán, Bangladesh, Sri Lanka"
    },
    "AMR": {
        "name": "Americana (Mezclada)",
        "color": "#9B59B6",
        "description": "Poblaciones americanas con mezcla",
        "regions": "Latinoamérica, poblaciones nativas mezcladas"
    }
}

# Frecuencias alélicas para simulación
AIMS_FREQUENCIES = {
    "rs2814778": {"AFR": 0.99, "EUR": 0.01, "EAS": 0.00, "SAS": 0.02, "AMR": 0.25},
    "rs1426654": {"AFR": 0.05, "EUR": 0.98, "EAS": 0.03, "SAS": 0.85, "AMR": 0.55},
    "rs16891982": {"AFR": 0.02, "EUR": 0.96, "EAS": 0.01, "SAS": 0.08, "AMR": 0.45},
    "rs12913832": {"AFR": 0.02, "EUR": 0.78, "EAS": 0.01, "SAS": 0.05, "AMR": 0.35},
    "rs3827760": {"AFR": 0.01, "EUR": 0.02, "EAS": 0.85, "SAS": 0.05, "AMR": 0.40},
    "rs1545397": {"AFR": 0.85, "EUR": 0.12, "EAS": 0.05, "SAS": 0.15, "AMR": 0.35},
    "rs2031526": {"AFR": 0.15, "EUR": 0.10, "EAS": 0.75, "SAS": 0.20, "AMR": 0.30},
    "rs7657799": {"AFR": 0.70, "EUR": 0.15, "EAS": 0.10, "SAS": 0.18, "AMR": 0.30},
    "rs10497191": {"AFR": 0.88, "EUR": 0.22, "EAS": 0.15, "SAS": 0.25, "AMR": 0.40},
    "rs7554936": {"AFR": 0.08, "EUR": 0.55, "EAS": 0.90, "SAS": 0.45, "AMR": 0.50},
}


@st.cache_resource
def load_model():
    """Carga el modelo entrenado."""
    model_path = MODELS_DIR / "bga_predictor.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        return data
    return None


def generate_random_genotype(population: str) -> dict:
    """Genera un genotipo aleatorio para una población."""
    genotypes = {}
    for marker, freqs in AIMS_FREQUENCIES.items():
        p = freqs[population]
        q = 1 - p
        probs = [q**2, 2*p*q, p**2]
        genotypes[marker] = np.random.choice([0, 1, 2], p=probs)
    return genotypes


def predict_ancestry(model_data: dict, genotypes: dict) -> dict:
    """Realiza predicción de ancestralidad."""
    model = model_data['models']['Random Forest']
    feature_names = model_data['feature_names']
    label_encoder = model_data['label_encoder']
    
    # Construir vector de features (usar 0 para marcadores no disponibles)
    X = np.array([[genotypes.get(f, 0) for f in feature_names]])
    
    # Predecir
    pred_idx = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X)[0]
    
    prob_dict = {
        label_encoder.inverse_transform([i])[0]: float(p)
        for i, p in enumerate(probs)
    }
    
    return {
        "prediction": pred_label,
        "confidence": float(max(probs)),
        "probabilities": prob_dict
    }


def plot_probability_bars(probs: dict) -> go.Figure:
    """Crea gráfico de barras de probabilidades."""
    populations = list(probs.keys())
    values = list(probs.values())
    colors = [POPULATION_INFO[p]["color"] for p in populations]
    names = [POPULATION_INFO[p]["name"] for p in populations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=names,
            orientation='h',
            marker_color=colors,
            text=[f'{v:.1%}' for v in values],
            textposition='inside',
            textfont=dict(color='white', size=14)
        )
    ])
    
    fig.update_layout(
        title="Probabilidades de Ancestralidad",
        xaxis_title="Probabilidad",
        yaxis_title="",
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def plot_pie_chart(probs: dict) -> go.Figure:
    """Crea gráfico circular de probabilidades."""
    populations = list(probs.keys())
    values = list(probs.values())
    colors = [POPULATION_INFO[p]["color"] for p in populations]
    names = [POPULATION_INFO[p]["name"] for p in populations]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=names,
            values=values,
            marker_colors=colors,
            hole=0.4,
            textinfo='percent+label',
            textfont=dict(size=12)
        )
    ])
    
    fig.update_layout(
        title="Composición Ancestral",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Predictor de Ancestralidad Biogeográfica</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema de inferencia de origen geográfico ancestral basado en marcadores AIMs</p>', 
                unsafe_allow_html=True)
    
    # Cargar modelo
    model_data = load_model()
    
    if model_data is None:
        st.error(f"""
        ⚠️ No se pudo cargar el modelo. 
        
        Asegúrate de haber ejecutado primero:
        1. `python src/generate_data.py`
        2. `python src/model.py`
        
        Ruta esperada: {MODELS_DIR / 'bga_predictor.pkl'}
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        st.subheader("Modo de entrada")
        input_mode = st.radio(
            "Selecciona cómo introducir los datos:",
            ["🎲 Generar muestra aleatoria", "✏️ Introducir genotipos manualmente", "📁 Analizar archivo CSV"]
        )
        
        if "🎲" in input_mode:
            st.subheader("Simulación")
            sim_population = st.selectbox(
                "Población a simular:",
                options=list(POPULATION_INFO.keys()),
                format_func=lambda x: f"{x} - {POPULATION_INFO[x]['name']}"
            )
            
            if st.button("🔄 Generar nueva muestra", use_container_width=True):
                st.session_state.genotypes = generate_random_genotype(sim_population)
                st.session_state.true_population = sim_population
        
        st.divider()
        
        st.subheader("ℹ️ Acerca de")
        st.markdown("""
        Este sistema utiliza **30 marcadores AIMs** 
        (Ancestry Informative Markers) para predecir 
        el origen geográfico ancestral de una muestra de ADN.
        
        **Modelo:** Random Forest  
        **Precisión:** ~96.6%  
        **Poblaciones:** 5 grupos continentales
        """)
    
    # Inicializar genotipos si no existen
    if 'genotypes' not in st.session_state:
        st.session_state.genotypes = generate_random_genotype("EUR")
        st.session_state.true_population = "EUR"
    
    # Contenido principal
    if "✏️" in input_mode:
        st.subheader("Introducción manual de genotipos")
        st.info("💡 Introduce el genotipo (0, 1, 2) para cada marcador. 0=AA, 1=Aa, 2=aa")
        
        cols = st.columns(5)
        manual_genotypes = {}
        
        for i, (marker, _) in enumerate(AIMS_FREQUENCIES.items()):
            with cols[i % 5]:
                manual_genotypes[marker] = st.number_input(
                    marker,
                    min_value=0,
                    max_value=2,
                    value=st.session_state.genotypes.get(marker, 0),
                    key=f"manual_{marker}"
                )
        
        st.session_state.genotypes = manual_genotypes
        st.session_state.true_population = None  # En modo manual no hay población real
    
    # Modo CSV - Análisis de múltiples muestras
    elif "📁" in input_mode:
        st.subheader("📁 Análisis de archivo CSV")
        
        # Cargar archivo por defecto o subido
        csv_option = st.radio(
            "Selecciona origen de datos:",
            ["Usar muestras_ejemplo.csv", "Subir mi propio archivo"],
            horizontal=True
        )
        
        df_samples = None
        
        if csv_option == "Usar muestras_ejemplo.csv":
            csv_path = DATA_DIR / "muestras_ejemplo.csv"
            if csv_path.exists():
                df_samples = pd.read_csv(csv_path)
                st.success(f"✓ Cargadas {len(df_samples)} muestras de muestras_ejemplo.csv")
            else:
                st.error("No se encontró el archivo muestras_ejemplo.csv en la carpeta data/")
        else:
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
            if uploaded_file is not None:
                df_samples = pd.read_csv(uploaded_file)
                st.success(f"✓ Cargadas {len(df_samples)} muestras")
        
        if df_samples is not None:
            st.markdown("---")
            st.subheader("🔬 Resultados del Análisis")
            
            # Obtener nombres de marcadores del modelo
            feature_names = model_data['feature_names']
            
            # Procesar cada muestra
            results_list = []
            
            for idx, row in df_samples.iterrows():
                sample_id = row.get('sample_id', f'Muestra_{idx+1}')
                real_pop = row.get('population', 'DESCONOCIDO')
                
                # Extraer genotipos
                genotypes = {marker: row[marker] for marker in feature_names if marker in row}
                
                # Predecir
                result = predict_ancestry(model_data, genotypes)
                
                # Verificación
                if real_pop != 'DESCONOCIDO':
                    verificacion = "✅" if real_pop == result['prediction'] else "❌"
                else:
                    verificacion = "❓"
                
                results_list.append({
                    'Muestra': sample_id,
                    'Población Real': real_pop,
                    'Predicción': result['prediction'],
                    'Ancestralidad': POPULATION_INFO[result['prediction']]['name'],
                    'Confianza': f"{result['confidence']:.1%}",
                    'Verificación': verificacion
                })
            
            # Mostrar tabla de resultados
            results_df = pd.DataFrame(results_list)
            
            # Colorear según verificación
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Confianza': st.column_config.TextColumn('Confianza'),
                    'Verificación': st.column_config.TextColumn('Verificación')
                }
            )
            
            # Estadísticas
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            total = len(results_df)
            conocidos = results_df[results_df['Población Real'] != 'DESCONOCIDO']
            desconocidos = results_df[results_df['Población Real'] == 'DESCONOCIDO']
            
            if len(conocidos) > 0:
                aciertos = len(conocidos[conocidos['Verificación'] == '✅'])
                precision = aciertos / len(conocidos)
                
                with col1:
                    st.metric("Total muestras", total)
                with col2:
                    st.metric("Muestras verificables", len(conocidos))
                with col3:
                    st.metric("Precisión", f"{precision:.1%}", delta=f"{aciertos}/{len(conocidos)} correctas")
            else:
                with col1:
                    st.metric("Total muestras", total)
                with col2:
                    st.metric("Muestras desconocidas", len(desconocidos))
                with col3:
                    st.metric("Verificación", "N/A", delta="Sin datos reales")
            
            # Mostrar detalle de una muestra seleccionada
            st.markdown("---")
            st.subheader("🔍 Detalle de muestra")
            
            selected_sample = st.selectbox(
                "Selecciona una muestra para ver detalle:",
                options=df_samples['sample_id'].tolist() if 'sample_id' in df_samples.columns else [f'Muestra_{i+1}' for i in range(len(df_samples))]
            )
            
            # Encontrar la muestra seleccionada
            if 'sample_id' in df_samples.columns:
                sample_row = df_samples[df_samples['sample_id'] == selected_sample].iloc[0]
            else:
                sample_idx = int(selected_sample.split('_')[1]) - 1
                sample_row = df_samples.iloc[sample_idx]
            
            # Predecir para mostrar gráficos
            genotypes = {marker: sample_row[marker] for marker in feature_names if marker in sample_row}
            result = predict_ancestry(model_data, genotypes)
            
            # Mostrar gráficos
            chart_cols = st.columns(2)
            with chart_cols[0]:
                fig_bars = plot_probability_bars(result['probabilities'])
                st.plotly_chart(fig_bars, use_container_width=True)
            with chart_cols[1]:
                fig_pie = plot_pie_chart(result['probabilities'])
                st.plotly_chart(fig_pie, use_container_width=True)
        
        return  # Salir de main() para modo CSV
    
    # Realizar predicción
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Genotipos de la muestra")
        
        genotype_df = pd.DataFrame([
            {"Marcador": k, "Genotipo": v, "Interpretación": ["AA", "Aa", "aa"][v]}
            for k, v in st.session_state.genotypes.items()
        ])
        
        st.dataframe(
            genotype_df,
            use_container_width=True,
            hide_index=True,
            height=350
        )
    
    with col2:
        # Predecir
        result = predict_ancestry(model_data, st.session_state.genotypes)
        
        st.subheader("🎯 Resultado de la Predicción")
        
        # Métricas principales
        pred_pop = result['prediction']
        pred_info = POPULATION_INFO[pred_pop]
        
        metric_cols = st.columns(3)
        
        with metric_cols[0]:
            st.metric(
                "Ancestralidad Predicha",
                pred_info['name'],
                delta=f"Código: {pred_pop}"
            )
        
        with metric_cols[1]:
            st.metric(
                "Confianza",
                f"{result['confidence']:.1%}",
                delta="Alta" if result['confidence'] > 0.8 else "Media"
            )
        
        with metric_cols[2]:
            true_pop = getattr(st.session_state, 'true_population', None)
            if true_pop is not None:
                is_correct = true_pop == pred_pop
                st.metric(
                    "Verificación",
                    "✅ Correcto" if is_correct else "❌ Incorrecto",
                    delta=f"Real: {true_pop}"
                )
            else:
                st.metric(
                    "Verificación",
                    "❓ N/A",
                    delta="Entrada manual"
                )
    
    # Gráficos
    st.divider()
    
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        fig_bars = plot_probability_bars(result['probabilities'])
        st.plotly_chart(fig_bars, use_container_width=True)
    
    with chart_cols[1]:
        fig_pie = plot_pie_chart(result['probabilities'])
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Información adicional
    with st.expander("📚 Información sobre los marcadores AIMs utilizados"):
        st.markdown("""
        | Marcador | Gen/Región | Función conocida |
        |----------|------------|------------------|
        | rs2814778 | DARC (Duffy) | Receptor de quimiocinas, alta diferenciación AFR |
        | rs1426654 | SLC24A5 | Pigmentación de piel, alta frecuencia EUR |
        | rs16891982 | SLC45A2 | Pigmentación, diferencia EUR/AFR |
        | rs3827760 | EDAR | Morfología del cabello, alta frecuencia EAS |
        | rs12913832 | HERC2 | Color de ojos, alta frecuencia EUR |
        
        Los AIMs son variantes genéticas (SNPs) que muestran grandes diferencias 
        de frecuencia entre poblaciones de diferentes orígenes geográficos.
        """)
    
    with st.expander("⚖️ Consideraciones éticas y limitaciones"):
        st.warning("""
        **Limitaciones importantes:**
        
        1. **No identifica individuos**: Este sistema estima probabilidades de ancestralidad, 
           no identifica personas específicas.
        
        2. **Ancestralidad ≠ Etnia/Nacionalidad**: La ancestralidad genética no equivale 
           a identidad cultural, étnica o nacional.
        
        3. **Poblaciones mezcladas**: Las predicciones son menos precisas en individuos 
           con ancestralidad mixta reciente.
        
        4. **Uso responsable**: Esta herramienta es para fines educativos y de investigación. 
           Su uso en contextos legales requiere validación adicional y consideraciones éticas.
        
        5. **Datos sintéticos**: Este demo utiliza datos generados basados en frecuencias 
           alélicas publicadas, no datos reales de pacientes.
        """)


if __name__ == "__main__":
    main()

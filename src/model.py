"""
Modelo de Machine Learning para Predicción de Ancestralidad Biogeográfica.

Implementa múltiples clasificadores y selecciona el mejor basándose en
validación cruzada.

Algoritmos:
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Naive Bayes (bayesiano, como en genética forense)
- Logistic Regression
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Obtener directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Crear directorios si no existen
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class BGAPredictor:
    """
    Predictor de Ancestralidad Biogeográfica.
    
    Entrena múltiples modelos y permite predecir el origen
    geográfico ancestral de muestras genéticas.
    """
    
    POPULATION_NAMES = {
        "AFR": "Africana",
        "EUR": "Europea", 
        "EAS": "Este Asiática",
        "SAS": "Sur Asiática",
        "AMR": "Americana (mezclada)"
    }
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.best_model_name: str = ""
        self.training_results: Dict = {}
        
    def _get_features_and_labels(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extrae features (genotipos) y labels (poblaciones)."""
        # Identificar columnas de marcadores (excluir metadata)
        exclude_cols = ['sample_id', 'population'] + \
                      [c for c in df.columns if c.startswith('prop_')]
        
        self.feature_names = [c for c in df.columns if c not in exclude_cols]
        
        X = df[self.feature_names].values
        
        # Filtrar solo poblaciones puras para entrenamiento
        pure_mask = ~df['population'].str.startswith('ADM_')
        y = df.loc[pure_mask, 'population'].values
        X = X[pure_mask]
        
        return X, y
    
    def train(
        self, 
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict:
        """
        Entrena múltiples modelos y evalúa su rendimiento.
        
        Args:
            df: DataFrame con genotipos y etiquetas
            test_size: Proporción para test
            cv_folds: Número de folds para validación cruzada
            
        Returns:
            Diccionario con resultados de entrenamiento
        """
        print("=" * 60)
        print("ENTRENAMIENTO DEL PREDICTOR DE ANCESTRALIDAD BIOGEOGRÁFICA")
        print("=" * 60)
        
        # Preparar datos
        X, y = self._get_features_and_labels(df)
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nDatos de entrenamiento:")
        print(f"  - Muestras: {len(X)}")
        print(f"  - Marcadores AIMs: {len(self.feature_names)}")
        print(f"  - Poblaciones: {list(self.label_encoder.classes_)}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            stratify=y_encoded,
            random_state=42
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Definir modelos
        models_config = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "SVM (RBF)": SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                class_weight='balanced',
                random_state=42
            )
        }
        
        # Entrenar y evaluar cada modelo
        results = {}
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        print(f"\n{'Modelo':<25} {'CV Accuracy':<15} {'Test Accuracy':<15}")
        print("-" * 55)
        
        best_score = 0
        
        for name, model in models_config.items():
            # Usar datos escalados para SVM y Logistic Regression
            if name in ["SVM (RBF)", "Logistic Regression"]:
                X_tr, X_te = X_train_scaled, X_test_scaled
            else:
                X_tr, X_te = X_train, X_test
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='accuracy')
            
            # Entrenar modelo final
            model.fit(X_tr, y_train)
            
            # Evaluar en test
            y_pred = model.predict(X_te)
            test_acc = accuracy_score(y_test, y_pred)
            
            # Guardar resultados
            results[name] = {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "test_accuracy": test_acc,
                "model": model
            }
            
            self.models[name] = model
            
            print(f"{name:<25} {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  {test_acc:.4f}")
            
            # Actualizar mejor modelo
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                self.best_model_name = name
        
        print("-" * 55)
        print(f"\n✓ Mejor modelo: {self.best_model_name} (CV: {best_score:.4f})")
        
        # Reporte detallado del mejor modelo
        best_model = self.models[self.best_model_name]
        if self.best_model_name in ["SVM (RBF)", "Logistic Regression"]:
            y_pred_best = best_model.predict(X_test_scaled)
        else:
            y_pred_best = best_model.predict(X_test)
            
        print(f"\n{'=' * 60}")
        print("REPORTE DE CLASIFICACIÓN (Mejor Modelo)")
        print("=" * 60)
        print(classification_report(
            y_test, y_pred_best,
            target_names=self.label_encoder.classes_
        ))
        
        # Feature importance (si está disponible)
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            
            print(f"\nTop 10 marcadores más informativos:")
            for i, idx in enumerate(indices):
                print(f"  {i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
        
        self.training_results = results
        return results
    
    def predict(
        self, 
        genotypes: Dict[str, int],
        model_name: str = None
    ) -> Dict:
        """
        Predice la ancestralidad de una muestra.
        
        Args:
            genotypes: Dict {marcador: genotipo (0,1,2)}
            model_name: Nombre del modelo a usar (default: mejor)
            
        Returns:
            Dict con predicción y probabilidades
        """
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        
        # Construir vector de features
        X = np.array([[genotypes.get(f, 0) for f in self.feature_names]])
        
        # Escalar si es necesario
        if model_name in ["SVM (RBF)", "Logistic Regression"]:
            X = self.scaler.transform(X)
        
        # Predecir
        pred_idx = model.predict(X)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Probabilidades
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            prob_dict = {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            }
        else:
            prob_dict = {pred_label: 1.0}
        
        return {
            "prediction": pred_label,
            "prediction_name": self.POPULATION_NAMES.get(pred_label, pred_label),
            "confidence": float(max(prob_dict.values())),
            "probabilities": prob_dict,
            "model_used": model_name
        }
    
    def predict_batch(
        self, 
        df: pd.DataFrame,
        model_name: str = None
    ) -> pd.DataFrame:
        """Predice ancestralidad para múltiples muestras."""
        results = []
        
        for _, row in df.iterrows():
            genotypes = {f: row[f] for f in self.feature_names if f in row}
            pred = self.predict(genotypes, model_name)
            results.append({
                "sample_id": row.get("sample_id", "unknown"),
                "true_population": row.get("population", "unknown"),
                "predicted": pred["prediction"],
                "confidence": pred["confidence"],
                **{f"prob_{k}": v for k, v in pred["probabilities"].items()}
            })
        
        return pd.DataFrame(results)
    
    def save(self, path: str = None):
        """Guarda el predictor entrenado."""
        if path is None:
            path = MODELS_DIR / "bga_predictor.pkl"
        else:
            path = Path(path)
            
        save_data = {
            "models": self.models,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "best_model_name": self.best_model_name
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"\nModelo guardado en: {path}")
    
    @classmethod
    def load(cls, path: str = None) -> 'BGAPredictor':
        """Carga un predictor guardado."""
        if path is None:
            path = MODELS_DIR / "bga_predictor.pkl"
        else:
            path = Path(path)
            
        predictor = cls()
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        predictor.models = save_data["models"]
        predictor.label_encoder = save_data["label_encoder"]
        predictor.scaler = save_data["scaler"]
        predictor.feature_names = save_data["feature_names"]
        predictor.best_model_name = save_data["best_model_name"]
        
        return predictor


def main():
    """Entrena y evalúa el predictor."""
    # Cargar datos
    train_path = DATA_DIR / "training_data.csv"
    test_path = DATA_DIR / "test_data.csv"
    
    if not train_path.exists():
        print("ERROR: No se encontraron los datos de entrenamiento.")
        print(f"Ejecuta primero: python src/generate_data.py")
        print(f"Ruta esperada: {train_path}")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Crear y entrenar predictor
    predictor = BGAPredictor()
    predictor.train(train_df)
    
    # Guardar modelo
    predictor.save()
    
    # Guardar feature names para la interfaz
    feature_names_path = DATA_DIR / "feature_names.json"
    with open(feature_names_path, 'w') as f:
        json.dump(predictor.feature_names, f)
    print(f"Feature names guardados en: {feature_names_path}")
    
    # Evaluar en datos de test (incluyendo mezcladas)
    print(f"\n{'=' * 60}")
    print("EVALUACIÓN EN DATOS DE PRUEBA")
    print("=" * 60)
    
    # Solo muestras puras
    pure_test = test_df[~test_df['population'].str.startswith('ADM_')]
    predictions = predictor.predict_batch(pure_test)
    
    accuracy = (predictions['true_population'] == predictions['predicted']).mean()
    print(f"\nAccuracy en muestras puras: {accuracy:.4f}")
    
    # Ejemplo de predicción individual
    print(f"\n{'=' * 60}")
    print("EJEMPLO DE PREDICCIÓN INDIVIDUAL")
    print("=" * 60)
    
    # Tomar una muestra aleatoria
    sample = test_df.iloc[0]
    genotypes = {f: sample[f] for f in predictor.feature_names}
    
    result = predictor.predict(genotypes)
    
    print(f"\nMuestra: {sample['sample_id']}")
    print(f"Población real: {sample['population']}")
    print(f"\nPredicción: {result['prediction']} ({result['prediction_name']})")
    print(f"Confianza: {result['confidence']:.2%}")
    print(f"\nProbabilidades por población:")
    for pop, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {pop}: {prob:.2%} {bar}")


if __name__ == "__main__":
    main()

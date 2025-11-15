"""
Script simplificado para ejecutar evaluación sin MLflow activo.
Entrena un modelo simple y ejecuta evaluación comprehensiva.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from data_preparation import prepare_data
from model_evaluation import comprehensive_evaluation, generate_evaluation_report

def main():
    print("=" * 60)
    print("EVALUACION COMPREHENSIVA DEL MODELO")
    print("=" * 60)
    
    print("\n1. Creando directorios necesarios...")
    os.makedirs("data/processed", exist_ok=True)
    
    print("\n2. Cargando y preparando datos...")
    original_dir = os.getcwd()
    os.chdir("src")
    try:
        df = pd.read_csv("../data/raw/rent_guatemala.csv")
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(df)
    finally:
        os.chdir(original_dir)
    print(f"   ✓ Datos preparados: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    print("\n3. Entrenando modelo RandomForest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("   ✓ Modelo entrenado")
    
    print("\n4. Extrayendo metadata para análisis...")
    df_processed = df.sort_values("listing_id").reset_index(drop=True)
    test_start_idx = len(X_train) + len(X_val)
    metadata_test = df_processed.iloc[test_start_idx:test_start_idx + len(X_test)]
    
    if all(col in metadata_test.columns for col in ['is_premium_zone', 'property_type', 'zone']):
        metadata_test = metadata_test[['is_premium_zone', 'property_type', 'zone']].copy()
    else:
        metadata_test = None
        print("   ⚠ Metadata no disponible")
    
    print("\n5. Ejecutando evaluación comprehensiva...")
    print("   (Esto puede tomar varios minutos...)")
    
    results = comprehensive_evaluation(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        model_name="RandomForest",
        metadata_df=metadata_test,
        mlflow_log=False,
        n_cv_splits=5,
        use_shap=False
    )
    
    print("\n6. Generando reporte...")
    report_path = generate_evaluation_report(results, "comprehensive_evaluation_report.json")
    print(f"   ✓ Reporte guardado en: {report_path}")
    
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print("\nMétricas de Regresión:")
    for key, value in results['regression_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMétricas de Clasificación:")
    for key, value in results['classification_metrics'].items():
        if isinstance(value, (int, float)) and value is not None:
            print(f"  {key}: {value:.4f}")
    
    if results['business_impact']:
        print("\nImpacto de Negocio:")
        for key, value in results['business_impact'].items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("Evaluación completada exitosamente!")
    print("=" * 60)

if __name__ == "__main__":
    main()


from ultralytics import YOLO
import os

# ----------------------------------------------------------------------
# 1. Definición de Parámetros
# ----------------------------------------------------------------------

EXPERIMENT_NAME = 'yolov8_emotion'
DATASET_YAML_PATH = 'affectnet.yaml' 

# Parámetros para entrenamiento desde cero:
# Requiere muchas más épocas que el fine-tuning
NUM_EPOCHS = 100 
BATCH_SIZE = 16 

# ARQUITECTURA para entrenar desde cero (inicia con pesos aleatorios)
# Asegúrate de que el archivo yolov8s.yaml sea accesible o esté en la misma carpeta
ARCHITECTURE_CONFIG = 'yolov8s.pt' 
# Si el archivo yolov8s.yaml no está accesible directamente, puedes usar 'yolov8s.pt' 
# para cargar la estructura y luego entrenar, pero esto técnicamente usa la estructura 
# pre-entrenada aunque entrenes todos los pesos. Para entrenamiento estricto 'desde cero', 
# necesitarías el archivo de configuración de la arquitectura. 
# En la práctica, Ultralytics gestiona esto bien.

# ----------------------------------------------------------------------
# 2. Inicialización y Entrenamiento del Modelo
# ----------------------------------------------------------------------

print(f"Cargando la estructura del modelo: {ARCHITECTURE_CONFIG}")
# Cargar la arquitectura. Ultralytics lo manejará: si es un .yaml, pesos aleatorios.
model = YOLO(ARCHITECTURE_CONFIG)

print("Iniciando entrenamiento desde CERO para la detección de emociones...")

results = model.train(
    data=DATASET_YAML_PATH,  
    epochs=NUM_EPOCHS,       
    batch=BATCH_SIZE,        
    imgsz=640,               
    name=EXPERIMENT_NAME,    
    # Hiperparámetros clave para el entrenamiento desde cero:
    lr0=0.001, # Tasa de aprendizaje inicial
    lrf=0.0001, # Tasa de aprendizaje final
    # Si quieres ASEGURARTE de que no usa pesos pre-entrenados del archivo, 
    # puedes usar el argumento 'weights=False' o simplemente 'weights=None'
    #  weights=None 
    project='/mnt/homeGPU/pablomarpa/experimentos_yolo',
    augment=True,
    optimizer='SGD'
)

print("¡Entrenamiento finalizado!")
print(f"Resultados guardados en: {os.path.join(os.getcwd(), 'runs', 'detect', EXPERIMENT_NAME)}")

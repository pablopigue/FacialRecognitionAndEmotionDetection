from ultralytics import YOLO
import cv2
import torch
import os

# 1. Configuración de Dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando: {device.upper()}")

# 2. Cargar el modelo de EMOCIONES
# Asegúrate de que la ruta sea la correcta para tu modelo de emociones
emotion_model_path = "./yolov8_emotion/weights/best.pt"
model = YOLO(emotion_model_path).to(device)

# 3. RUTA DE LA IMAGEN
image_path = "tu_imagen.jpg"  # <-- Cambia esto por la ruta de tu foto

if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen en {image_path}")
else:
    # 4. Leer imagen
    img = cv2.imread(image_path)
    
    # 5. Inferencia
    # El modelo buscará las clases de emociones directamente
    results = model(img, conf=0.4, device=device, verbose=False)

    # 6. Dibujar los resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Obtener clase y confianza
            cls = int(box.cls[0])
            label_name = r.names[cls]
            conf = float(box.conf[0])

            # Color llamativo para la emoción (Cyan)
            color = (255, 255, 0)

            # Dibujar rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Texto con la emoción
            text = f"{label_name.upper()} ({conf:.2f})"
            cv2.putText(img, text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # 7. Mostrar y Guardar
    cv2.imshow("Deteccion de Emociones", img)
    
    output_path = "resultado_emocion.jpg"
    cv2.imwrite(output_path, img)
    print(f"Proceso completado. Imagen guardada como: {output_path}")

    print("Presiona cualquier tecla para cerrar la ventana...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

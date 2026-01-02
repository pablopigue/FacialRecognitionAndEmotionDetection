from ultralytics import YOLO
import cv2
import torch
import os

# 1. Configuración de Dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando: {device.upper()}")

# 2. Cargar el modelo de caras
# Asegúrate de que la ruta al .pt sea correcta
model_path = "./yolov8_face/weights/best.pt"
face_model = YOLO(model_path).to(device)

# 3. RUTA DE LA IMAGEN (Cambia esto por tu archivo)
image_path = "tu_imagen.jpg" 

if not os.path.exists(image_path):
    print(f"Error: No se encontró la imagen en {image_path}")
else:
    # 4. Cargar la imagen
    img = cv2.imread(image_path)

    # 5. Realizar la detección
    # stream=False porque es solo una imagen
    results = face_model(img, conf=0.5, device=device, verbose=False)

    # 6. Dibujar resultados
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Obtener coordenadas
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Etiqueta de confianza
            label = f"Cara: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 7. Mostrar y Guardar
    cv2.imshow("Deteccion de Caras", img)
    
    # Guardar el resultado en el disco
    output_name = "resultado_deteccion.jpg"
    cv2.imwrite(output_name, img)
    print(f"Resultado guardado como: {output_name}")

    print("Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

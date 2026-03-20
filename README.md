# Project

## Environment Setup

To run the `.py` files, you must have `ultralytics` installed. There are two ways to do this:

**Option 1:**
pip install ultralytics

**Option 2: Create a Python virtual environment and install (Linux):**
```bash
python3 -m venv venv
```
source ./venv/bin/activate
pip install ultralytics

To run a script, simply use python *.py, replacing * with the name of the file you wish to execute.

For Windows, use the following commands:
python3 -m venv venv
.\venv\Scripts\Activate.ps1
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
pip install ultralytics

## File Descriptions
caras_extraidas
When running the pruebaModeloDeteccionCaras.py script on an image, the faces extracted from that image are saved in this folder.

### emociones_extraidas
When running pruebaModeloDeteccionEmociones.py on one or more faces, the results are saved here. These are images featuring a bounding box around the face and text encoding the detected emotion.

### imagenesPruebaManual
This folder stores sample images used to test the models separately and check their performance before implementing real-time functionality.

### yolo_v8 emotion
This folder stores the results of the emotion detection model: weights, precision/recall charts, confusion matrices, and visual results from various epochs.

### yolov8_face
This folder contains the face detector results: confusion matrices, learning curves, and parameter metric comparisons. It also includes images of different training batches showing detected faces and their confidence scores. The file args.yaml displays the model's training data.

### aplicacionFinal.py
The main executable program that combines both face and emotion detection. Once the environment is set up, run it using:

python3 aplicacionFinal.py

During execution, a pop-up window may appear asking for permission to access the camera.

### DeepFace.ipynb
Initially, we attempted to train and execute the project using Google Colab. However, this was not possible as we exceeded the permitted execution times and faced penalties. Furthermore, Colab does not support real-time execution. This Notebook serves as proof of our initial work, where we downloaded a model to detect emotions from photos. It highlights challenges such as connecting Google resources to local hardware.

### emotions.yaml
Contains the paths for the training, validation, and test datasets, as well as the emotion labels.

### face.yaml
Contains the paths for the training, validation, and test datasets, along with the class declarations (only one class possible).

### pruebaModeloDeteccionCaras.py
A script that loads the trained facial detection model and runs it on test images located in imagenesPruebaManual. It crops the identified face regions and saves them individually in caras_extraidas, allowing for isolated verification of detection quality.

### pruebaModeloDeteccionEmociones.py
A script designed to validate the emotion classification model. It takes cropped face images as input, predicts the corresponding emotion using the trained weights, and saves the visual result (labeled image with probability) in emociones_extraidas.

### train_face_yolo.py
The script responsible for starting the YOLOv8 model training specifically for face detection. It reads the dataset configuration from face.yaml and, upon completion, saves the resulting weights (best.pt and last.pt) and performance metrics in yolov8_face. Note: Since the datasets are not included, this file is for reference only. To run it, you must download the datasets mentioned in deteccion_de_emociones.pdf.

### train_yolo_emotion.py
The script responsible for training the emotion classification model. It uses the configuration in emotions.yaml to process the dataset and generates weight files, confusion matrices, and learning charts stored in yolo_v8 emotion. Note: Since the datasets are not included, this file is for reference only. To run it, you must download the datasets mentioned in deteccion_de_emociones.pdf.

# Proyecto

## Configuración del entorno



Para poder ejecutar los .py se debe tener ultralytics instalado, para ello dos opciones:

pip install ultralytics

Crear entorno python e instalar ultralytics para linux:
```
python3 -m venv venv
source ./venv/bin/activate
pip install ultralytics
```
Para ejecutar simplemente usar ``python \*.py.`` Siendo \* el nombre del archivo a ejecutar.



Para Windows, utilizar los siguientes comandos:

```
python3 -m venv venv

.\\venv\\Scripts\\Activate.ps1

Set-ExecutionPolicy -Scope CurrentUsr -ExecutionPolicy RemoteSigned

pip install ultralytics
```



## Descripción de archivos

### caras\_extraidas
Al ejecutar el código ``pruebaModeloDeteccionCaras.py`` sobre una imagen, se guardan las caras extraídas en dicha imagen en la carpeta <kbd>caras_extraidas</kbd>

### emociones\_extraidas
Al ejecutar el código ``pruebaModeloDeteccionEmociones`` sobre una o varias caras, se guardan los resultados en la carpeta <kbd>emociones_extraidas</kbd>. Se trata de imágenes con un rectángulo delimitador de la cara y un texto codificando la emoción.

### imagenesPruebaManual
Para comprobar cómo se va comportando el modelo entrenado antes de implementar el funcionamiento en tiempo real, se prueban los modelos por separado con imágenes ejemplo que se almacenan en esta carpeta. 

### yolo\_v8 emotion
En esta carpeta se almacenan los resultados del modelo que detecta las emociones: pesos, gráficas de precisión, recall, matrices de confusión, resultados visuales de algunas épocas...


### yolov8\_face



Esta carpeta contiene los resultados del detector de caras: las matrices de confusión, las curvas de aprendizaje y comparación de parámetros con métricas. También contiene imágenes con los diferentes batches de entrenamiento con las caras detectadas, así como su grado de confianza de que lo detectado sea una cara. "*args.yaml*" muestra los datos del entrenamiento del modelo.



### aplicacionFinal.py



Ejecutable del programa final, que combina tanto la detección de caras como la detección de emociones. Para utilizarlo, una vez configurado el entorno, se debe usar la línea


```
python3 aplicacionFinal.py
```


Durante la ejecución, puede salir una ventana emergente para conceder permisos de la cámara a la aplicación.

### DeepFace.ipynb
En un principio, intentamos realizar el entrenamiento y la ejecución del proyecto usando GoogleColab, pero esto no fue posible porque sobrepásabmos los tiempos de ejecución permitidos y obteníamos una penalización. No solo eso, sino que esta plataforma no permite ejecución en tiempo real. Prueba de ello es este Notebook, en el que descargamos un modelo y podemos detectar emociones a partir de fotografías. Se plantean retos como hacer la conexión entre los recursos de Google y el hardware de nuestros dispositivos.


### emotions.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test como las etiquetas de las emociones.



### face.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test y la declaración de las clases (solo hay una clase posible).



### pruebaModeloDeteccionCaras.py

Script que carga el modelo entrenado de detección facial y lo ejecuta sobre imágenes de prueba ubicadas en <kbd>imagenesPruebaManual</kbd>. Recorta las regiones identificadas como rostros y las guarda individualmente en la carpeta <kbd>caras_extraidas</kbd>, permitiendo verificar la calidad de la detección de forma aislada.

### pruebaModeloDeteccionEmociones.py

Script diseñado para validar el modelo de clasificación de emociones. Toma imágenes de rostros recortados como entrada, predice la emoción correspondiente utilizando los pesos entrenados y guarda el resultado visual (imagen con etiqueta y probabilidad) en la carpeta <kbd>emociones_extraidas</kbd>.

### train_face_yolo.py

Código encargado de iniciar el entrenamiento del modelo YOLOv8 específico para la detección de caras. Lee la configuración del dataset desde ``face.yaml`` y, tras completar las épocas definidas, guarda los pesos resultantes ('best.pt' y 'last.pt') y las métricas de rendimiento en la carpeta <kbd>yolov8_face</kbd>. Destacar que como no se incluyen los datasets este archivo es simplemente para visualización de como se realizó, pues realmente su ejecución no funcionará. En caso de querer probarlo se deben descargar los datasets presentados en ``deteccion_de_emociones.pdf``.

### train_yolo_emotion.py

Código responsable del entrenamiento del modelo para la clasificación de emociones. Utiliza la configuración definida en ``emotions.yaml`` para procesar el conjunto de datos y genera los archivos de pesos, matrices de confusión y gráficas de aprendizaje que se almacenan en la carpeta <kbd>yolo_v8 emotion</kbd>. Destacar que como no se incluyen los datasets este archivo es simplemente para visualización de como se realizó, pues realmente su ejecución no funcionará. En caso de querer probarlo se deben descargar los datasets presentados en ``deteccion_de_emociones.pdf``.






# Proyect

## Environment Configuration

To execute the `.py` files, `ultralytics` must be installed. There are two options for this:

`pip install ultralytics`

Create a Python environment and install ultralytics for Linux:
```bash
python3 -m venv venv
source ./venv/bin/activate
pip install ultralytics

To run, simply use python *.py, where * is the name of the file to execute.

For Windows, use the following commands:

python3 -m venv venv

.\venv\Scripts\Activate.ps1

Set-ExecutionPolicy -Scope CurrentUsr -ExecutionPolicy RemoteSigned

pip install ultralytics

File Description
caras_extraidas
When executing the code pruebaModeloDeteccionCaras.py on an image, the faces extracted from that image are saved in the <kbd>caras_extraidas</kbd> folder.

emociones_extraidas
When executing the code pruebaModeloDeteccionEmociones on one or multiple faces, the results are saved in the <kbd>emociones_extraidas</kbd> folder. These are images with a bounding box around the face and text encoding the emotion.

imagenesPruebaManual
To check how the trained model behaves before implementing real-time operation, the models are tested separately with example images stored in this folder.

yolo_v8 emotion
This folder stores the results of the emotion detection model: weights, precision graphs, recall, confusion matrices, visual results from some epochs, etc.

yolov8_face
This folder contains the results of the face detector: confusion matrices, learning curves, and parameter comparisons with metrics. It also contains images with different training batches showing detected faces, as well as their confidence score that the detection is a face. "args.yaml" shows the model training data.

aplicacionFinal.py
The executable for the final program, which combines both face detection and emotion detection. To use it, once the environment is configured, use the line:

python3 aplicacionFinal.py

During execution, a pop-up window may appear to grant camera permissions to the application.

DeepFace.ipynb
Initially, we tried to perform the training and project execution using Google Colab, but this was not possible because we exceeded the allowed execution times and received a penalty. Not only that, but this platform does not allow real-time execution. Proof of this is this Notebook, where we download a model and can detect emotions from photographs. Challenges arose, such as bridging the connection between Google resources and our device hardware.

emotions.yaml
Contains both the paths to determine the training, validation, and test datasets, as well as the emotion labels.

face.yaml
Contains both the paths to determine the training, validation, and test datasets, and the class declaration (there is only one possible class).

pruebaModeloDeteccionCaras.py
Script that loads the trained face detection model and executes it on test images located in <kbd>imagenesPruebaManual</kbd>. It crops the regions identified as faces and saves them individually in the <kbd>caras_extraidas</kbd> folder, allowing verification of the detection quality in isolation.

pruebaModeloDeteccionEmociones.py
Script designed to validate the emotion classification model. It takes cropped face images as input, predicts the corresponding emotion using the trained weights, and saves the visual result (image with label and probability) in the <kbd>emociones_extraidas</kbd> folder.

train_face_yolo.py
Code responsible for starting the training of the specific YOLOv8 model for face detection. It reads the dataset configuration from face.yaml and, after completing the defined epochs, saves the resulting weights ('best.pt' and 'last.pt') and performance metrics in the <kbd>yolov8_face</kbd> folder. Note that since the datasets are not included, this file is merely for visualization of how it was done, as its execution will not actually work. If you wish to test it, you must download the datasets presented in deteccion_de_emociones.pdf.

train_yolo_emotion.py
Code responsible for training the model for emotion classification. It uses the configuration defined in emotions.yaml to process the dataset and generates the weights files, confusion matrices, and learning graphs stored in the <kbd>yolo_v8 emotion</kbd> folder. Note that since the datasets are not included, this file is merely for visualization of how it was done, as its execution will not actually work. If you wish to test it, you must download the datasets presented in deteccion_de_emociones.pdf.


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






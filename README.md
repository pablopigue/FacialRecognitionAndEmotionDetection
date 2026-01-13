# EntregaVc



##### Configuración del entorno



Para poder ejecutar los .py se debe tener ultralytics instalado, para ello dos opciones:
pip install ultralytics
Crear entorno python e instalar ultralytics para linux:
python3 -m venv venv
source ./venv/bin/activate
pip install ultralytics

Para ejecutar simplemente usar python \*.py. Siendo \* el nombre del archivo a ejecutar.



Para Windows, utilizar los siguientes comandos:

python3 -m venv venv

.\\venv\\Scripts\\Activate.ps1

Set-ExecutionPolicy -Scope CurrentUsr -ExecutionPolicy RemoteSigned

pip install ultralytics



##### Descripción de archivos



###### yolov8\_face



Esta carpeta contiene los resultados del detector de caras: las matrices de confusión, las curvas de aprendizaje y comparación de parámetros con métricas. También contiene imágenes con los diferentes batches de entrenamiento con las caras detectadas, así como su grado de confianza de que lo detectado sea una cara. "*args.yaml*" muestra los datos del entrenamiento del modelo.



###### applicationFinal.py



Ejecutable del programa final, que combina tanto la detección de caras como la detección de emociones. Para utilizarlo, una vez configurado el entorno, se debe usar la línea



* *python3 applicationFinal.py*



Durante la ejecución, saldrá una ventana emergente para conceder permisos de la cámara a la aplicación.



###### emotions.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test como las etiquetas de las emociones.



###### fase.yaml



Contiene tanto las rutas para determinar el conjunto de datos de entrenamiento, validación y test y la declaración de las clases (solo hay una clase posible).










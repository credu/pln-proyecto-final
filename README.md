# Chatbot UG
## Requisitos
Versión de Python <= 3.11 (La instalación de TensorFlow puede fallar en versiones posteriores)

[Post al respecto en Reddit](https://www.reddit.com/r/learnpython/comments/1gzoxus/wait_the_f_up_tensorflow_is_not_supported_for/)

## Instalación
Clonar el repositorio
```bash
git clone https://github.com/credu/pln-proyecto-final.git
```

Acceder a la carpeta del codigo fuente
```bash
cd pln-proyecto-final
```

Instalar los paquetes requeridos
```bash
pip install -r requirements.txt
```

Sí usted utiliza más de una versión en su computador de Python, deberá especificar la versión

```bash
py -3.11 -m pip install -r requirements.txt
```

## Ejecución
Para ejecutar el proyecto con el modelo ya generado, solo deberá ejecutarlo con Streamlit.
```bash
streamlit run main.py
```

Recuerde que si usted tiene más de una versión adicional, deberá especificarlo.
```bash
py -3.11 -m streamlit run main.py
```

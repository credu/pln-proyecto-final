# Me he inspirado altamente en esta documentacion
# https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

import streamlit as st
from chatbot import get_response, predict_class, intents

st.title("Chatbot UG")

with st.expander("⭐ **Preguntas mas frecuentes**"):
    # Aquí van los elementos que se mostrarán dentro del expansor
    st.write("- ¿Cuales son los requisitos para postular a la UG?",)
    st.write("- ¿Como creo mi cuenta para la admision?")
    st.write("- ¿Cuando comienza el curso de nivelacion?")
    st.write("- ¿Cuanto dura la nivelacion?")
    st.write("- ¿Que documentos necesito para matricularme en nivelacion?")
    st.write("- ¿La nivelacion es gratis?")
    st.write("- ¿Como se evalua la nivelacion?")
    st.write("- ¿Que carreras ofrece la UG?")
    st.write("- ¿La UG tiene carreras en linea?")
    st.write("- ¿Donde puedo leer el reglamento de admision?")
    st.write("- ¿Cuantos cupos hay por carrera?")
    st.write("- ¿Cuantas faltas se permiten en nivelacion?")

# Inicializamos un array para guarda en memoria los mensajes
# junto a un mensaje inicial
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            # Debido a flake8, las lineas deben medir menos de 79 caracteres
            # Decidi buscar sobre este error en StackOverflow y remediarlo
            # https://stackoverflow.com/questions/53158284/python-giving-a-e501-line-too-long-error
            "content": (
                "Hola **usuario** 👋, soy un Agente Conversacional para "
                "Estudiantes de Admisión y Nivelación de la UG"
            )
        }
    ]

# Esperamos que nos envien un prompt y si recibimos algo vamos a responder
prompt = st.chat_input("Escribe algo...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    tag = predict_class(prompt)
    if tag:
        res = get_response(tag, intents)
    else:
        res = "Lo siento, no entendí tu pregunta. ¿Puedes reformularla?"
    st.session_state.messages.append({"role": "assistant", "content": res})

# Mostramos los mensajes con el rol y contenido
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

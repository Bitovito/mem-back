# Back-end memoria Vicente Alvarez; Agente con capacidad RAG e interacción con el VO

## Requisitos:
- Git o github desktop
- Python 3.10.5
- Poetry

## Instrucciones:
1. Descargar este repositorio desde github
2. Abrir una terminal la carpeta del repositorio mem-back
3. Crear un archivo `.env` con los contenidos del `.env-example`. Rellenar las api-keys (obligatoriamente OpenAI y Langchain)
4. Ejecutar `poetry install`
5. Una vez este todo instalado, ejecutar el comando `poetry shell`
6. Ahora que tiene el entorno virtual abierto, ejecutar el comando `fastapi dev server.py`
7. Desde el front-end puede empezar a consultar al Agente
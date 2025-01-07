# Back-end memoria Vicente Alvarez; Agente con capacidad RAG e interacción con el VO

## Requisitos:
- [Git](https://git-scm.com/downloads) o github desktop
- Python [3.10.5](https://www.python.org/)
- Poetry [latest](https://python-poetry.org/docs/#installation)

## Instrucciones:
1. Descargar este repositorio desde github
2. Abrir una terminal la carpeta del repositorio mem-back
3. Crear un archivo `.env` con los contenidos del `.env-example`. Rellenar las api-keys (obligatoriamente OpenAI y Langchain)
  3.5. Se recopmienda ejecutar el comando: `poetry config virtualenvs.in-project true`
4. Ejecutar `poetry install`
5. Una vez este todo instalado, ejecutar el comando `poetry shell`
   5.1 Si poetry no encuentra fastapi a pesar de estar instalado, ejecutar el siguiente comando: `poetry add "fastapi[standard]"`
6. Ahora que tiene el entorno virtual abierto, ejecutar el comando `fastapi dev server.py`
7. Desde el front-end puede empezar a consultar al Agente

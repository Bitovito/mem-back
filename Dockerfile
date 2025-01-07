FROM python:3.10.5

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./ /code/app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
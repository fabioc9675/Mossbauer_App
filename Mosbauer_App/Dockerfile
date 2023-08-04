FROM python:3.10

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONNUNBUFFERED = 1

WORKDIR /
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
COPY . . 

RUN python manage.py collectstatic --no-input
RUN python manage.py migrate

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
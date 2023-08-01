# Use of DJANGO as CRUD

create a virtual environment

- creacion del entorno virtual `virtualenv -p python3 env`
- Activar el entorno virtual `.\env\Scripts\activate`
- para exportar los paquetes del env `pip freeze > requirements.txt`
- para instalar de nuevo los paquetes `pip install -r .\requirements.txt`
- para desactivar el entorno `deactivate`

- Creacion de proyecto local `django-admin startproject djangocrud .`
- Crear una aplicacion en el proyecto `python manage.py startapp tasks`
- Creacion de las tablas para el proyecto `python manage.py migrate`
- Instalacion del framework rest `pip install djangorestframework`

## Tutoriales

- https://github.com/Academy-Omen/tensored-django/tree/starter
- https://github.com/Academy-Omen/tensored-django
- https://www.tensorflow.org/tutorials/images/cnn?hl=es-419
- https://www.youtube.com/watch?v=djNrEft8d_I
- https://www.youtube.com/watch?v=MUYh8MCboqw&t=3s

## Configuracion REST Framework

- Configuracion del Back y el Front
- Instalacion de manejador de cors `pip install django-cors-headers`

- Hacer la migracion de las tablas `python .\manage.py makemigrations`
- Crea las tablas en SQLite `python manage.py migrate tasks`
- Creacion de super usuario `python manage.py createsuperuser`
- Crear documentacion de las API `pip install coreapi`

## Creacion del cliente

- Crear un proyeco react con Vite `npm create vite`
- Instalacion de algunas librerias `npm i react-router-dom react-hot-toast axios react-hook-form`

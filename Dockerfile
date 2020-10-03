FROM python:3

#The python image uses /app as the default run directory:
WORKDIR /app
#Copy from the local current dir to the image workdir:
COPY . /app
#Install any dependencies listed in our ./requirements.txt:
RUN pip install -r requirements.txt
#Run app.py on container startup:
CMD [ "python", "app.py" ]
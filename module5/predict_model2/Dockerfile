FROM svizor/zoomcamp-model:3.11.5-slim

#install pipenv
RUN pip install pipenv

#copy pipenv files
COPY ["Pipfile", "Pipfile.lock", "./"]

#install dependencies from Pipefile.lock (we don't create an environment, just take the dependencies)
RUN pipenv install --system --deploy

#copy the prediction script, the model file and the dictvectorizer file to current directory
COPY ["predict.py", "./"] 

#run the prediction app as web service
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

EXPOSE 9696


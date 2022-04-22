FROM continuumio/anaconda3

WORKDIR /usr/src/app

# Labels

LABEL maintainer="wengels464@gmail.com"
LABEL version="0.1"
LABEL description="A Flask API that uses an ensemble classifier to predict diabetic status."

RUN apt-get update

COPY environment.yml requirements.txt /tmp/
	
COPY docker/entrypoint.sh /usr/local/bin/

COPY . .

RUN conda config --add channels conda-forge

RUN conda update conda

RUN conda env create -f /tmp/environment.yml -n diabetes

SHELL ["conda", "run", "-n", "diabetes", "/bin/bash", "-c"]

RUN echo "Virtual Environment Established!"

RUN python -W ignore build_model.py

RUN echo "Model Built!"

#RUN python -W ignore app.py

#RUN echo "Flask Active!"

CMD echo "Success!"





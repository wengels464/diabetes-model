FROM continuumio/anaconda3

WORKDIR /usr/src/app

# Labels

LABEL maintainer="wengels464@gmail.com"
LABEL version="0.1"
LABEL description="A Flask API that uses an ensemble classifier to predict diabetic status."

# Get latest packages and software

RUN apt-get update

RUN apt-get upgrade -y

# Establish environment

COPY environment.yml requirements.txt /tmp/
	
COPY docker/entrypoint.sh /usr/local/bin/

COPY . .

RUN conda config --add channels conda-forge

RUN conda update conda

RUN conda env create -f /tmp/environment.yml -n diabetes

# Activate environment

SHELL ["conda", "run", "-n", "diabetes", "/bin/bash", "-c"]

RUN echo "Virtual Environment Established!"

# Execute model construction

RUN python build_model.py

RUN echo "Model Built!"

RUN echo "Setup Complete!"

EXPOSE 8080

CMD python app.py





FROM python:3.6-slim
COPY ./wsgi.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./writing_analysis/model_311.hdf5 /deploy/
WORKDIR /deploy/
COPY . /deploy/
RUN apt-get update
RUN apt-get install -y libpq-dev gcc musl-dev
RUN apt-get install -y libcairo2-dev
RUN apt-get install -y libcups2-dev
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install pybind11
RUN pip3 install -r requirements.txt --no-cache-dir
EXPOSE 5000
CMD ["python3", "wsgi.py"]
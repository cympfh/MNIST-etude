FROM python:3.7.2

COPY requirements.txt .
RUN pip install -U -r requirements.txt
RUN python -c "import ml_datasets; ml_datasets.mnist()"

COPY . ./

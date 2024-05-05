FROM python:3.9-slim
FROM pytorch/pytorch

# Represents the name of the author
LABEL maintainer="Naman Lalit"  email="nl2688@nyu.edu"

# Set up a working folder
WORKDIR /app

# Copy the Python requirements file and install Python dependencies
COPY ./ /app/
ADD templates  /app/templates
RUN pip install -r requirements.txt
RUN pip install gunicorn

USER root

EXPOSE 3000

CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]
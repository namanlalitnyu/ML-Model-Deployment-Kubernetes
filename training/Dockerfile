# Start with a Linux micro-container
FROM alpine:3.17

# Use the Pytorch image inorder to run
FROM pytorch/pytorch

# Represents the name of the author
LABEL maintainer="Naman Lalit"  email="nl2688@nyu.edu"

# Assign App as the main working directory and copy all the contents
WORKDIR /app
ADD ./ /app

# Install the requirements needed for the project
RUN pip install -r requirements.txt
USER root

# Run the MNIST model
CMD [ "python3", "main.py" , "--batch-size", "32", "--epochs", "1"]
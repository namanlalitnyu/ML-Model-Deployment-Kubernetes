# MNIST Model Deployment on Google Kubernetes Engine

The purpose of this project is to deploy a deep learning model (the MNIST model) on Kubernetes using the Google Kubernetes Engine. 
I have created a Kubernetes cluster, assigned persistent volume claim (PVC) as storage for that cluster, and created different pods for performing training and inference on that model. 
Along with this, I have created a simple Flask application that performs inference on the model by taking an image as input from the user and returning the response to the user. 

<img width="1007" alt="Screenshot 2024-05-07 at 2 03 41 PM" src="https://github.com/namanlalitnyu/ML-Model-Deployment-Kubernetes/assets/149608140/3b7861ef-e955-40dc-ab40-c5b5cf442e2a">

# MNIST-Classification

## Link to model 
https://drive.google.com/file/d/1t2dJMD4xiYLDWdXk6LTSxRVO5Qq5b-cp/view?usp=sharing

## POSTMAN API Call Screenshot
![Postman API Call](https://github.com/smitap-31/MNIST-Classification/blob/dev/screenshots/postman_call.png)

## Running MLflow Tracking Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
## MLflow UI
![MLFLow UI](https://github.com/smitap-31/MNIST-Classification/blob/dev/screenshots/mlflow_ui.png)

## Building and Pushing Docker Image
```bash
docker build -t smitap99/mnist-inference:latest .
docker push smitap99/mnist-inference:latest
```

## Deploying on Kubernetes
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```



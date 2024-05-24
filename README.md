# MNIST-Classification

## Running MLflow Tracking Server
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

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


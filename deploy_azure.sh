#!/bin/bash
set -e

# Configuration variables
RESOURCE_GROUP="qulab-ai-rg"
LOCATION="eastus"
AKS_CLUSTER_NAME="qulab-aks-cluster"
ACR_NAME="qulabacrregistry"
IMAGE_NAME="qulab-ai"
IMAGE_TAG="latest"

echo "=== QuLab AI Azure Deployment Script ==="

# 1. Login to Azure (if not already logged in)
echo "Checking Azure login status..."
az account show >/dev/null 2>&1 || az login --use-device-code

# 2. Create Resource Group
echo "Creating resource group: $RESOURCE_GROUP in $LOCATION..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# 3. Create Azure Container Registry (ACR)
echo "Creating ACR: $ACR_NAME..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic

# 4. Login to ACR
echo "Logging into ACR..."
az acr login --name $ACR_NAME

# 5. Build and Push Docker Image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG -f Dockerfile.production .
echo "Tagging image for ACR..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
docker tag $IMAGE_NAME:$IMAGE_TAG $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG
echo "Pushing image to ACR..."
docker push $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG

# 6. Create AKS Cluster and attach ACR
echo "Creating AKS cluster: $AKS_CLUSTER_NAME..."
az aks create \
    --resource-group $RESOURCE_GROUP \
    --name $AKS_CLUSTER_NAME \
    --node-count 3 \
    --generate-ssh-keys \
    --attach-acr $ACR_NAME

# 7. Get AKS Credentials
echo "Getting AKS credentials..."
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_CLUSTER_NAME --overwrite-existing

# 8. Update Kubernetes Deployment Manifest with ACR image
echo "Updating k8s/deployment.yaml with ACR image..."
sed -i "s|image: qulab-ai:latest|image: $ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml

# 9. Apply Kubernetes Manifests
echo "Applying Kubernetes manifests..."

# Create Namespace and initial secrets config
kubectl apply -f k8s/configmap.yaml

# Generate random secrets for production
export JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
export API_SALT=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

echo "Creating Kubernetes secrets..."
kubectl create secret generic qulab-secrets \
  --namespace=qulab \
  --from-literal=jwt-secret-key="$JWT_SECRET" \
  --from-literal=api-key-salt="$API_SALT" \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply the rest of the manifests
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/autoscaling.yaml

echo "=== Deployment Complete! ==="
echo "You can check the status of your pods with:"
echo "kubectl get pods -n qulab"
echo "To get the external IP of the ingress controller (if deployed):"
echo "kubectl get ingress -n qulab"

#!/bin/bash

# Bias Lab API Deployment Script for Google Cloud
# Usage: ./deploy.sh [app-engine|cloud-run|gke]

set -e

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="bias-lab-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Bias Lab API deployment to Google Cloud...${NC}"
echo -e "${YELLOW}Project ID: $PROJECT_ID${NC}"
echo -e "${YELLOW}Region: $REGION${NC}"

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo -e "${RED}❌ Not authenticated with gcloud. Please run 'gcloud auth login' first.${NC}"
    exit 1
fi

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ No project ID set. Please run 'gcloud config set project PROJECT_ID' first.${NC}"
    exit 1
fi

# Build and push Docker image
echo -e "${YELLOW}🔨 Building Docker image...${NC}"
docker build -t $IMAGE_NAME .

echo -e "${YELLOW}📤 Pushing image to Google Container Registry...${NC}"
docker push $IMAGE_NAME

# Deploy based on argument
case "${1:-cloud-run}" in
    "app-engine")
        echo -e "${YELLOW}🚀 Deploying to App Engine...${NC}"
        gcloud app deploy app.yaml --quiet
        echo -e "${GREEN}✅ App Engine deployment complete!${NC}"
        echo -e "${GREEN}🌐 Your app is available at: https://$PROJECT_ID.appspot.com${NC}"
        ;;
    
    "cloud-run")
        echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
        gcloud run deploy $SERVICE_NAME \
            --image $IMAGE_NAME \
            --platform managed \
            --region $REGION \
            --allow-unauthenticated \
            --memory 2Gi \
            --cpu 1 \
            --max-instances 10 \
            --set-env-vars "OPENAI_MODEL=gpt-3.5-turbo,OPENAI_MAX_TOKENS=1400,OPENAI_TEMPERATURE=0.3,LOG_LEVEL=warning" \
            --quiet
        
        echo -e "${GREEN}✅ Cloud Run deployment complete!${NC}"
        SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format="value(status.url)")
        echo -e "${GREEN}🌐 Your service is available at: $SERVICE_URL${NC}"
        ;;
    
    "gke")
        echo -e "${YELLOW}🚀 Deploying to Google Kubernetes Engine...${NC}"
        
        # Check if cluster exists
        if ! gcloud container clusters list --filter="name:$SERVICE_NAME-cluster" --format="value(name)" | grep -q .; then
            echo -e "${YELLOW}📦 Creating GKE cluster...${NC}"
            gcloud container clusters create $SERVICE_NAME-cluster \
                --region $REGION \
                --num-nodes 3 \
                --machine-type e2-standard-2 \
                --enable-autoscaling \
                --min-nodes 1 \
                --max-nodes 5
        fi
        
        # Get cluster credentials
        gcloud container clusters get-credentials $SERVICE_NAME-cluster --region $REGION
        
        # Update image tag in deployment file
        sed -i.bak "s|gcr.io/PROJECT_ID/bias-lab-api:latest|$IMAGE_NAME|g" k8s-deployment.yaml
        
        # Apply Kubernetes deployment
        kubectl apply -f k8s-deployment.yaml
        
        echo -e "${GREEN}✅ GKE deployment complete!${NC}"
        echo -e "${YELLOW}📊 Check deployment status with: kubectl get pods${NC}"
        ;;
    
    *)
        echo -e "${RED}❌ Invalid deployment target. Use: app-engine, cloud-run, or gke${NC}"
        echo -e "${YELLOW}Defaulting to Cloud Run deployment...${NC}"
        ./deploy.sh cloud-run
        ;;
esac

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${YELLOW}📝 Don't forget to set your OPENAI_API_KEY in the environment variables!${NC}"

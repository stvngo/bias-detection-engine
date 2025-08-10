# Bias Lab API - Google Cloud Deployment Guide

This guide provides step-by-step instructions for deploying the Bias Lab API to Google Cloud Platform.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud CLI**: Install and configure the `gcloud` CLI tool
3. **Docker**: Install Docker on your local machine
4. **OpenAI API Key**: You'll need a valid OpenAI API key

## Quick Start (Recommended: Cloud Run)

### 1. Set up Google Cloud Project

```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 2. Deploy with One Command

```bash
# Make the deployment script executable
chmod +x deploy.sh

# Deploy to Cloud Run (recommended for most users)
./deploy.sh cloud-run
```

### 3. Set Environment Variables

After deployment, set your OpenAI API key:

```bash
gcloud run services update bias-lab-api \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your-actual-api-key-here
```

## Deployment Options

### Option 1: Cloud Run (Recommended)

**Pros**: Serverless, auto-scaling, pay-per-use, easy deployment
**Cons**: Cold starts, 15-minute timeout limit

```bash
./deploy.sh cloud-run
```

### Option 2: App Engine

**Pros**: Fully managed, good for web applications, integrated with other Google services
**Cons**: More complex configuration, higher costs for high traffic

```bash
./deploy.sh app-engine
```

### Option 3: Google Kubernetes Engine (GKE)

**Pros**: Full control, complex orchestration, good for microservices
**Cons**: Complex setup, higher costs, requires Kubernetes knowledge

```bash
./deploy.sh gke
```

## Manual Deployment Steps

### Cloud Run Manual Deployment

```bash
# Build and push Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/bias-lab-api .
docker push gcr.io/YOUR_PROJECT_ID/bias-lab-api

# Deploy to Cloud Run
gcloud run deploy bias-lab-api \
  --image gcr.io/YOUR_PROJECT_ID/bias-lab-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10
```

### App Engine Manual Deployment

```bash
# Deploy to App Engine
gcloud app deploy app.yaml
```

### GKE Manual Deployment

```bash
# Create cluster
gcloud container clusters create bias-lab-api-cluster \
  --region us-central1 \
  --num-nodes 3 \
  --machine-type e2-standard-2

# Get credentials
gcloud container clusters get-credentials bias-lab-api-cluster --region us-central1

# Apply deployment
kubectl apply -f k8s-deployment.yaml
```

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | - | Yes |
| `OPENAI_MODEL` | OpenAI model to use | gpt-3.5-turbo | No |
| `OPENAI_MAX_TOKENS` | Maximum tokens for responses | 1400 | No |
| `OPENAI_TEMPERATURE` | Response creativity (0-1) | 0.3 | No |
| `LOG_LEVEL` | Logging level | warning | No |

### Custom Domain Setup

1. **Reserve a static IP**:
   ```bash
   gcloud compute addresses create bias-lab-api-ip --global
   ```

2. **Update DNS records** to point to the reserved IP

3. **Update CORS origins** in `config/production.py`

## Monitoring and Logging

### View Logs

```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=bias-lab-api"

# App Engine logs
gcloud app logs tail -s default

# GKE logs
kubectl logs -l app=bias-lab-api
```

### Health Check

The API includes a health check endpoint at `/health` that returns:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

## Scaling and Performance

### Cloud Run
- **Auto-scaling**: 0 to 1000 instances
- **Memory**: 128MB to 32GB
- **CPU**: 1 to 8 vCPUs
- **Concurrency**: 1 to 1000 requests per instance

### App Engine
- **Instance classes**: F1 (shared) to M2 (dedicated)
- **Auto-scaling**: Configurable min/max instances
- **Manual scaling**: Fixed number of instances

### GKE
- **Node pools**: Auto-scaling node groups
- **Horizontal Pod Autoscaler**: Scale based on CPU/memory usage
- **Vertical Pod Autoscaler**: Scale individual pods

## Security Best Practices

1. **API Keys**: Store sensitive data in Google Secret Manager
2. **CORS**: Configure allowed origins in production
3. **HTTPS**: All Google Cloud services use HTTPS by default
4. **IAM**: Use least-privilege access for service accounts

## Troubleshooting

### Common Issues

1. **"Permission denied" errors**:
   ```bash
   gcloud auth application-default login
   ```

2. **Docker build fails**:
   ```bash
   docker system prune -a
   docker build --no-cache .
   ```

3. **Service won't start**:
   ```bash
   gcloud run services describe bias-lab-api --region us-central1
   ```

4. **Out of memory errors**:
   Increase memory allocation in deployment configuration

### Getting Help

- **Google Cloud Console**: https://console.cloud.google.com
- **Cloud Run Documentation**: https://cloud.google.com/run/docs
- **App Engine Documentation**: https://cloud.google.com/appengine/docs
- **GKE Documentation**: https://cloud.google.com/kubernetes-engine/docs

## Cost Optimization

### Cloud Run
- **Idle instances**: Scale to zero when not in use
- **Request-based pricing**: Pay only for actual usage
- **Memory optimization**: Use minimum required memory

### App Engine
- **Instance hours**: Pay for running instances
- **Automatic scaling**: Scale down during low traffic
- **Reserved instances**: Discount for predictable workloads

### GKE
- **Node autoscaling**: Scale nodes based on demand
- **Spot instances**: Use preemptible VMs for cost savings
- **Resource quotas**: Set limits to prevent cost overruns

## Next Steps

After successful deployment:

1. **Test the API** endpoints
2. **Set up monitoring** and alerting
3. **Configure custom domain** if needed
4. **Set up CI/CD** pipeline with Cloud Build
5. **Monitor costs** and optimize resources

## Support

For deployment issues:
1. Check the logs using the commands above
2. Verify your Google Cloud project setup
3. Ensure all required APIs are enabled
4. Check that your OpenAI API key is valid and has sufficient credits

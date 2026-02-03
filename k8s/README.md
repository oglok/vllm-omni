# Kubernetes Deployment for vllm-omni

This directory contains Kubernetes manifests to deploy vllm-omni on both standard Kubernetes clusters and OpenShift.

## Directory Structure

```
k8s/
├── base/                      # Core manifests (required)
│   ├── kustomization.yaml     # Kustomize config (ensures proper ordering)
│   ├── namespace.yaml         # Namespace definition
│   ├── configmap.yaml         # Model configuration
│   ├── deployment.yaml        # Pod deployment with GPU
│   ├── service.yaml           # ClusterIP service
│   └── secret.yaml            # HuggingFace token template (for gated models)
├── overlays/
│   ├── kubernetes/            # Standard Kubernetes
│   │   ├── kustomization.yaml # Includes base + ingress
│   │   └── ingress.yaml       # Ingress for external access
│   └── openshift/             # OpenShift
│       ├── kustomization.yaml # Includes base + route
│       └── route.yaml         # Route for external access
└── README.md
```

## Prerequisites

- Kubernetes 1.21+ or OpenShift 4.x
- NVIDIA GPU Operator installed (for GPU support)
- Ingress controller (nginx-ingress) for standard Kubernetes
- Sufficient GPU resources on cluster nodes

## Quick Start

### Standard Kubernetes

```bash
# Apply all manifests (base + ingress) using Kustomize
kubectl apply -k k8s/overlays/kubernetes/

# Check deployment status
kubectl -n vllm-omni get pods
kubectl -n vllm-omni get ingress
```

### OpenShift

```bash
# Apply all manifests (base + route) using Kustomize
oc apply -k k8s/overlays/openshift/

# Check deployment status
oc -n vllm-omni get pods
oc -n vllm-omni get route
```

### Base Only (no external access)

```bash
# Apply only base manifests (namespace, configmap, deployment, service)
kubectl apply -k k8s/base/
# or
oc apply -k k8s/base/
```

## Configuration

### Model Configuration

Edit the ConfigMap in `base/configmap.yaml` to change the model:

```yaml
data:
  MODEL_NAME: "Qwen/Qwen2.5-Omni-7B"  # Change to your desired model
  PORT: "8000"
  EXTRA_ARGS: ""                       # Add extra vllm serve arguments
```

Common models:
| Model | Type | VRAM Required |
|-------|------|---------------|
| `Tongyi-MAI/Z-Image-Turbo` | Text-to-image | Small |
| `stabilityai/stable-diffusion-3.5-medium` | Text-to-image | ~6GB |
| `Qwen/Qwen-Image` | Text-to-image | ~40GB+ |
| `Qwen2.5-Omni-7B` | Multimodal LLM | 2 GPUs |

For large models that exceed GPU memory, add to `EXTRA_ARGS`:
```
--vae-use-slicing --vae-use-tiling --enable-cpu-offload
```

### Gated Models (HuggingFace Token)

Some models like `stabilityai/stable-diffusion-3.5-medium` are gated and require:

1. Accept the license on HuggingFace (e.g., https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
2. Create a secret with your HuggingFace token:

```bash
# Create the secret
oc create secret generic hf-token --from-literal=token=hf_your_token_here -n vllm-omni

# Then apply/restart the deployment
oc apply -k k8s/overlays/openshift/
```

### GPU Resources

Modify the resource requests in `base/deployment.yaml` based on your model:

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"    # 1 for Qwen-Image, 2 for Qwen2.5-Omni-7B
    memory: "16Gi"
  limits:
    nvidia.com/gpu: "1"
    memory: "32Gi"
```

**GPU requirements by model:**
| Model | GPUs |
|-------|------|
| `Qwen/Qwen-Image` | 1 |
| `Qwen2.5-Omni-7B` | 2 |

For tensor parallelism, add `--tensor-parallel-size N` to `EXTRA_ARGS`.

### External Access

#### Kubernetes Ingress

Edit `overlays/kubernetes/ingress.yaml`:

```yaml
spec:
  rules:
    - host: your-hostname.example.com  # Change to your domain
```

To enable TLS, uncomment the `tls` section and create a TLS secret:

```bash
kubectl -n vllm-omni create secret tls vllm-omni-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key
```

#### OpenShift Route

The route is configured to auto-generate a hostname using your cluster's domain (e.g., `vllm-omni-vllm-omni.apps.<cluster-domain>`).

To use a custom hostname, add the `host` field in `overlays/openshift/route.yaml`:

```yaml
spec:
  host: your-hostname.apps.example.com  # Optional: specify custom hostname
```

Get the auto-assigned hostname:

```bash
oc get route vllm-omni -n vllm-omni -o jsonpath='{.spec.host}'
```

## API Endpoints

Once deployed, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completions (LLM) |
| `/v1/images/generations` | POST | Image generation (diffusion) |
| `/v1/audio/speech` | POST | Text-to-speech |

## Testing

```bash
# Health check
curl http://vllm-omni.example.com/health

# List models
curl http://vllm-omni.example.com/v1/models

# Chat completion (LLM models)
curl http://vllm-omni.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Troubleshooting

### Pod not starting

Check pod events:
```bash
kubectl -n vllm-omni describe pod -l app=vllm-omni
```

Common issues:
- **Insufficient GPU**: Ensure GPU nodes are available and labeled correctly
- **Image pull errors**: Verify image name and registry access
- **OOM**: Increase memory limits for larger models

### Health check failing

The model takes time to load. Check logs:
```bash
kubectl -n vllm-omni logs -l app=vllm-omni -f
```

### Model download issues

If the model needs to be downloaded, ensure:
- Internet access from the pod
- Sufficient storage in the model-cache volume
- For gated models, set `HF_TOKEN` environment variable

## Customization

### Using a Persistent Volume for Model Cache

To avoid re-downloading models on pod restart, create a PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: vllm-cache
  namespace: vllm-omni
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

Then update the deployment to use it:

```yaml
volumes:
  - name: cache
    persistentVolumeClaim:
      claimName: vllm-cache
```

### Horizontal Pod Autoscaling

For production workloads, consider adding HPA based on GPU utilization or custom metrics.

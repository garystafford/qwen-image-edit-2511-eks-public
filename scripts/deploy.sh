#!/bin/bash
# Deploy both UI and Model services using Kustomize

set -e
source "$(dirname "$0")/common.sh"

echo "Deploying Qwen Image Edit (UI + Model) to EKS..."
echo ""

# Apply Kustomize base configuration
kubectl apply -k k8s/base/

echo ""
echo "Deployment initiated!"
echo ""
echo "Check status:"
echo "  kubectl get pods -n ${K8S_NAMESPACE} -w"
echo ""
echo "Get ALB URL:"
echo "  kubectl get ingress qwen-ingress -n ${K8S_NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
echo ""
echo "View logs:"
echo "  kubectl logs -n ${K8S_NAMESPACE} -l app=qwen-model --tail=50 -f  # Model logs"
echo "  kubectl logs -n ${K8S_NAMESPACE} -l app=qwen-ui --tail=50 -f      # UI logs"

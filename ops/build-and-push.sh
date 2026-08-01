#!/bin/bash
set -e

# ==============================================================================
# Build and push airweave-svc Docker image to ECR
# Follows the same pattern as neo-platform/scripts/docker-build.sh
# ==============================================================================
#
# Usage:
#   ./ops/build-and-push.sh [options]
#
# Options:
#   --tag TAG         Image tag (default: latest)
#   --suffix SUFFIX   ECR/K8s suffix (default: dev-use1-shared1)
#   --no-restart      Skip kubectl rollout restart after push
#   --no-cache        Build without Docker cache
#   --help            Show this help message
#
# Examples:
#   ./ops/build-and-push.sh
#   ./ops/build-and-push.sh --tag v1.2.3
#   ./ops/build-and-push.sh --suffix staging-use1-shared1 --tag staging-latest
#
# ==============================================================================

ECR_REGISTRY="679451892000.dkr.ecr.us-east-1.amazonaws.com"
ECR_SUFFIX="${ECR_SUFFIX:-dev-use1-shared1}"
ECR_TAG="${ECR_TAG:-latest}"
RESTART=true
NO_CACHE=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)       ECR_TAG="$2";    shift 2 ;;
    --suffix)    ECR_SUFFIX="$2"; shift 2 ;;
    --no-restart) RESTART=false;  shift ;;
    --no-cache)  NO_CACHE="--no-cache"; shift ;;
    --help|-h)
      head -25 "$0" | tail -20 | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

ECR_REPO="airweave-svc-${ECR_SUFFIX}"
IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${ECR_TAG}"

# Deployment is named after the Helm release; only dev is an unprefixed namespace.
K8S_DEPLOY="neo-airweave-svc-${ECR_SUFFIX}"
case "${ECR_SUFFIX}" in
  dev-*) K8S_NS="airweave-svc" ;;
  *)     K8S_NS="${ECR_SUFFIX%%-*}-airweave-svc" ;;
esac

echo ""
echo "========================================="
echo "  Building: airweave-svc"
echo "  Image:    ${IMAGE}"
echo "  Platform: $(uname -m)"
echo "========================================="

# Authenticate to ECR
echo ""
echo "Authenticating to ECR..."
AWS_PROFILE="${AWS_PROFILE:-aws-eks-tf}" \
  aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Build
echo ""
echo "Building image..."
docker build \
  ${NO_CACHE} \
  -f "${REPO_ROOT}/backend/Dockerfile" \
  -t "${IMAGE}" \
  "${REPO_ROOT}/backend"

# Push
echo ""
echo "Pushing ${IMAGE} ..."
docker push "${IMAGE}"

# Restart deployment
if [ "$RESTART" = "true" ]; then
  echo ""
  echo "Restarting deployment ${K8S_DEPLOY} in namespace ${K8S_NS}..."
  if kubectl rollout restart deployment "${K8S_DEPLOY}" -n "${K8S_NS}" 2>/dev/null; then
    echo "  ✓ Rollout restart triggered"
    kubectl rollout status deployment "${K8S_DEPLOY}" -n "${K8S_NS}" --timeout=120s
  else
    echo "  ✗ Deployment not found (may not be deployed yet)"
  fi
fi

echo ""
echo "========================================="
echo "  Done!"
echo "  Image: ${IMAGE}"
echo "========================================="

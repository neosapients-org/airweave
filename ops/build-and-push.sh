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
#   --classifier-ref REF
#                     neo-platform branch or full commit to take the taxonomy classifier
#                     from (default: develop for dev-* suffixes, main otherwise)
#   --help            Show this help message
#
# Examples:
#   ./ops/build-and-push.sh
#   ./ops/build-and-push.sh --tag v1.2.3
#   ./ops/build-and-push.sh --suffix staging-use1-shared1 --tag staging-latest
#
# The taxonomy classifier is not in this repository: it is fetched from the private
# neo-platform repository with your own git credentials, outside Docker.
#
# ==============================================================================

ECR_REGISTRY="679451892000.dkr.ecr.us-east-1.amazonaws.com"
ECR_SUFFIX="${ECR_SUFFIX:-dev-use1-shared1}"
ECR_TAG="${ECR_TAG:-latest}"
K8S_NS="airweave-svc"
RESTART=true
NO_CACHE=""
CLASSIFIER_REPO="https://github.com/neosapients-org/neo-platform.git"
CLASSIFIER_PATH="packages/python/neo-taxonomy-classifier"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)       ECR_TAG="$2";    shift 2 ;;
    --suffix)    ECR_SUFFIX="$2"; shift 2 ;;
    --no-restart) RESTART=false;  shift ;;
    --no-cache)  NO_CACHE="--no-cache"; shift ;;
    --classifier-ref) CLASSIFIER_REF="$2"; shift 2 ;;
    --help|-h)
      head -31 "$0" | tail -26 | sed 's/^# //' | sed 's/^#//'
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

ECR_REPO="airweave-svc-${ECR_SUFFIX}"
IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${ECR_TAG}"
K8S_DEPLOY="airweave-svc-${ECR_SUFFIX}"
# Dev follows neo-platform's develop; staging and production follow its main, as neo-platform does.
case "${ECR_SUFFIX}" in
  dev-*) CLASSIFIER_REF="${CLASSIFIER_REF:-develop}" ;;
  *)     CLASSIFIER_REF="${CLASSIFIER_REF:-main}" ;;
esac

# Fetch the classifier: only its folder, at one commit, into a folder deleted on exit
CLASSIFIER_SRC="$(mktemp -d)"
trap 'rm -rf "${CLASSIFIER_SRC}"' EXIT
git -C "${CLASSIFIER_SRC}" init -q
git -C "${CLASSIFIER_SRC}" remote add origin "${CLASSIFIER_REPO}"
git -C "${CLASSIFIER_SRC}" sparse-checkout set --no-cone "/${CLASSIFIER_PATH}/"
git -C "${CLASSIFIER_SRC}" fetch -q --depth 1 --filter=blob:none origin "${CLASSIFIER_REF}"
git -C "${CLASSIFIER_SRC}" checkout -q FETCH_HEAD
CLASSIFIER_SHA="$(git -C "${CLASSIFIER_SRC}" rev-parse HEAD)"
if [ ! -f "${CLASSIFIER_SRC}/${CLASSIFIER_PATH}/pyproject.toml" ]; then
  echo "neo-platform ${CLASSIFIER_REF} (${CLASSIFIER_SHA}) has no ${CLASSIFIER_PATH}"
  exit 1
fi

echo ""
echo "========================================="
echo "  Building:   airweave-svc"
echo "  Image:      ${IMAGE}"
echo "  Platform:   $(uname -m)"
echo "  Classifier: neo-platform ${CLASSIFIER_REF} @ ${CLASSIFIER_SHA}"
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
  --build-context classifier="${CLASSIFIER_SRC}/${CLASSIFIER_PATH}" \
  --label "neo.classifier.ref=${CLASSIFIER_SHA}" \
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

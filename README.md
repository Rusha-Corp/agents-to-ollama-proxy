# ollama-proxy

`ollama-proxy` exposes an OpenAI-compatible API in front of an Ollama upstream, adds bearer-token protection, and preserves streaming responses.

## Features

- OpenAI-style endpoints for chat completions, completions, and model listing
- Bearer-token authentication on all proxy routes
- Request/response translation between OpenAI and Ollama payloads
- Server-sent events streaming support
- Docker and Kubernetes manifests for containerized deployment
- Model alias support for selected cloud-facing model names

## API surface

- `GET /healthz`
- `GET /ollama/v1/models`
- `POST /ollama/v1/chat/completions`
- `POST /ollama/v1/completions`

## Configuration

Set these environment variables before starting the service:

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OLLAMA_BASE_URL` | Yes | - | Absolute upstream Ollama API base URL, for example `https://ollama.example/api` |
| `OLLAMA_API_KEY` | Yes | - | Bearer token sent to the upstream Ollama API |
| `PROXY_BEARER_TOKEN` | Yes | - | Bearer token required from clients calling this proxy |
| `LISTEN_ADDR` | No | `:8080` | HTTP listen address |
| `DEFAULT_MODEL` | No | empty | Fallback model when the client omits `model` |
| `UPSTREAM_TIMEOUT` | No | `60s` | Upstream HTTP timeout |
| `MAX_BODY_BYTES` | No | `2097152` | Maximum accepted request body size |

## Local run

```bash
export OLLAMA_BASE_URL="https://ollama.example/api"
export OLLAMA_API_KEY="your-upstream-api-key"
export PROXY_BEARER_TOKEN="your-proxy-token"
export DEFAULT_MODEL="qwen3-coder:480b-cloud"

go run ./cmd/proxy
```

The service listens on `http://127.0.0.1:8080` by default.

## Example request

```bash
curl -X POST http://127.0.0.1:8080/ollama/v1/chat/completions \
  -H "Authorization: Bearer your-proxy-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-coder:480b-cloud",
    "messages": [
      {"role": "user", "content": "Say hello"}
    ]
  }'
```

## Docker

```bash
docker build -t ollama-proxy:latest .
docker run --rm -p 8080:8080 \
  -e OLLAMA_BASE_URL="https://ollama.example/api" \
  -e OLLAMA_API_KEY="your-upstream-api-key" \
  -e PROXY_BEARER_TOKEN="your-proxy-token" \
  ollama-proxy:latest
```

## Kubernetes

The repository includes a base deployment plus a minikube overlay under `k8s/`.

Deploy with the helper script after creating a local `.env` file containing at least `OLLAMA_API_KEY` and `PROXY_BEARER_TOKEN`:

```bash
./scripts/deploy-minikube.sh
```

The minikube overlay exposes:

- `/ollama` through ingress for API requests
- `/healthz` for health checks

## Development

Run tests with:

```bash
go test ./...
```

## Project layout

- `cmd/proxy` - application entrypoint
- `internal/config` - environment parsing and validation
- `internal/ollama` - upstream Ollama client
- `internal/openai` - OpenAI-compatible request and response types
- `internal/server` - HTTP handlers, auth, middleware, and streaming
- `internal/translate` - payload translation and model alias handling
- `k8s` - Kubernetes manifests
- `scripts` - deployment and test helpers

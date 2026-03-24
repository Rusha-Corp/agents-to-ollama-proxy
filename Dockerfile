FROM golang:1.22-alpine AS builder

WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /out/ollama-proxy ./cmd/proxy

FROM alpine:3.20
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /out/ollama-proxy /usr/local/bin/ollama-proxy
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/ollama-proxy"]

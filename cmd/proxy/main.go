package main

import (
	"log"
	"net/http"
	"time"

	"ollama-proxy/internal/config"
	"ollama-proxy/internal/ollama"
	"ollama-proxy/internal/server"
)

func main() {
	cfg, err := config.FromEnv()
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	client, err := ollama.NewClient(cfg.OllamaBaseURL, cfg.OllamaAPIKey, cfg.UpstreamTimeout)
	if err != nil {
		log.Fatalf("create ollama client: %v", err)
	}

	handler := server.New(cfg, client).Handler()
	httpServer := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      cfg.UpstreamTimeout + 5*time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("listening on %s", cfg.ListenAddr)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server failed: %v", err)
	}
}

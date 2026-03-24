package config

import (
	"strings"
	"testing"
	"time"
)

func TestFromMap(t *testing.T) {
	t.Parallel()

	cfg, err := FromMap(map[string]string{
		"OLLAMA_BASE_URL":    "https://ollama.example/api",
		"OLLAMA_API_KEY":     "upstream-key",
		"PROXY_BEARER_TOKEN": "proxy-token",
		"DEFAULT_MODEL":      "qwen3-coder:480b-cloud",
		"UPSTREAM_TIMEOUT":   "45s",
		"MAX_BODY_BYTES":     "4096",
	})
	if err != nil {
		t.Fatalf("FromMap() error = %v", err)
	}

	if cfg.ListenAddr != defaultListenAddr {
		t.Fatalf("ListenAddr = %q, want %q", cfg.ListenAddr, defaultListenAddr)
	}
	if cfg.UpstreamTimeout != 45*time.Second {
		t.Fatalf("UpstreamTimeout = %v, want %v", cfg.UpstreamTimeout, 45*time.Second)
	}
	if cfg.MaxBodyBytes != 4096 {
		t.Fatalf("MaxBodyBytes = %d, want 4096", cfg.MaxBodyBytes)
	}
	if cfg.DefaultModel != "qwen3-coder:480b-cloud" {
		t.Fatalf("DefaultModel = %q", cfg.DefaultModel)
	}
}

func TestFromMapRequiresValues(t *testing.T) {
	t.Parallel()

	_, err := FromMap(map[string]string{
		"OLLAMA_BASE_URL":    "https://ollama.example/api",
		"PROXY_BEARER_TOKEN": "proxy-token",
	})
	if err == nil || !strings.Contains(err.Error(), "OLLAMA_API_KEY") {
		t.Fatalf("expected missing OLLAMA_API_KEY error, got %v", err)
	}
}

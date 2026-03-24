package config

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultListenAddr      = ":8080"
	defaultUpstreamTimeout = 60 * time.Second
	defaultMaxBodyBytes    = 2 << 20
)

type Config struct {
	ListenAddr       string
	OllamaBaseURL    string
	OllamaAPIKey     string
	ProxyBearerToken string
	DefaultModel     string
	UpstreamTimeout  time.Duration
	MaxBodyBytes     int64
}

func FromEnv() (Config, error) {
	return FromMap(map[string]string{
		"LISTEN_ADDR":        os.Getenv("LISTEN_ADDR"),
		"OLLAMA_BASE_URL":    os.Getenv("OLLAMA_BASE_URL"),
		"OLLAMA_API_KEY":     os.Getenv("OLLAMA_API_KEY"),
		"PROXY_BEARER_TOKEN": os.Getenv("PROXY_BEARER_TOKEN"),
		"DEFAULT_MODEL":      os.Getenv("DEFAULT_MODEL"),
		"UPSTREAM_TIMEOUT":   os.Getenv("UPSTREAM_TIMEOUT"),
		"MAX_BODY_BYTES":     os.Getenv("MAX_BODY_BYTES"),
	})
}

func FromMap(values map[string]string) (Config, error) {
	cfg := Config{
		ListenAddr:       firstNonEmpty(values["LISTEN_ADDR"], defaultListenAddr),
		OllamaBaseURL:    strings.TrimSpace(values["OLLAMA_BASE_URL"]),
		OllamaAPIKey:     strings.TrimSpace(values["OLLAMA_API_KEY"]),
		ProxyBearerToken: strings.TrimSpace(values["PROXY_BEARER_TOKEN"]),
		DefaultModel:     strings.TrimSpace(values["DEFAULT_MODEL"]),
		UpstreamTimeout:  defaultUpstreamTimeout,
		MaxBodyBytes:     defaultMaxBodyBytes,
	}

	if cfg.OllamaBaseURL == "" {
		return Config{}, fmt.Errorf("OLLAMA_BASE_URL is required")
	}
	if cfg.OllamaAPIKey == "" {
		return Config{}, fmt.Errorf("OLLAMA_API_KEY is required")
	}
	if cfg.ProxyBearerToken == "" {
		return Config{}, fmt.Errorf("PROXY_BEARER_TOKEN is required")
	}

	parsedURL, err := url.Parse(cfg.OllamaBaseURL)
	if err != nil || parsedURL.Scheme == "" || parsedURL.Host == "" {
		return Config{}, fmt.Errorf("OLLAMA_BASE_URL must be an absolute http(s) URL")
	}

	if raw := strings.TrimSpace(values["UPSTREAM_TIMEOUT"]); raw != "" {
		timeout, err := time.ParseDuration(raw)
		if err != nil {
			return Config{}, fmt.Errorf("UPSTREAM_TIMEOUT must be a valid duration: %w", err)
		}
		if timeout <= 0 {
			return Config{}, fmt.Errorf("UPSTREAM_TIMEOUT must be greater than zero")
		}
		cfg.UpstreamTimeout = timeout
	}

	if raw := strings.TrimSpace(values["MAX_BODY_BYTES"]); raw != "" {
		maxBodyBytes, err := strconv.ParseInt(raw, 10, 64)
		if err != nil {
			return Config{}, fmt.Errorf("MAX_BODY_BYTES must be an integer: %w", err)
		}
		if maxBodyBytes <= 0 {
			return Config{}, fmt.Errorf("MAX_BODY_BYTES must be greater than zero")
		}
		cfg.MaxBodyBytes = maxBodyBytes
	}

	return cfg, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

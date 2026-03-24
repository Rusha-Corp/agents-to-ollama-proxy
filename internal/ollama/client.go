package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Client struct {
	baseURL    *url.URL
	apiKey     string
	httpClient *http.Client
}

type ChatStream struct {
	decoder *json.Decoder
	body    io.ReadCloser
}

type UpstreamError struct {
	StatusCode int
	Message    string
}

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("upstream returned status %d", e.StatusCode)
}

func NewClient(baseURL, apiKey string, timeout time.Duration) (*Client, error) {
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return nil, fmt.Errorf("parse ollama base url: %w", err)
	}

	return &Client{
		baseURL: parsedURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, request ChatRequest) (ChatResponse, error) {
	var response ChatResponse
	if err := c.doJSON(ctx, http.MethodPost, "/chat", request, &response); err != nil {
		return ChatResponse{}, err
	}
	return response, nil
}

func (c *Client) ChatStream(ctx context.Context, request ChatRequest) (*ChatStream, error) {
	response, err := c.do(ctx, http.MethodPost, "/chat", request)
	if err != nil {
		return nil, err
	}

	return &ChatStream{
		decoder: json.NewDecoder(response.Body),
		body:    response.Body,
	}, nil
}

func (c *Client) ListModels(ctx context.Context) (TagsResponse, error) {
	var response TagsResponse
	if err := c.doJSON(ctx, http.MethodGet, "/tags", nil, &response); err != nil {
		return TagsResponse{}, err
	}
	return response, nil
}

func (s *ChatStream) Next() (ChatResponse, error) {
	var response ChatResponse
	if err := s.decoder.Decode(&response); err != nil {
		if errors.Is(err, io.EOF) {
			return ChatResponse{}, io.EOF
		}
		return ChatResponse{}, fmt.Errorf("decode upstream stream event: %w", err)
	}
	return response, nil
}

func (s *ChatStream) Close() error {
	return s.body.Close()
}

func (c *Client) doJSON(ctx context.Context, method, path string, requestBody any, responseBody any) error {
	response, err := c.do(ctx, method, path, requestBody)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if responseBody == nil {
		return nil
	}

	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(responseBody); err != nil {
		return fmt.Errorf("decode upstream response: %w", err)
	}
	return nil
}

func (c *Client) do(ctx context.Context, method, path string, requestBody any) (*http.Response, error) {
	var body io.Reader
	var payload []byte
	if requestBody != nil {
		encoded, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("marshal upstream request: %w", err)
		}
		payload = encoded
		body = bytes.NewReader(payload)
	}

	request, err := http.NewRequestWithContext(ctx, method, c.resolveURL(path), body)
	if err != nil {
		return nil, fmt.Errorf("create upstream request: %w", err)
	}
	request.Header.Set("Accept", "application/json")
	if requestBody != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	request.Header.Set("Authorization", "Bearer "+c.apiKey)

	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send upstream request: %w", err)
	}

	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		if len(payload) > 0 && response.StatusCode == http.StatusBadRequest {
			log.Printf("upstream_bad_request path=%s payload=%s", path, truncateForLog(payload, 8192))
		}
		defer response.Body.Close()
		return nil, parseUpstreamError(response)
	}

	return response, nil
}

func (c *Client) resolveURL(path string) string {
	resolved := *c.baseURL
	resolved.Path = strings.TrimRight(resolved.Path, "/") + "/" + strings.TrimLeft(path, "/")
	return resolved.String()
}

func parseUpstreamError(response *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	message := strings.TrimSpace(string(body))

	var ollamaErr ErrorResponse
	if err := json.Unmarshal(body, &ollamaErr); err == nil && strings.TrimSpace(ollamaErr.Error) != "" {
		message = strings.TrimSpace(ollamaErr.Error)
	}
	if message == "" {
		message = response.Status
	}

	return &UpstreamError{StatusCode: response.StatusCode, Message: message}
}

func truncateForLog(payload []byte, limit int) string {
	if len(payload) <= limit {
		return string(payload)
	}
	return string(payload[:limit]) + "...<truncated>"
}

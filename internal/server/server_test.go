package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ollama-proxy/internal/config"
	"ollama-proxy/internal/ollama"
	"ollama-proxy/internal/openai"
)

func TestHealthzDoesNotRequireAuth(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, nil)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if response.Header().Get("X-Request-Id") == "" {
		t.Fatal("expected X-Request-Id header")
	}
}

func TestChatCompletionsRequiresAuth(t *testing.T) {
	t.Parallel()

	handler := newTestHandler(t, nil)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/ollama/v1/chat/completions", bytes.NewBufferString(`{"model":"demo","messages":[{"role":"user","content":"hello"}]}`))

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusUnauthorized)
	}
}

func TestChatCompletionsTranslatesRequestAndResponse(t *testing.T) {
	t.Parallel()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Fatalf("path = %q, want /api/chat", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer upstream-key" {
			t.Fatalf("authorization = %q", r.Header.Get("Authorization"))
		}

		var request ollama.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		if request.Messages[0].Content != "hello" {
			t.Fatalf("content = %q, want hello", request.Messages[0].Content)
		}
		if request.Options["num_predict"] != float64(64) {
			t.Fatalf("num_predict = %v, want 64", request.Options["num_predict"])
		}

		writeJSON(w, http.StatusOK, ollama.ChatResponse{
			Model:           request.Model,
			CreatedAt:       "2026-03-24T00:00:00Z",
			Message:         ollama.Message{Role: "assistant", Content: "hello back"},
			DoneReason:      "stop",
			PromptEvalCount: 11,
			EvalCount:       7,
		})
	}))
	defer upstream.Close()

	handler := newTestHandler(t, upstream)
	maxTokens := 64
	body, _ := json.Marshal(openai.ChatCompletionRequest{
		Model:     "demo-model",
		Messages:  []openai.Message{{Role: "user", Content: "hello"}},
		MaxTokens: &maxTokens,
	})

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/ollama/v1/chat/completions", bytes.NewReader(body))
	request.Header.Set("Authorization", "Bearer proxy-token")

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		payload, _ := io.ReadAll(response.Body)
		t.Fatalf("status = %d, body = %s", response.Code, payload)
	}

	var result openai.ChatCompletionResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Object != "chat.completion" {
		t.Fatalf("object = %q", result.Object)
	}
	if result.Choices[0].Message.Content != "hello back" {
		t.Fatalf("content = %#v, want hello back", result.Choices[0].Message.Content)
	}
	if result.Usage.TotalTokens != 18 {
		t.Fatalf("total tokens = %d, want 18", result.Usage.TotalTokens)
	}
}

func TestModelsEndpoint(t *testing.T) {
	t.Parallel()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Fatalf("path = %q, want /api/tags", r.URL.Path)
		}
		writeJSON(w, http.StatusOK, ollama.TagsResponse{
			Models: []ollama.Model{{Name: "qwen3.5:35b", ModifiedAt: "2026-03-24T00:00:00Z"}},
		})
	}))
	defer upstream.Close()

	handler := newTestHandler(t, upstream)
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/ollama/v1/models", nil)
	request.Header.Set("Authorization", "Bearer proxy-token")

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}

	var result openai.ModelListResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(result.Data) != 1 || result.Data[0].ID != "qwen3.5:35b" {
		t.Fatalf("models = %#v", result.Data)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	t.Parallel()

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request ollama.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		if !request.Stream {
			t.Fatal("expected stream=true")
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"demo-model","created_at":"2026-03-24T00:00:00Z","message":{"role":"assistant","content":"hel"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"demo-model","created_at":"2026-03-24T00:00:01Z","message":{"role":"assistant","content":"lo"},"done":false}` + "\n"))
		_, _ = w.Write([]byte(`{"model":"demo-model","created_at":"2026-03-24T00:00:02Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2}` + "\n"))
	}))
	defer upstream.Close()

	handler := newTestHandler(t, upstream)
	body, _ := json.Marshal(openai.ChatCompletionRequest{
		Model:    "demo-model",
		Messages: []openai.Message{{Role: "user", Content: "hello"}},
		Stream:   true,
	})

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/ollama/v1/chat/completions", bytes.NewReader(body))
	request.Header.Set("Authorization", "Bearer proxy-token")

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusOK)
	}
	if got := response.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", got)
	}

	payload := response.Body.String()
	if !strings.Contains(payload, `"object":"chat.completion.chunk"`) {
		t.Fatalf("stream payload missing chat chunk: %s", payload)
	}
	if !strings.Contains(payload, `"content":"hel"`) || !strings.Contains(payload, `"content":"lo"`) {
		t.Fatalf("stream payload missing content chunks: %s", payload)
	}
	if !strings.Contains(payload, `"finish_reason":"stop"`) {
		t.Fatalf("stream payload missing finish reason: %s", payload)
	}
	if !strings.Contains(payload, `data: [DONE]`) {
		t.Fatalf("stream payload missing DONE marker: %s", payload)
	}
}

func TestRecoveryMiddlewareReturnsInternalServerError(t *testing.T) {
	t.Parallel()

	server := &Server{}
	handler := server.requestIDMiddleware(server.loggingMiddleware(server.recoverMiddleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("boom")
	}))))

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/panic", nil)

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusInternalServerError)
	}
	if response.Header().Get("X-Request-Id") == "" {
		t.Fatal("expected X-Request-Id header")
	}
	var result openai.ErrorEnvelope
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Error.Code != "internal_error" {
		t.Fatalf("error code = %q, want internal_error", result.Error.Code)
	}
}

func newTestHandler(t *testing.T, upstream *httptest.Server) http.Handler {
	t.Helper()

	baseURL := "https://ollama.example/api"
	if upstream != nil {
		baseURL = upstream.URL + "/api"
	}

	client, err := ollama.NewClient(baseURL, "upstream-key", 2*time.Second)
	if err != nil {
		t.Fatalf("NewClient() error = %v", err)
	}

	return New(config.Config{
		OllamaBaseURL:    baseURL,
		OllamaAPIKey:     "upstream-key",
		ProxyBearerToken: "proxy-token",
		DefaultModel:     "default-model",
		UpstreamTimeout:  2 * time.Second,
		MaxBodyBytes:     1 << 20,
	}, client).Handler()
}

package server

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime/debug"
	"time"

	"ollama-proxy/internal/config"
	"ollama-proxy/internal/ollama"
	"ollama-proxy/internal/openai"
	"ollama-proxy/internal/translate"
)

type contextKey string

const requestIDKey contextKey = "request_id"

type Server struct {
	config config.Config
	client *ollama.Client
	mux    *http.ServeMux
}

func New(cfg config.Config, client *ollama.Client) *Server {
	server := &Server{
		config: cfg,
		client: client,
		mux:    http.NewServeMux(),
	}
	server.routes()
	return server
}

func (s *Server) Handler() http.Handler {
	return s.requestIDMiddleware(s.loggingMiddleware(s.recoverMiddleware(s.mux)))
}

func (s *Server) routes() {
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})

	s.mux.Handle("GET /ollama/v1/models", requireBearerAuth(s.config.ProxyBearerToken, http.HandlerFunc(s.handleModels)))
	s.mux.Handle("POST /ollama/v1/chat/completions", requireBearerAuth(s.config.ProxyBearerToken, http.HandlerFunc(s.handleChatCompletions)))
	s.mux.Handle("POST /ollama/v1/completions", requireBearerAuth(s.config.ProxyBearerToken, http.HandlerFunc(s.handleCompletions)))
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	response, err := s.client.ListModels(r.Context())
	if err != nil {
		s.writeProxyError(w, r, err)
		return
	}

	writeJSON(w, http.StatusOK, translate.OllamaModelsToOpenAI(response))
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var request openai.ChatCompletionRequest
	if err := s.decodeRequest(w, r, &request); err != nil {
		return
	}

	upstreamRequest, err := translate.ChatCompletionToOllama(request, s.config.DefaultModel)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request")
		return
	}

	if request.Stream {
		s.handleChatCompletionsStream(w, r, upstreamRequest)
		return
	}

	response, err := s.client.Chat(r.Context(), upstreamRequest)
	if err != nil {
		s.writeProxyError(w, r, err)
		return
	}

	writeJSON(w, http.StatusOK, translate.OllamaToChatCompletion(response, upstreamRequest.Model))
}

func (s *Server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var request openai.CompletionRequest
	if err := s.decodeRequest(w, r, &request); err != nil {
		return
	}

	chatRequest, err := translate.CompletionToChatCompletion(request, s.config.DefaultModel)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request")
		return
	}

	upstreamRequest, err := translate.ChatCompletionToOllama(chatRequest, s.config.DefaultModel)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request")
		return
	}

	if request.Stream {
		s.handleCompletionsStream(w, r, upstreamRequest)
		return
	}

	response, err := s.client.Chat(r.Context(), upstreamRequest)
	if err != nil {
		s.writeProxyError(w, r, err)
		return
	}

	writeJSON(w, http.StatusOK, translate.OllamaToCompletion(response, upstreamRequest.Model))
}

func (s *Server) handleChatCompletionsStream(w http.ResponseWriter, r *http.Request, upstreamRequest ollama.ChatRequest) {
	stream, err := s.client.ChatStream(r.Context(), upstreamRequest)
	if err != nil {
		s.writeProxyError(w, r, err)
		return
	}
	defer stream.Close()

	streamID := translate.NewID("chatcmpl")
	if err := startSSE(w); err != nil {
		logRequestError(r, "start_sse_failed err=%v", err)
		writeError(w, http.StatusInternalServerError, err.Error(), "server_error", "streaming_unavailable")
		return
	}
	if err := writeSSE(w, translate.InitialChatCompletionChunk(streamID, upstreamRequest.Model)); err != nil {
		logRequestError(r, "initial_stream_chunk_failed err=%v", err)
		return
	}

	includeRole := false
	sawToolCalls := false
	for {
		response, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			logRequestError(r, "upstream_stream_read_failed err=%v", err)
			return
		}

		chunk := translate.OllamaToChatCompletionChunk(response, streamID, upstreamRequest.Model, includeRole)
		if len(chunk.Choices[0].Delta.ToolCalls) > 0 {
			sawToolCalls = true
		}
		if sawToolCalls && chunk.Choices[0].FinishReason != nil && *chunk.Choices[0].FinishReason == "stop" {
			finishReason := "tool_calls"
			chunk.Choices[0].FinishReason = &finishReason
		}
		if chunk.Choices[0].FinishReason == nil && chunk.Choices[0].Delta.Role == "" && chunk.Choices[0].Delta.Content == "" && len(chunk.Choices[0].Delta.ToolCalls) == 0 {
			continue
		}
		includeRole = false
		if err := writeSSE(w, chunk); err != nil {
			logRequestError(r, "stream_write_failed err=%v", err)
			return
		}
	}

	if err := writeSSEDone(w); err != nil {
		logRequestError(r, "stream_done_failed err=%v", err)
	}
}

func (s *Server) handleCompletionsStream(w http.ResponseWriter, r *http.Request, upstreamRequest ollama.ChatRequest) {
	stream, err := s.client.ChatStream(r.Context(), upstreamRequest)
	if err != nil {
		s.writeProxyError(w, r, err)
		return
	}
	defer stream.Close()

	streamID := translate.NewID("cmpl")
	if err := startSSE(w); err != nil {
		logRequestError(r, "start_sse_failed err=%v", err)
		writeError(w, http.StatusInternalServerError, err.Error(), "server_error", "streaming_unavailable")
		return
	}

	for {
		response, err := stream.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			logRequestError(r, "upstream_stream_read_failed err=%v", err)
			return
		}

		chunk := translate.OllamaToCompletionChunk(response, streamID, upstreamRequest.Model)
		if chunk.Choices[0].FinishReason == nil && chunk.Choices[0].Text == "" {
			continue
		}
		if err := writeSSE(w, chunk); err != nil {
			logRequestError(r, "stream_write_failed err=%v", err)
			return
		}
	}

	if err := writeSSEDone(w); err != nil {
		logRequestError(r, "stream_done_failed err=%v", err)
	}
}

func (s *Server) decodeRequest(w http.ResponseWriter, r *http.Request, target any) error {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, s.config.MaxBodyBytes))
	if err := decoder.Decode(target); err != nil {
		var syntaxError *json.SyntaxError
		switch {
		case errors.Is(err, io.EOF):
			writeError(w, http.StatusBadRequest, "request body is required", "invalid_request_error", "invalid_json")
		case errors.As(err, &syntaxError):
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON at byte %d", syntaxError.Offset), "invalid_request_error", "invalid_json")
		default:
			writeError(w, http.StatusBadRequest, "invalid JSON body", "invalid_request_error", "invalid_json")
		}
		return err
	}

	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		writeError(w, http.StatusBadRequest, "request body must contain a single JSON object", "invalid_request_error", "invalid_json")
		return fmt.Errorf("multiple JSON values in request body")
	}
	return nil
}

func (s *Server) writeProxyError(w http.ResponseWriter, r *http.Request, err error) {
	var upstreamErr *ollama.UpstreamError
	if errors.As(err, &upstreamErr) {
		status := upstreamErr.StatusCode
		message := upstreamErr.Message
		code := "upstream_error"

		if status == http.StatusUnauthorized || status == http.StatusForbidden {
			status = http.StatusBadGateway
			message = "upstream authentication failed"
			code = "bad_gateway"
		} else if status >= http.StatusInternalServerError {
			status = http.StatusBadGateway
			message = "upstream service error"
			code = "bad_gateway"
		}

		logRequestError(r, "upstream_error status=%d code=%s message=%q", upstreamErr.StatusCode, code, upstreamErr.Message)
		writeError(w, status, message, "api_error", code)
		return
	}

	logRequestError(r, "proxy_error err=%v", err)
	writeError(w, http.StatusBadGateway, "upstream request failed", "api_error", "bad_gateway")
}

func (s *Server) requestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestID := r.Header.Get("X-Request-Id")
		if requestID == "" {
			requestID = translate.NewID("req")
		}
		w.Header().Set("X-Request-Id", requestID)
		next.ServeHTTP(w, r.WithContext(context.WithValue(r.Context(), requestIDKey, requestID)))
	})
}

func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		startedAt := time.Now()
		responseWriter := &responseRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(responseWriter, r)
		log.Printf(
			"request_id=%s method=%s path=%s status=%d bytes=%d duration_ms=%d remote=%s ua=%q",
			requestIDFromContext(r.Context()),
			r.Method,
			r.URL.Path,
			responseWriter.status,
			responseWriter.bytes,
			time.Since(startedAt).Milliseconds(),
			r.RemoteAddr,
			r.UserAgent(),
		)
	})
}

func (s *Server) recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if recovered := recover(); recovered != nil {
				log.Printf(
					"request_id=%s panic method=%s path=%s err=%v stack=%q",
					requestIDFromContext(r.Context()),
					r.Method,
					r.URL.Path,
					recovered,
					string(debug.Stack()),
				)
				if responseWriter, ok := w.(*responseRecorder); ok && responseWriter.wroteHeader {
					return
				}
				writeError(w, http.StatusInternalServerError, "internal server error", "server_error", "internal_error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message, errorType, code string) {
	writeJSON(w, status, openai.ErrorEnvelope{Error: openai.ErrorResponse{Message: message, Type: errorType, Code: code}})
}

func startSSE(w http.ResponseWriter) error {
	if _, ok := w.(http.Flusher); !ok {
		return fmt.Errorf("streaming is not supported by the response writer")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	w.(http.Flusher).Flush()
	return nil
}

type responseRecorder struct {
	http.ResponseWriter
	status      int
	bytes       int
	wroteHeader bool
}

func (r *responseRecorder) WriteHeader(status int) {
	r.status = status
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(status)
}

func (r *responseRecorder) Write(payload []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	count, err := r.ResponseWriter.Write(payload)
	r.bytes += count
	return count, err
}

func (r *responseRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func requestIDFromContext(ctx context.Context) string {
	requestID, _ := ctx.Value(requestIDKey).(string)
	return requestID
}

func logRequestError(r *http.Request, format string, args ...any) {
	prefix := fmt.Sprintf("request_id=%s method=%s path=%s ", requestIDFromContext(r.Context()), r.Method, r.URL.Path)
	log.Printf(prefix+format, args...)
}

func writeSSE(w http.ResponseWriter, payload any) error {
	writer := bufio.NewWriter(w)
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := writer.WriteString("data: "); err != nil {
		return err
	}
	if _, err := writer.Write(encoded); err != nil {
		return err
	}
	if _, err := writer.WriteString("\n\n"); err != nil {
		return err
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	w.(http.Flusher).Flush()
	return nil
}

func writeSSEDone(w http.ResponseWriter) error {
	writer := bufio.NewWriter(w)
	if _, err := writer.WriteString("data: [DONE]\n\n"); err != nil {
		return err
	}
	if err := writer.Flush(); err != nil {
		return err
	}
	w.(http.Flusher).Flush()
	return nil
}

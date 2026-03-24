package translate

import (
	"encoding/json"
	"testing"

	"ollama-proxy/internal/ollama"
	"ollama-proxy/internal/openai"
)

func TestChatCompletionToOllamaMultipartContent(t *testing.T) {
	t.Parallel()

	temperature := 0.3
	maxTokens := 128
	request := openai.ChatCompletionRequest{
		Model: "qwen3.5:35b",
		Messages: []openai.Message{{
			Role: "user",
			Content: []any{
				map[string]any{"type": "text", "text": "hello "},
				map[string]any{"type": "text", "text": "world"},
			},
		}},
		Temperature: &temperature,
		MaxTokens:   &maxTokens,
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}

	if translated.Messages[0].Content != "hello world" {
		t.Fatalf("content = %q, want %q", translated.Messages[0].Content, "hello world")
	}
	if translated.Options["num_predict"] != 128 {
		t.Fatalf("num_predict = %v, want 128", translated.Options["num_predict"])
	}
}

func TestChatCompletionToOllamaNormalizesAliasedModel(t *testing.T) {
	t.Parallel()

	request := openai.ChatCompletionRequest{
		Model:    "qwen3.5:35b",
		Messages: []openai.Message{{Role: "user", Content: "hello"}},
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}
	if translated.Model != "qwen3.5:397b" {
		t.Fatalf("model = %q, want qwen3.5:397b", translated.Model)
	}
}

func TestChatCompletionToOllamaPreservesTools(t *testing.T) {
	t.Parallel()

	request := openai.ChatCompletionRequest{
		Model:    "qwen3.5:35b",
		Messages: []openai.Message{{Role: "user", Content: "hello"}},
		Tools:    json.RawMessage(`[{"type":"function"}]`),
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}
	if string(translated.Tools) != `[{"type":"function"}]` {
		t.Fatalf("tools = %s", translated.Tools)
	}
}

func TestChatCompletionToOllamaPreservesStream(t *testing.T) {
	t.Parallel()

	request := openai.ChatCompletionRequest{
		Model:    "qwen3.5:35b",
		Messages: []openai.Message{{Role: "user", Content: "hello"}},
		Stream:   true,
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}
	if !translated.Stream {
		t.Fatal("expected stream=true")
	}
}

func TestCompletionToChatCompletion(t *testing.T) {
	t.Parallel()

	request := openai.CompletionRequest{
		Prompt: []any{"first", "second"},
	}

	translated, err := CompletionToChatCompletion(request, "fallback-model")
	if err != nil {
		t.Fatalf("CompletionToChatCompletion() error = %v", err)
	}

	if translated.Model != "fallback-model" {
		t.Fatalf("model = %q, want fallback-model", translated.Model)
	}
	if translated.Messages[0].Content != "first\nsecond" {
		t.Fatalf("content = %#v, want joined prompt", translated.Messages[0].Content)
	}
}

func TestOllamaToChatCompletionChunk(t *testing.T) {
	t.Parallel()

	chunk := OllamaToChatCompletionChunk(openaiToOllamaResponse("assistant", "hel", false, ""), "chatcmpl-1", "demo", true)
	if chunk.Object != "chat.completion.chunk" {
		t.Fatalf("object = %q", chunk.Object)
	}
	if chunk.Choices[0].Delta.Role != "assistant" {
		t.Fatalf("delta role = %#v", chunk.Choices[0].Delta)
	}
	if chunk.Choices[0].Delta.Content != "hel" {
		t.Fatalf("delta content = %#v", chunk.Choices[0].Delta.Content)
	}

	finalChunk := OllamaToChatCompletionChunk(openaiToOllamaResponse("assistant", "", true, "stop"), "chatcmpl-1", "demo", false)
	if finalChunk.Choices[0].FinishReason == nil || *finalChunk.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish reason = %#v", finalChunk.Choices[0].FinishReason)
	}
}

func TestOllamaToChatCompletionUsesToolFinishReason(t *testing.T) {
	t.Parallel()

	response := openaiToOllamaResponse("assistant", "", true, "stop")
	response.Message.ToolCalls = json.RawMessage(`[{"type":"function"}]`)
	translated := OllamaToChatCompletion(response, "demo")
	if translated.Choices[0].FinishReason == nil || *translated.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %#v", translated.Choices[0].FinishReason)
	}
	if string(translated.Choices[0].Message.ToolCalls) != `[{"type":"function"}]` {
		t.Fatalf("tool calls = %s", translated.Choices[0].Message.ToolCalls)
	}
}

func TestOllamaToChatCompletionPreservesRequestedAliasModel(t *testing.T) {
	t.Parallel()

	response := openaiToOllamaResponse("assistant", "OK", true, "stop")
	response.Model = "qwen3.5:397b"
	translated := OllamaToChatCompletion(response, "qwen3.5:35b")
	if translated.Model != "qwen3.5:35b" {
		t.Fatalf("model = %q, want qwen3.5:35b", translated.Model)
	}
}

func TestOllamaModelsToOpenAIIncludesAliases(t *testing.T) {
	t.Parallel()

	models := OllamaModelsToOpenAI(ollama.TagsResponse{Models: []ollama.Model{{Name: "qwen3.5:397b"}, {Name: "qwen3-coder:480b"}}})
	ids := map[string]bool{}
	for _, model := range models.Data {
		ids[model.ID] = true
	}
	for _, id := range []string{"qwen3.5:397b", "qwen3.5:27b", "qwen3.5:35b", "qwen3-coder:480b", "qwen2.5-coder:32b", "qwen3-coder:480b-cloud"} {
		if !ids[id] {
			t.Fatalf("expected model id %q in list", id)
		}
	}
}

func openaiToOllamaResponse(role, content string, done bool, doneReason string) ollama.ChatResponse {
	return ollama.ChatResponse{
		Model:      "demo",
		CreatedAt:  "2026-03-24T00:00:00Z",
		Message:    ollama.Message{Role: role, Content: content},
		Done:       done,
		DoneReason: doneReason,
	}
}

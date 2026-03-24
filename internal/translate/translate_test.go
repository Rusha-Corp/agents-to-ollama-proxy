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

func TestChatCompletionToOllamaConvertsAssistantToolCalls(t *testing.T) {
	t.Parallel()

	request := openai.ChatCompletionRequest{
		Model: "qwen3.5:35b",
		Messages: []openai.Message{{
			Role:      "assistant",
			Content:   "",
			ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}}]`),
		}},
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}

	var toolCalls []struct {
		Function struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(translated.Messages[0].ToolCalls, &toolCalls); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("len(toolCalls) = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("name = %q, want get_weather", toolCalls[0].Function.Name)
	}
	if toolCalls[0].Function.Arguments["city"] != "Tokyo" {
		t.Fatalf("arguments = %#v, want city=Tokyo", toolCalls[0].Function.Arguments)
	}
}

func TestChatCompletionToOllamaMapsToolResultToToolName(t *testing.T) {
	t.Parallel()

	request := openai.ChatCompletionRequest{
		Model: "qwen3.5:35b",
		Messages: []openai.Message{
			{
				Role:      "assistant",
				Content:   "",
				ToolCalls: json.RawMessage(`[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}}]`),
			},
			{
				Role:       "tool",
				Content:    "11 degrees celsius",
				ToolCallID: "call_1",
			},
		},
	}

	translated, err := ChatCompletionToOllama(request, "")
	if err != nil {
		t.Fatalf("ChatCompletionToOllama() error = %v", err)
	}

	if translated.Messages[1].Role != "tool" {
		t.Fatalf("role = %q, want tool", translated.Messages[1].Role)
	}
	if translated.Messages[1].ToolName != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", translated.Messages[1].ToolName)
	}
	if translated.Messages[1].Content != "11 degrees celsius" {
		t.Fatalf("content = %q, want tool output", translated.Messages[1].Content)
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
	response.Message.ToolCalls = json.RawMessage(`[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]`)
	translated := OllamaToChatCompletion(response, "demo")
	if translated.Choices[0].FinishReason == nil || *translated.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish reason = %#v", translated.Choices[0].FinishReason)
	}

	var toolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(translated.Choices[0].Message.ToolCalls, &toolCalls); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("len(toolCalls) = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].ID == "" {
		t.Fatal("expected generated tool call id")
	}
	if toolCalls[0].Type != "function" {
		t.Fatalf("type = %q, want function", toolCalls[0].Type)
	}
	if toolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("name = %q, want get_weather", toolCalls[0].Function.Name)
	}
	if toolCalls[0].Function.Arguments != `{"city":"Tokyo"}` {
		t.Fatalf("arguments = %q, want {\"city\":\"Tokyo\"}", toolCalls[0].Function.Arguments)
	}
}

func TestOllamaToChatCompletionChunkConvertsToolCalls(t *testing.T) {
	t.Parallel()

	response := openaiToOllamaResponse("assistant", "", false, "")
	response.Message.ToolCalls = json.RawMessage(`[{"function":{"name":"get_weather","arguments":{"city":"Tokyo"}}}]`)

	chunk := OllamaToChatCompletionChunk(response, "chatcmpl-1", "demo", false)

	var toolCalls []struct {
		Index    int    `json:"index"`
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(chunk.Choices[0].Delta.ToolCalls, &toolCalls); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(toolCalls) != 1 {
		t.Fatalf("len(toolCalls) = %d, want 1", len(toolCalls))
	}
	if toolCalls[0].Index != 0 {
		t.Fatalf("index = %d, want 0", toolCalls[0].Index)
	}
	if toolCalls[0].ID != "call_chatcmpl-1_0" {
		t.Fatalf("id = %q, want call_chatcmpl-1_0", toolCalls[0].ID)
	}
	if toolCalls[0].Type != "function" {
		t.Fatalf("type = %q, want function", toolCalls[0].Type)
	}
	if toolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("name = %q, want get_weather", toolCalls[0].Function.Name)
	}
	if toolCalls[0].Function.Arguments != `{"city":"Tokyo"}` {
		t.Fatalf("arguments = %q, want {\"city\":\"Tokyo\"}", toolCalls[0].Function.Arguments)
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

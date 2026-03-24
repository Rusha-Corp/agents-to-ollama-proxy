package translate

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"ollama-proxy/internal/ollama"
	"ollama-proxy/internal/openai"
)

var modelAliases = map[string]string{
	"qwen3-coder:480b-cloud": "qwen3-coder:480b",
	"qwen2.5-coder:32b":      "qwen3-coder:480b",
	"qwen3.5:27b":            "qwen3.5:397b",
	"qwen3.5:35b":            "qwen3.5:397b",
	"minimax-m2.5:cloud":     "minimax-m2.5",
	"kimi-k2.5:cloud":        "kimi-k2.5",
}

func ChatCompletionToOllama(request openai.ChatCompletionRequest, defaultModel string) (ollama.ChatRequest, error) {
	model := normalizeModel(firstNonEmpty(request.Model, defaultModel))
	if model == "" {
		return ollama.ChatRequest{}, fmt.Errorf("model is required")
	}

	messages := make([]ollama.Message, 0, len(request.Messages))
	for _, message := range request.Messages {
		content, err := messageText(message.Content)
		if err != nil {
			return ollama.ChatRequest{}, fmt.Errorf("message content: %w", err)
		}
		messages = append(messages, ollama.Message{
			Role:       message.Role,
			Content:    content,
			ToolCalls:  message.ToolCalls,
			ToolCallID: message.ToolCallID,
		})
	}
	if len(messages) == 0 {
		return ollama.ChatRequest{}, fmt.Errorf("at least one message is required")
	}

	return ollama.ChatRequest{
		Model:      model,
		Messages:   messages,
		Stream:     request.Stream,
		Tools:      request.Tools,
		ToolChoice: request.ToolChoice,
		Options:    buildOptions(request.Temperature, request.TopP, request.MaxTokens),
	}, nil
}

func CompletionToChatCompletion(request openai.CompletionRequest, defaultModel string) (openai.ChatCompletionRequest, error) {
	if request.Suffix != "" {
		return openai.ChatCompletionRequest{}, fmt.Errorf("suffix is not supported")
	}

	prompt, err := promptText(request.Prompt)
	if err != nil {
		return openai.ChatCompletionRequest{}, err
	}

	return openai.ChatCompletionRequest{
		Model: normalizeModel(firstNonEmpty(request.Model, defaultModel)),
		Messages: []openai.Message{{
			Role:    "user",
			Content: prompt,
		}},
		Temperature: request.Temperature,
		TopP:        request.TopP,
		MaxTokens:   request.MaxTokens,
		Stream:      request.Stream,
		Tools:       request.Tools,
		ToolChoice:  request.ToolChoice,
	}, nil
}

func OllamaToChatCompletion(response ollama.ChatResponse, requestedModel string) openai.ChatCompletionResponse {
	model := responseModel(requestedModel, response.Model)
	created := parseUnix(response.CreatedAt)
	finishReason := mapFinishReason(response.DoneReason)
	usage := openai.Usage{
		PromptTokens:     response.PromptEvalCount,
		CompletionTokens: response.EvalCount,
		TotalTokens:      response.PromptEvalCount + response.EvalCount,
	}

	return openai.ChatCompletionResponse{
		ID:      NewID("chatcmpl"),
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []openai.ChatChoice{{
			Index: 0,
			Message: openai.Message{
				Role:       firstNonEmpty(response.Message.Role, "assistant"),
				Content:    response.Message.Content,
				ToolCalls:  response.Message.ToolCalls,
				ToolCallID: response.Message.ToolCallID,
			},
			FinishReason: stringPointer(chatFinishReason(finishReason, response.Message.ToolCalls)),
		}},
		Usage: usage,
	}
}

func OllamaToCompletion(response ollama.ChatResponse, requestedModel string) openai.CompletionResponse {
	model := responseModel(requestedModel, response.Model)
	created := parseUnix(response.CreatedAt)
	finishReason := mapFinishReason(response.DoneReason)
	usage := openai.Usage{
		PromptTokens:     response.PromptEvalCount,
		CompletionTokens: response.EvalCount,
		TotalTokens:      response.PromptEvalCount + response.EvalCount,
	}

	return openai.CompletionResponse{
		ID:      NewID("cmpl"),
		Object:  "text_completion",
		Created: created,
		Model:   model,
		Choices: []openai.CompletionChoice{{
			Index:        0,
			Text:         response.Message.Content,
			FinishReason: stringPointer(finishReason),
		}},
		Usage: usage,
	}
}

func OllamaToChatCompletionChunk(response ollama.ChatResponse, id, requestedModel string, includeRole bool) openai.ChatCompletionChunkResponse {
	model := responseModel(requestedModel, response.Model)
	created := parseUnix(response.CreatedAt)
	var finishReason *string
	if response.Done || response.DoneReason != "" {
		finishReason = stringPointer(mapFinishReason(response.DoneReason))
	}

	delta := openai.ChunkDelta{}
	if includeRole {
		delta.Role = firstNonEmpty(response.Message.Role, "assistant")
	}
	if response.Message.Content != "" {
		delta.Content = response.Message.Content
	}
	if len(response.Message.ToolCalls) > 0 {
		delta.ToolCalls = response.Message.ToolCalls
	}
	if finishReason != nil {
		value := chatFinishReason(*finishReason, response.Message.ToolCalls)
		finishReason = stringPointer(value)
	}

	return openai.ChatCompletionChunkResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []openai.ChatChunkChoice{{
			Index:        0,
			Delta:        delta,
			FinishReason: finishReason,
		}},
	}
}

func InitialChatCompletionChunk(id, requestedModel string) openai.ChatCompletionChunkResponse {
	return openai.ChatCompletionChunkResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   responseModel(requestedModel, ""),
		Choices: []openai.ChatChunkChoice{{
			Index: 0,
			Delta: openai.ChunkDelta{Role: "assistant"},
		}},
	}
}

func OllamaToCompletionChunk(response ollama.ChatResponse, id, requestedModel string) openai.CompletionChunkResponse {
	model := responseModel(requestedModel, response.Model)
	created := parseUnix(response.CreatedAt)
	var finishReason *string
	if response.Done || response.DoneReason != "" {
		finishReason = stringPointer(mapFinishReason(response.DoneReason))
	}

	return openai.CompletionChunkResponse{
		ID:      id,
		Object:  "text_completion",
		Created: created,
		Model:   model,
		Choices: []openai.CompletionChunkChoice{{
			Index:        0,
			Text:         response.Message.Content,
			FinishReason: finishReason,
		}},
	}
}

func OllamaModelsToOpenAI(response ollama.TagsResponse) openai.ModelListResponse {
	models := make([]openai.ModelInfo, 0, len(response.Models)+len(modelAliases))
	seen := map[string]struct{}{}
	for _, model := range response.Models {
		created := parseUnix(model.ModifiedAt)
		modelID := firstNonEmpty(model.Model, model.Name)
		models = append(models, openai.ModelInfo{
			ID:      modelID,
			Object:  "model",
			Created: created,
			OwnedBy: "ollama",
		})
		seen[modelID] = struct{}{}
		for alias, target := range modelAliases {
			if target != modelID {
				continue
			}
			if _, exists := seen[alias]; exists {
				continue
			}
			models = append(models, openai.ModelInfo{
				ID:      alias,
				Object:  "model",
				Created: created,
				OwnedBy: "ollama",
			})
			seen[alias] = struct{}{}
		}
	}

	return openai.ModelListResponse{Object: "list", Data: models}
}

func buildOptions(temperature, topP *float64, maxTokens *int) map[string]any {
	options := map[string]any{}
	if temperature != nil {
		options["temperature"] = *temperature
	}
	if topP != nil {
		options["top_p"] = *topP
	}
	if maxTokens != nil {
		options["num_predict"] = *maxTokens
	}
	if len(options) == 0 {
		return nil
	}
	return options
}

func messageText(content any) (string, error) {
	switch value := content.(type) {
	case nil:
		return "", nil
	case string:
		return value, nil
	case []any:
		var builder strings.Builder
		for _, rawPart := range value {
			part, ok := rawPart.(map[string]any)
			if !ok {
				return "", fmt.Errorf("unsupported multipart message content")
			}
			partType, _ := part["type"].(string)
			if partType != "" && partType != "text" {
				return "", fmt.Errorf("unsupported message part type %q", partType)
			}
			text, ok := part["text"].(string)
			if !ok {
				return "", fmt.Errorf("text message part is missing text")
			}
			builder.WriteString(text)
		}
		return builder.String(), nil
	default:
		return "", fmt.Errorf("unsupported content type %T", content)
	}
}

func promptText(prompt any) (string, error) {
	switch value := prompt.(type) {
	case string:
		return value, nil
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			text, ok := item.(string)
			if !ok {
				return "", fmt.Errorf("prompt array must only contain strings")
			}
			parts = append(parts, text)
		}
		return strings.Join(parts, "\n"), nil
	default:
		return "", fmt.Errorf("prompt must be a string or string array")
	}
}

func mapFinishReason(reason string) string {
	switch reason {
	case "length":
		return "length"
	case "stop", "":
		return "stop"
	default:
		return reason
	}
}

func parseUnix(value string) int64 {
	if value == "" {
		return time.Now().Unix()
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Now().Unix()
	}
	return parsed.Unix()
}

func NewID(prefix string) string {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return prefix + "-" + hex.EncodeToString(bytes)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func stringPointer(value string) *string {
	return &value
}

func chatFinishReason(reason string, toolCalls []byte) string {
	if len(toolCalls) > 0 {
		return "tool_calls"
	}
	return reason
}

func normalizeModel(model string) string {
	if target, ok := modelAliases[model]; ok {
		return target
	}
	return model
}

func responseModel(requestedModel, upstreamModel string) string {
	if requestedModel != "" {
		return requestedModel
	}
	return firstNonEmpty(upstreamModel, requestedModel)
}

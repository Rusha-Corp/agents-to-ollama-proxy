package ollama

import "encoding/json"

type Message struct {
	Role       string          `json:"role"`
	Content    string          `json:"content"`
	ToolCalls  json.RawMessage `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

type ChatRequest struct {
	Model      string          `json:"model"`
	Messages   []Message       `json:"messages"`
	Stream     bool            `json:"stream"`
	Tools      json.RawMessage `json:"tools,omitempty"`
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
	Options    map[string]any  `json:"options,omitempty"`
}

type ChatResponse struct {
	Model           string  `json:"model"`
	CreatedAt       string  `json:"created_at,omitempty"`
	Message         Message `json:"message"`
	Done            bool    `json:"done,omitempty"`
	DoneReason      string  `json:"done_reason,omitempty"`
	PromptEvalCount int     `json:"prompt_eval_count,omitempty"`
	EvalCount       int     `json:"eval_count,omitempty"`
}

type TagsResponse struct {
	Models []Model `json:"models"`
}

type Model struct {
	Name       string `json:"name"`
	Model      string `json:"model,omitempty"`
	ModifiedAt string `json:"modified_at,omitempty"`
	Size       int64  `json:"size,omitempty"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}

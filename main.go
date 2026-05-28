package main

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

//go:embed static/*
var staticFiles embed.FS

//go:embed templates/*
var templateFiles embed.FS

// Settings persisted to disk.
type Settings struct {
	SearxngURL string `json:"searxng_url"`
	LLMURL     string `json:"llm_url"`
	LLMAPIKey  string `json:"llm_api_key"`
	LLMModel   string `json:"llm_model"`
}

var (
	settings   Settings
	settingsMu sync.RWMutex
	configPath string
	templates  *template.Template
)

func loadSettings() error {
	settingsMu.Lock()
	defer settingsMu.Unlock()

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			settings = Settings{}
			return nil
		}
		return err
	}
	return json.Unmarshal(data, &settings)
}

func saveSettings() error {
	settingsMu.RLock()
	data, err := json.MarshalIndent(settings, "", "  ")
	settingsMu.RUnlock()
	if err != nil {
		return err
	}
	return os.WriteFile(configPath, data, 0644)
}

func getSettings() Settings {
	settingsMu.RLock()
	defer settingsMu.RUnlock()
	return settings
}

// SearxNG response types.
type SearxResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Content string `json:"content"`
}

type SearxResponse struct {
	Results []SearxResult `json:"results"`
}

func searchSearxng(query string, cfg Settings) ([]SearxResult, error) {
	if cfg.SearxngURL == "" {
		return nil, fmt.Errorf("SearxNG URL not configured")
	}

	base := strings.TrimRight(cfg.SearxngURL, "/")
	u := fmt.Sprintf("%s/search?q=%s&format=json&categories=general&language=en", base, url.QueryEscape(query))

	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, fmt.Errorf("searxng request build failed: %w", err)
	}

	// If using an internal OpenHost URL (host.containers.internal), set the
	// Host header from the configured URL so the router routes correctly.
	// Also try the OPENHOST_ROUTER_URL as the transport when the configured
	// URL points to an external domain that the container can't reach directly.
	parsedURL, _ := url.Parse(cfg.SearxngURL)
	routerURL := os.Getenv("OPENHOST_ROUTER_URL")
	if routerURL != "" && parsedURL != nil && parsedURL.Host != "" {
		// Rewrite the request to go through the internal router, preserving the Host header.
		internalBase := strings.TrimRight(routerURL, "/")
		internalURL := fmt.Sprintf("%s/search?q=%s&format=json&categories=general&language=en", internalBase, url.QueryEscape(query))
		req, err = http.NewRequest("GET", internalURL, nil)
		if err != nil {
			return nil, fmt.Errorf("searxng internal request build failed: %w", err)
		}
		req.Host = parsedURL.Host
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("searxng request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("searxng returned %d: %s", resp.StatusCode, string(body[:min(len(body), 200)]))
	}

	var sr SearxResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, fmt.Errorf("searxng decode error: %w", err)
	}

	limit := 8
	if len(sr.Results) < limit {
		limit = len(sr.Results)
	}
	return sr.Results[:limit], nil
}

func buildPrompt(query string, results []SearxResult) []map[string]string {
	var contextBuf strings.Builder
	for i, r := range results {
		contextBuf.WriteString(fmt.Sprintf("[%d] %s\n%s\n%s\n\n", i+1, r.Title, r.URL, r.Content))
	}

	systemMsg := `You are a helpful search assistant. Answer the user's question using the provided search results. 
Cite sources using [1], [2], etc. corresponding to the numbered results below.
Be concise, accurate, and well-structured. Use markdown formatting.
If the search results don't contain enough information to answer, say so honestly.

Search Results:
` + contextBuf.String()

	return []map[string]string{
		{"role": "system", "content": systemMsg},
		{"role": "user", "content": query},
	}
}

// OpenAI-compatible streaming types.
type ChatRequest struct {
	Model    string              `json:"model"`
	Messages []map[string]string `json:"messages"`
	Stream   bool                `json:"stream"`
}

type ChatChunkChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
}

type ChatChunk struct {
	Choices []ChatChunkChoice `json:"choices"`
}

func handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	cfg := getSettings()
	configured := cfg.LLMURL != "" && cfg.LLMModel != "" && cfg.SearxngURL != ""
	templates.ExecuteTemplate(w, "index.html", map[string]any{
		"Configured": configured,
	})
}

func handleSettings(w http.ResponseWriter, r *http.Request) {
	cfg := getSettings()
	// Mask the API key for display.
	maskedKey := ""
	if cfg.LLMAPIKey != "" {
		if len(cfg.LLMAPIKey) > 8 {
			maskedKey = cfg.LLMAPIKey[:4] + "..." + cfg.LLMAPIKey[len(cfg.LLMAPIKey)-4:]
		} else {
			maskedKey = "****"
		}
	}
	templates.ExecuteTemplate(w, "settings.html", map[string]any{
		"Settings":  cfg,
		"MaskedKey": maskedKey,
	})
}

func handleSettingsSave(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}

	settingsMu.Lock()
	settings.SearxngURL = strings.TrimSpace(r.FormValue("searxng_url"))
	settings.LLMURL = strings.TrimSpace(r.FormValue("llm_url"))
	newKey := strings.TrimSpace(r.FormValue("llm_api_key"))
	if newKey != "" {
		settings.LLMAPIKey = newKey
	}
	settings.LLMModel = strings.TrimSpace(r.FormValue("llm_model"))
	settingsMu.Unlock()

	if err := saveSettings(); err != nil {
		log.Printf("failed to save settings: %v", err)
		http.Error(w, "save failed", http.StatusInternalServerError)
		return
	}

	w.Header().Set("HX-Redirect", "/settings")
	w.WriteHeader(http.StatusOK)
	fmt.Fprint(w, `<p style="color: #4ade80;">Settings saved.</p>`)
}

func handleSearch(w http.ResponseWriter, r *http.Request) {
	query := strings.TrimSpace(r.FormValue("q"))
	if query == "" {
		http.Error(w, "empty query", http.StatusBadRequest)
		return
	}

	cfg := getSettings()
	if cfg.SearxngURL == "" || cfg.LLMURL == "" || cfg.LLMModel == "" {
		templates.ExecuteTemplate(w, "error.html", map[string]string{
			"Error": "Please configure settings first.",
		})
		return
	}

	// Search SearxNG.
	results, err := searchSearxng(query, cfg)
	if err != nil {
		templates.ExecuteTemplate(w, "error.html", map[string]string{
			"Error": fmt.Sprintf("Search failed: %v", err),
		})
		return
	}

	if len(results) == 0 {
		templates.ExecuteTemplate(w, "error.html", map[string]string{
			"Error": "No search results found.",
		})
		return
	}

	// Render sources first.
	templates.ExecuteTemplate(w, "sources.html", map[string]any{
		"Results": results,
		"Query":   query,
	})
}

func handleStream(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "empty query", http.StatusBadRequest)
		return
	}

	cfg := getSettings()

	// Re-search to get context (or we could cache, but keeping it simple).
	results, err := searchSearxng(query, cfg)
	if err != nil {
		fmt.Fprintf(w, "data: Search error: %s\n\ndata: [DONE]\n\n", err)
		return
	}

	messages := buildPrompt(query, results)

	reqBody := ChatRequest{
		Model:    cfg.LLMModel,
		Messages: messages,
		Stream:   true,
	}
	bodyJSON, _ := json.Marshal(reqBody)

	base := strings.TrimRight(cfg.LLMURL, "/")
	// If the URL already ends with /v1, don't double it.
	var llmURL string
	if strings.HasSuffix(base, "/v1") {
		llmURL = base + "/chat/completions"
	} else {
		llmURL = base + "/v1/chat/completions"
	}
	req, err := http.NewRequestWithContext(r.Context(), "POST", llmURL, bytes.NewReader(bodyJSON))
	if err != nil {
		fmt.Fprintf(w, "data: Request error: %s\n\ndata: [DONE]\n\n", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.LLMAPIKey != "" {
		// Anthropic uses x-api-key; most others use Authorization: Bearer.
		if strings.Contains(cfg.LLMURL, "anthropic.com") {
			req.Header.Set("x-api-key", cfg.LLMAPIKey)
		} else {
			req.Header.Set("Authorization", "Bearer "+cfg.LLMAPIKey)
		}
	}

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(w, "data: LLM request failed: %s\n\ndata: [DONE]\n\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(w, "data: LLM error (%d): %s\n\ndata: [DONE]\n\n", resp.StatusCode, string(errBody[:min(len(errBody), 300)]))
		return
	}

	// Stream SSE.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
			return
		}

		var chunk ChatChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			content := chunk.Choices[0].Delta.Content
			escaped, _ := json.Marshal(content)
			fmt.Fprintf(w, "data: %s\n\n", escaped)
			flusher.Flush()
		}
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func handleHealthcheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprint(w, `{"status":"ok"}`)
}

func main() {
	dataDir := os.Getenv("OPENHOST_APP_DATA_DIR")
	if dataDir == "" {
		dataDir = "."
	}
	configPath = filepath.Join(dataDir, "settings.json")

	if err := loadSettings(); err != nil {
		log.Printf("warning: could not load settings: %v", err)
	}

	funcMap := template.FuncMap{
		"add": func(a, b int) int { return a + b },
	}
	var err error
	templates, err = template.New("").Funcs(funcMap).ParseFS(templateFiles, "templates/*.html")
	if err != nil {
		log.Fatalf("failed to parse templates: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleIndex)
	mux.HandleFunc("/settings", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			handleSettingsSave(w, r)
		} else {
			handleSettings(w, r)
		}
	})
	mux.HandleFunc("/search", handleSearch)
	mux.HandleFunc("/stream", handleStream)
	mux.HandleFunc("/healthcheck", handleHealthcheck)
	mux.Handle("/static/", http.FileServer(http.FS(staticFiles)))

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	log.Printf("listening on :%s", port)
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		log.Fatal(err)
	}
}

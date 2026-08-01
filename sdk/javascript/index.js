/**
 * CodeMind JavaScript SDK
 * Official JS SDK for the CodeMind AI Platform.
 */
class CodeMindClient {
  constructor({ apiKey = '', baseUrl = 'http://localhost:8000', timeout = 120000 } = {}) {
    this.apiKey = apiKey || process.env.CODEMIND_API_KEY || '';
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeout = timeout;
  }
  get _headers() {
    return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${this.apiKey}`, 'User-Agent': 'CodeMind-JS-SDK/2.0' };
  }
  async _post(endpoint, data) {
    const res = await fetch(`${this.baseUrl}${endpoint}`, { method: 'POST', headers: this._headers, body: JSON.stringify(data) });
    if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail || `HTTP ${res.status}`); }
    return res.json();
  }
  async _get(endpoint) { return (await fetch(`${this.baseUrl}${endpoint}`, { headers: this._headers })).json(); }
  async chat(message, { system, sessionId, maxTokens = 2048, temperature = 0.7, history = [], tools } = {}) {
    const messages = [];
    if (system) messages.push({ role: 'system', content: system });
    messages.push(...history, { role: 'user', content: message });
    const r = await this._post('/v1/chat/completions', { model: 'codemind-v2', messages, max_tokens: maxTokens, temperature, session_id: sessionId, tools });
    return r.choices[0].message.content;
  }
  async complete(prefix, { suffix = '', language = 'python', maxTokens = 256, temperature = 0.3 } = {}) { return (await this._post('/v1/ide/complete', { prefix, suffix, language, max_tokens: maxTokens, temperature })).completion; }
  async explain(code, language = 'python') { return (await this._post('/v1/ide/explain', { code, language })).explanation; }
  async fix(code, error, language = 'python') { return (await this._post('/v1/ide/fix', { code, error, language })).fixed_code; }
  async document(code, language = 'python') { return (await this._post('/v1/ide/document', { code, language })).documented_code; }
  async refactor(code, instruction, language = 'python') { return (await this._post('/v1/ide/refactor', { code, instruction, language })).refactored_code; }
  async test(code, { framework = 'jest', language = 'javascript' } = {}) { return (await this._post('/v1/ide/test', { code, framework, language })).tests; }
  async execute(code, language = 'python', stdin = null) { return this._post('/v1/execute', { code, language, stdin }); }
  async generateImage(prompt, { negativePrompt = '', size = '512x512', steps = 20, seed } = {}) { return (await this._post('/v1/images/generations', { prompt, negative_prompt: negativePrompt, size, steps, seed, n: 1 })).data[0]; }
  async transcribe(audioPath, language = null) { return (await this._post('/v1/audio/transcriptions', { audio_path: audioPath, language })).text; }
  async speak(text, outputPath = null) { return (await this._post('/v1/audio/speech', { input: text, output_path: outputPath })).audio_path; }
  async remember(content, sessionId = 'global') { return (await this._post('/v1/memory/store', { content, session_id: sessionId })).stored; }
  async recall(query, { sessionId, topK = 5 } = {}) { return (await this._post('/v1/memory/recall', { query, session_id: sessionId, top_k: topK })).memories; }
  async search(query, maxResults = 5) { return (await this._post('/v1/search', { query, max_results: maxResults })).results; }
  async health() { return this._get('/health'); }
  async models() { return (await this._get('/v1/models')).data; }
  stream(onToken, onDone) {
    const ws = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/v1/ws/stream`);
    ws.onmessage = e => { const d = JSON.parse(e.data); if (d.done) onDone?.(); else onToken?.(d.content); };
    return { send: (messages, opts = {}) => ws.send(JSON.stringify({ messages, ...opts })), close: () => ws.close() };
  }
}
if (typeof module !== 'undefined') module.exports = { CodeMindClient };

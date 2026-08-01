# CodeMind AI Platform — API Reference

## Authentication
Every request needs: `Authorization: Bearer YOUR_API_KEY`
Generate: `python codemind.py generate-key`

---

## OpenAI-Compatible Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/chat/completions | Chat with CodeMind |
| POST | /v1/completions | Text completion |
| POST | /v1/images/generations | Generate images |
| POST | /v1/audio/transcriptions | Speech to text |
| POST | /v1/audio/speech | Text to speech |
| GET  | /v1/models | List models |

## CodeMind Extensions

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/execute | Run code in sandbox |
| POST | /v1/search | DuckDuckGo search |
| POST | /v1/memory/store | Store in memory |
| POST | /v1/memory/recall | Recall from memory |
| GET  | /v1/memory/stats | Memory statistics |

## IDE API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /v1/ide/complete | Inline completion |
| POST | /v1/ide/explain | Explain code |
| POST | /v1/ide/fix | Fix errors |
| POST | /v1/ide/document | Generate docs |
| POST | /v1/ide/refactor | Refactor code |
| POST | /v1/ide/test | Generate tests |

## WebSocket
`WS /v1/ws/stream` — Real-time streaming responses

---

## Python SDK

```python
from sdk.python import CodeMindClient

client = CodeMindClient(
    api_key="cm-pro-yourkey",
    base_url="http://localhost:8000"
)

# Chat
response = client.chat("Build a FastAPI REST API")

# IDE features
completion = client.complete("def fibonacci(n):")
explanation = client.explain("lambda x: x**2")
fixed = client.fix("print(x", "NameError: x not defined")
docs = client.document("def add(a, b): return a + b")
tests = client.test("def multiply(a, b): return a * b")

# Code execution
result = client.execute("print('Hello from CodeMind!')")
print(result['stdout'])

# Image generation
img_path = client.generate_image("a cyberpunk city at night")

# Voice
client.speak("Hello, I am CodeMind AI")
text = client.transcribe("recording.wav")

# Memory
client.remember("User prefers Python and FastAPI")
memories = client.recall("what does the user prefer?")

# Web search
results = client.search("latest Python 3.13 features")

# Health check
print(client.health())
```

## JavaScript SDK

```javascript
const { CodeMindClient } = require('./sdk/javascript')

const client = new CodeMindClient({
    apiKey: 'cm-pro-yourkey',
    baseUrl: 'http://localhost:8000'
})

const response = await client.chat('Build a React component')
const completion = await client.complete('function fibonacci(n) {')
const result = await client.execute("console.log('hello')", 'javascript')
const img = await client.generateImage('sunset over mountains')
```

## OpenAI Drop-in Replacement

```python
import openai

# Point to your local CodeMind server
openai.api_key = "cm-pro-yourkey"
openai.base_url = "http://localhost:8000/v1"

# Now use exactly like OpenAI — no code changes needed
response = openai.chat.completions.create(
    model="codemind-v2",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## cURL Examples

```bash
# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"codemind-v2","messages":[{"role":"user","content":"Hello"}]}'

# Execute code
curl -X POST http://localhost:8000/v1/execute \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"code":"print(2+2)","language":"python"}'

# Generate image
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a futuristic city","size":"512x512"}'

# Search web
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"latest AI news","max_results":5}'
```

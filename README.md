# Planetary Comparative Analysis Agent

Delivers a cited, sectioned comparative analysis of Earth and Jupiter using authoritative knowledge base documents. Implements a retrieval-augmented generation (RAG) pipeline with Azure AI Search and Azure OpenAI, returning structured, source-cited responses via a FastAPI interface.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy the example environment file and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

**Direct execution:**
```
python code/agent.py
```

**As a FastAPI server:**
```
uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`

**General**
- `ENVIRONMENT`

**Azure Key Vault (optional for production)**
- `USE_KEY_VAULT`
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MODELS` (JSON list, optional)

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`

**Observability (Azure SQL)**
- `SERVICE_NAME`
- `SERVICE_VERSION`
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`
- `VALIDATION_CONFIG_PATH`

---

## API Endpoints

### **GET** `/health`
- **Description:** Health check endpoint.
- **Response:**
  ```
  {
    "status": "ok"
  }
  ```

### **POST** `/analyze`
- **Description:** Returns a cited, structured comparative analysis of Earth and Jupiter.
- **Request body:**
  ```
  {}
  ```
  *(No fields required; SYSTEM_PROMPT is used internally)*

- **Response:**
  ```
  {
    "success": true|false,
    "result": "string (optional, present if success)",
    "error": "string (optional, present if failure)",
    "tips": "string (optional, present if failure)"
  }
  ```

### **422 Error Handler**
- **Description:** Handles malformed JSON requests.
- **Response:**
  ```
  {
    "success": false,
    "error": "Malformed JSON request. Please check your input formatting (quotes, commas, brackets).",
    "tips": "Ensure your request body is valid JSON and does not exceed 50,000 characters."
  }
  ```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t planetary-comparative-analysis-agent -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name planetary-comparative-analysis-agent planetary-comparative-analysis-agent
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs planetary-comparative-analysis-agent
```

### 7. Stop the container:
```
docker stop planetary-comparative-analysis-agent
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Planetary Comparative Analysis Agent** — Reliable, cited planetary science comparisons with Azure-powered RAG.
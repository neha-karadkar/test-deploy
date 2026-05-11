import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, model_validator

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

from config import Config

# Constants for agent operation
SYSTEM_PROMPT = (
    "You are a planetary science expert tasked with delivering a comprehensive, professional comparative analysis of Earth and Jupiter. Your response must:\n"
    "- Clearly state the equatorial diameter of each planet in both miles and kilometers, citing the source document for each value. - Quantitatively describe the size difference, including how many Earths could fit inside Jupiter, with explicit citation. - Compare the average distances of Earth and Jupiter from the Sun (in miles and kilometers), citing sources. - Explain the significance of their orbital positions and the scale of the difference. - Structure your answer in clear, logical sections (e.g., Physical Dimensions, Size Comparison, Orbital Distances, Spatial Relationship). - Only use information from the provided knowledge base context (Earth.pdf, Jupiter.pdf). Do not speculate or use external sources. - If relevant information is not found, state this clearly and suggest the user rephrase or provide more details. - Format all numerical values for clarity and cite the source after each fact."
)
OUTPUT_FORMAT = (
    "- Sectioned, structured text with headings for each comparison aspect. - All facts and figures must be followed by (Source: DocumentName.pdf). - Use bullet points or tables where appropriate for clarity."
)
FALLBACK_RESPONSE = (
    "The requested comparative information could not be found in the available knowledge base documents (Earth.pdf, Jupiter.pdf). Please rephrase your query or specify additional details for further assistance."
)
SELECTED_DOCUMENT_TITLES = ["Earth.pdf", "Jupiter.pdf"]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# Sanitizer utility for LLM output
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")


def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()


@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# Pydantic request/response models
class AnalyzeRequest(BaseModel):
    # No dynamic parameters needed; SYSTEM_PROMPT is used internally
    pass

class AnalyzeResponse(BaseModel):
    success: bool = Field(..., description="Whether the operation was successful")
    result: Optional[str] = Field(None, description="Agent's comparative analysis response")
    error: Optional[str] = Field(None, description="Error message if operation failed")
    tips: Optional[str] = Field(None, description="Helpful tips for fixing input or retrying")

# Error handler utility
class ErrorHandler:
    """Centralized error handling for agent errors."""

    def handle_error(self, error_code: str, context: dict = None) -> str:
        """Map error codes to user-friendly responses."""
        context = context or {}
        if error_code == "NO_CONTEXT_FOUND":
            return FALLBACK_RESPONSE
        elif error_code == "INVALID_DOCUMENT_FILTER":
            return "Document filtering failed. Please ensure only Earth.pdf and Jupiter.pdf are selected."
        else:
            return "An unexpected error occurred. Please try again or contact support."

# Chunk retrieval service
class ChunkRetriever:
    """Retrieves relevant chunks from Azure AI Search."""

    def __init__(self):
        self.search_client = None

    def _get_search_client(self):
        if self.search_client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            api_key = Config.AZURE_SEARCH_API_KEY
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            if not endpoint or not api_key or not index_name:
                raise ValueError("Azure Search configuration missing.")
            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self.search_client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, filter_titles: List[str], top_k: int = 5) -> List[str]:
        """Retrieve top-K relevant chunks from Azure AI Search, filtered by document titles."""
        search_client = self._get_search_client()
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        # Embed the query
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_tool_call(
                tool_name="openai_client.embeddings.create",
                latency_ms=int((_time.time() - _t0) * 1000),
                output=str(embedding_resp)[:200] if embedding_resp else None,
                status="success",
            )
        except Exception:
            pass
        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        # Build OData filter for selected document titles
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top_k,
            "select": ["chunk", "title"],
        }
        if filter_titles:
            odata_parts = [f"title eq '{t}'" for t in filter_titles]
            search_kwargs["filter"] = " or ".join(odata_parts)
        _t1 = _time.time()
        results = search_client.search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass
        context_chunks = [r["chunk"] for r in results if r.get("chunk")]
        return context_chunks

# LLM service
class LLMService:
    """Handles LLM calls to Azure OpenAI."""

    def __init__(self):
        self.client = None

    def _get_llm_client(self):
        if self.client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_response(self, system_prompt: str, user_prompt: str, context_chunks: List[str]) -> str:
        """Call Azure OpenAI GPT-4.1 with system prompt, user prompt, and context chunks."""
        client = self._get_llm_client()
        _llm_kwargs = Config.get_llm_kwargs()
        # Build system message with output format appended
        system_message = system_prompt + "\n\nOutput Format: " + OUTPUT_FORMAT
        # Compose context as a single string
        context_str = "\n\n".join(context_chunks) if context_chunks else ""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": context_str}
        ]
        _t0 = _time.time()
        response = await client.chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4.1",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4.1",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

# Response formatting utility
class ResponseFormatter:
    """Ensures sectioned, cited, structured output."""

    def format_response(self, raw_response: str) -> str:
        """Format LLM output for clarity and citation."""
        # Sanitize LLM output
        text = sanitize_llm_output(raw_response, content_type="text")
        # Additional formatting can be added here if needed
        return text

# Main agent orchestrator
class AgentOrchestrator:
    """Coordinates retrieval, LLM, formatting, and error handling."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()
        self.response_formatter = ResponseFormatter()
        self.error_handler = ErrorHandler()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self) -> Dict[str, Any]:
        """Main orchestration method for planetary comparative analysis."""
        async with trace_step(
            "retrieve_chunks",
            step_type="tool_call",
            decision_summary="Retrieve relevant chunks from Azure AI Search",
            output_fn=lambda r: f"chunks={len(r)}",
        ) as step:
            try:
                chunks = await self.chunk_retriever.retrieve_chunks(
                    query=SYSTEM_PROMPT,
                    filter_titles=SELECTED_DOCUMENT_TITLES,
                    top_k=5
                )
                step.capture({"chunks": len(chunks)})
            except Exception as e:
                logging.error(f"Chunk retrieval failed: {e}")
                return {
                    "success": False,
                    "result": None,
                    "error": self.error_handler.handle_error("NO_CONTEXT_FOUND"),
                    "tips": "Ensure knowledge base documents are available and try again."
                }
        if not chunks:
            return {
                "success": False,
                "result": None,
                "error": self.error_handler.handle_error("NO_CONTEXT_FOUND"),
                "tips": "No relevant information found. Please rephrase your query."
            }
        async with trace_step(
            "llm_generate_response",
            step_type="llm_call",
            decision_summary="Generate cited, structured comparative analysis",
            output_fn=lambda r: f"response_len={len(r)}",
        ) as step:
            try:
                raw_response = await self.llm_service.generate_response(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=SYSTEM_PROMPT,
                    context_chunks=chunks
                )
                step.capture({"response_len": len(raw_response)})
            except Exception as e:
                logging.error(f"LLM call failed: {e}")
                return {
                    "success": False,
                    "result": None,
                    "error": self.error_handler.handle_error("NO_CONTEXT_FOUND"),
                    "tips": "LLM service unavailable. Please try again later."
                }
        async with trace_step(
            "format_response",
            step_type="format",
            decision_summary="Format LLM output for clarity and citation",
            output_fn=lambda r: f"formatted_len={len(r)}",
        ) as step:
            try:
                formatted = self.response_formatter.format_response(raw_response)
                step.capture({"formatted_len": len(formatted)})
            except Exception as e:
                logging.error(f"Formatting failed: {e}")
                return {
                    "success": False,
                    "result": None,
                    "error": self.error_handler.handle_error("NO_CONTEXT_FOUND"),
                    "tips": "Formatting error. Please try again."
                }
        return {
            "success": True,
            "result": formatted,
            "error": None,
            "tips": None
        }

# FastAPI observability lifespan function
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)
    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# FastAPI app
app = FastAPI(
    title="Planetary Comparative Analysis Agent",
    description="Delivers cited, sectioned comparative analysis of Earth and Jupiter using authoritative knowledge base documents.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# JSON error handler for malformed requests
@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON request. Please check your input formatting (quotes, commas, brackets).",
            "tips": "Ensure your request body is valid JSON and does not exceed 50,000 characters.",
        },
    )

# Main business endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def analyze_endpoint():
    """Planetary comparative analysis endpoint."""
    agent = AgentOrchestrator()
    try:
        result = await agent.process_query()
        # Sanitize output
        if result.get("result"):
            result["result"] = sanitize_llm_output(result["result"], content_type="text")
        return AnalyzeResponse(**result)
    except Exception as e:
        logging.error(f"Agent execution failed: {e}")
        return AnalyzeResponse(
            success=False,
            result=None,
            error="Agent execution failed. Please try again or contact support.",
            tips="If the error persists, please provide more details or rephrase your query."
        )

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())
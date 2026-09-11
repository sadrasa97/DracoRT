"""
OpenAI-Compatible API Server

Provides REST API endpoints compatible with OpenAI's API format:
- POST /v1/completions       — Text completions
- POST /v1/chat/completions  — Chat completions (message format)
- GET  /v1/models            — List available models
- GET  /health               — Health check
- GET  /metrics              — Prometheus metrics

Supports both streaming (SSE) and non-streaming responses.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("draco.server")


# ======================================================================
# Request/Response types (OpenAI-compatible)
# ======================================================================

@dataclass
class CompletionRequest:
    """OpenAI-compatible completion request."""
    model: str
    prompt: str = ""
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stop: Optional[List[str]] = None
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage] = field(default_factory=list)
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    stop: Optional[List[str]] = None
    stream: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None


@dataclass
class CompletionChoice:
    """A single completion choice."""
    text: str
    index: int = 0
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


@dataclass
class ChatCompletionChoice:
    """A single chat completion choice."""
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class UsageInfo:
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResponse:
    """OpenAI-compatible completion response."""
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "text": c.text,
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: List[ChatCompletionChoice] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


@dataclass
class StreamDelta:
    """A delta in a streaming response."""
    role: Optional[str] = None
    content: Optional[str] = None


@dataclass
class StreamChoice:
    """A choice in a streaming response."""
    index: int = 0
    delta: StreamDelta = field(default_factory=StreamDelta)
    finish_reason: Optional[str] = None


@dataclass
class StreamChunk:
    """A single streaming chunk (SSE)."""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[StreamChoice] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [],
        }
        for c in self.choices:
            cd: Dict[str, Any] = {
                "index": c.index,
                "delta": {},
                "finish_reason": c.finish_reason,
            }
            if c.delta.role:
                cd["delta"]["role"] = c.delta.role
            if c.delta.content is not None:
                cd["delta"]["content"] = c.delta.content
            d["choices"].append(cd)
        return d

    def to_sse(self) -> str:
        return f"data: {json.dumps(self.to_dict())}\n\n"


# ======================================================================
# Request parsing helpers
# ======================================================================

def parse_completion_request(data: Dict[str, Any]) -> CompletionRequest:
    """Parse a JSON dict into a CompletionRequest."""
    return CompletionRequest(
        model=data.get("model", ""),
        prompt=data.get("prompt", ""),
        max_tokens=data.get("max_tokens", 16),
        temperature=data.get("temperature", 1.0),
        top_p=data.get("top_p", 1.0),
        top_k=data.get("top_k", -1),
        n=data.get("n", 1),
        stop=data.get("stop"),
        stream=data.get("stream", False),
        presence_penalty=data.get("presence_penalty", 0.0),
        frequency_penalty=data.get("frequency_penalty", 0.0),
        seed=data.get("seed"),
        user=data.get("user"),
    )


def parse_chat_completion_request(data: Dict[str, Any]) -> ChatCompletionRequest:
    """Parse a JSON dict into a ChatCompletionRequest."""
    messages = []
    for m in data.get("messages", []):
        messages.append(ChatMessage(role=m.get("role", "user"), content=m.get("content", "")))
    return ChatCompletionRequest(
        model=data.get("model", ""),
        messages=messages,
        max_tokens=data.get("max_tokens", 16),
        temperature=data.get("temperature", 1.0),
        top_p=data.get("top_p", 1.0),
        top_k=data.get("top_k", -1),
        n=data.get("n", 1),
        stop=data.get("stop"),
        stream=data.get("stream", False),
        presence_penalty=data.get("presence_penalty", 0.0),
        frequency_penalty=data.get("frequency_penalty", 0.0),
        seed=data.get("seed"),
        user=data.get("user"),
    )


def messages_to_prompt(messages: List[ChatMessage], tokenizer: Any = None) -> str:
    """Convert chat messages to a single prompt string.

    Uses the model tokenizer's chat template when available (correct for
    instruction-tuned models like Qwen2.5 / Llama-3, whose templates carry
    their own special tokens and stop behavior); falls back to a generic
    "User:/Assistant:" format otherwise.
    """
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}\n")
    parts.append("Assistant:")
    return "\n".join(parts)


# ======================================================================
# DracoServer
# ======================================================================

class DracoServer:
    """
    OpenAI-compatible inference server.

    Wraps a Draco LLM instance and exposes OpenAI-compatible REST endpoints.
    Can be used with any WSGI/ASGI server, or with the built-in HTTP server.

    Usage:
        llm = LLM(model="meta-llama/Llama-3-8B")
        server = DracoServer(llm=llm, host="0.0.0.0", port=8000)
        server.start()  # blocking
    """

    def __init__(
        self,
        llm: Any = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_name: Optional[str] = None,
    ):
        self.llm = llm
        self.host = host
        self.port = port
        self.model_name = model_name or "draco-model"
        self._start_time = time.time()
        self._request_count = 0

    def handle_completion(self, request: CompletionRequest) -> CompletionResponse:
        """Handle a /v1/completions request."""
        from draco.engine.sampling import SamplingParams

        self._request_count += 1

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
            n=request.n,
        )

        if self.llm is not None:
            outputs = self.llm.generate([request.prompt], params)
            if outputs:
                out = outputs[0]
                text = out.outputs[0].text if out.outputs else ""
                token_ids = out.outputs[0].token_ids if out.outputs else []
                usage = UsageInfo(
                    prompt_tokens=len(out.prompt_token_ids) if out.prompt_token_ids else 0,
                    completion_tokens=len(token_ids),
                    total_tokens=(len(out.prompt_token_ids) if out.prompt_token_ids else 0) + len(token_ids),
                )
            else:
                text = ""
                usage = UsageInfo()
        else:
            text = f"[Draco server echo] {request.prompt[:100]}"
            usage = UsageInfo(prompt_tokens=len(request.prompt.split()), completion_tokens=0)

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=self.model_name,
            choices=[CompletionChoice(text=text, index=0, finish_reason="stop")],
            usage=usage,
        )

    def handle_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Handle a /v1/chat/completions request."""
        from draco.engine.sampling import SamplingParams

        self._request_count += 1

        # Convert messages to prompt (chat template when available)
        tokenizer = getattr(self.llm, "_tokenizer", None) if self.llm is not None else None
        prompt = messages_to_prompt(request.messages, tokenizer)

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
            n=request.n,
        )

        if self.llm is not None:
            outputs = self.llm.generate([prompt], params)
            if outputs:
                out = outputs[0]
                text = out.outputs[0].text if out.outputs else ""
                token_ids = out.outputs[0].token_ids if out.outputs else []
                usage = UsageInfo(
                    prompt_tokens=len(out.prompt_token_ids) if out.prompt_token_ids else 0,
                    completion_tokens=len(token_ids),
                    total_tokens=(len(out.prompt_token_ids) if out.prompt_token_ids else 0) + len(token_ids),
                )
            else:
                text = ""
                usage = UsageInfo()
        else:
            text = f"[Draco server echo] I received your message."
            usage = UsageInfo(prompt_tokens=10, completion_tokens=8)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=text),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )

    def handle_chat_completion_stream(self, request: ChatCompletionRequest):
        """Handle a streaming /v1/chat/completions request. Yields StreamChunk objects."""
        from draco.engine.sampling import SamplingParams

        self._request_count += 1
        tokenizer = getattr(self.llm, "_tokenizer", None) if self.llm is not None else None
        prompt = messages_to_prompt(request.messages, tokenizer)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            max_tokens=request.max_tokens,
            stop=request.stop,
            seed=request.seed,
        )

        # First chunk: role
        yield StreamChunk(
            id=chunk_id,
            created=created,
            model=self.model_name,
            choices=[StreamChoice(
                index=0,
                delta=StreamDelta(role="assistant"),
                finish_reason=None,
            )],
        )

        if self.llm is not None:
            # Use streaming generator
            from draco.engine.stream import StreamingGenerator
            gen = StreamingGenerator(self.llm)
            for stream_output in gen.stream(
                prompt, max_tokens=request.max_tokens,
                temperature=request.temperature, top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else -1,
                stop_token_ids=None,
            ):
                if stream_output.deltas:
                    last_delta = stream_output.deltas[-1]
                    yield StreamChunk(
                        id=chunk_id,
                        created=created,
                        model=self.model_name,
                        choices=[StreamChoice(
                            index=0,
                            delta=StreamDelta(content=last_delta.text),
                            finish_reason="stop" if stream_output.finished else None,
                        )],
                    )
        else:
            # Echo mode: send a simple response
            echo_text = "Hello! I'm the Draco server."
            for ch in echo_text:
                yield StreamChunk(
                    id=chunk_id,
                    created=created,
                    model=self.model_name,
                    choices=[StreamChoice(
                        index=0,
                        delta=StreamDelta(content=ch),
                        finish_reason=None,
                    )],
                )

        # Final chunk
        yield StreamChunk(
            id=chunk_id,
            created=created,
            model=self.model_name,
            choices=[StreamChoice(
                index=0,
                delta=StreamDelta(),
                finish_reason="stop",
            )],
        )

    def handle_models(self) -> Dict[str, Any]:
        """Handle GET /v1/models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model_name,
                    "object": "model",
                    "created": int(self._start_time),
                    "owned_by": "draco",
                }
            ],
        }

    def handle_health(self) -> Dict[str, Any]:
        """Handle GET /health."""
        return {
            "status": "ok",
            "model": self.model_name,
            "uptime_seconds": time.time() - self._start_time,
            "request_count": self._request_count,
        }

    def handle_metrics(self) -> str:
        """Handle GET /metrics — Prometheus-compatible metrics."""
        if self.llm is not None and hasattr(self.llm, "_metrics"):
            from draco.metrics import MetricsCollector
            collector = self.llm._metrics or MetricsCollector()
            return collector.to_prometheus()
        return "# No metrics available\n"

    def __repr__(self) -> str:
        return (
            f"DracoServer("
            f"model={self.model_name!r}, "
            f"host={self.host!r}, "
            f"port={self.port})"
        )


# ======================================================================
# Simple HTTP handler (stdlib)
# ======================================================================

def create_app(llm: Any = None, model_name: Optional[str] = None) -> DracoServer:
    """Create a DracoServer instance."""
    return DracoServer(llm=llm, model_name=model_name)


class DracoHTTPHandler:
    """
    Simple HTTP request handler using Python's http.server.
    Routes requests to DracoServer methods.

    Usage:
        from http.server import HTTPServer
        from draco.server.app import DracoHTTPHandler

        server = DracoServer(llm=llm)
        http_server = HTTPServer(("0.0.0.0", 8000), DracoHTTPHandler)
        http_server.draco_server = server
        http_server.serve_forever()
    """

    draco_server: Optional[DracoServer] = None

    def _parse_body(self) -> Dict[str, Any]:
        """Read and parse JSON body from the request."""
        import json as json_mod
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            return json_mod.loads(body.decode("utf-8"))
        return {}

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send a JSON response."""
        import json as json_mod
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json_mod.dumps(data).encode("utf-8"))

    def _send_sse(self, chunks: Any) -> None:
        """Send a Server-Sent Events response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        for chunk in chunks:
            self.wfile.write(chunk.to_sse().encode("utf-8"))
            self.wfile.flush()

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        server = self.draco_server
        if server is None:
            self._send_json({"error": "Server not initialized"}, 500)
            return

        if self.path == "/health" or self.path == "/v1/health":
            self._send_json(server.handle_health())
        elif self.path == "/v1/models":
            self._send_json(server.handle_models())
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(server.handle_metrics().encode("utf-8"))
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests."""
        server = self.draco_server
        if server is None:
            self._send_json({"error": "Server not initialized"}, 500)
            return

        data = self._parse_body()

        if self.path == "/v1/completions":
            req = parse_completion_request(data)
            if req.stream:
                # Streaming not supported for /v1/completions yet
                resp = server.handle_completion(req)
                self._send_json(resp.to_dict())
            else:
                resp = server.handle_completion(req)
                self._send_json(resp.to_dict())

        elif self.path == "/v1/chat/completions":
            req = parse_chat_completion_request(data)
            if req.stream:
                chunks = server.handle_chat_completion_stream(req)
                self._send_sse(chunks)
            else:
                resp = server.handle_chat_completion(req)
                self._send_json(resp.to_dict())

        else:
            self._send_json({"error": "Not found"}, 404)

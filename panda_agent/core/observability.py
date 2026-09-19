"""Core Observability & Metrics Module for Panda Agent.

Provides integration for Prometheus metrics and OpenTelemetry tracing.
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# --- PROMETHEUS METRICS ---

# Latency tracking
LLM_LATENCY = Histogram(
    'panda_llm_latency_seconds', 
    'Time spent waiting for LLM response', 
    ['model_name']
)
TOOL_LATENCY = Histogram(
    'panda_tool_latency_seconds', 
    'Time spent executing tools', 
    ['tool_name']
)
RAG_LATENCY = Histogram(
    'panda_rag_latency_seconds', 
    'Time spent in RAG retrieval'
)

# Token usage tracking
TOKEN_USAGE = Counter(
    'panda_tokens_total', 
    'Total tokens consumed/generated', 
    ['type']
) # type: prompt, completion

# Tool error tracking
TOOL_ERRORS = Counter(
    'panda_tool_errors_total', 
    'Total tool execution failures', 
    ['tool_name', 'error_type']
)

# Active Sessions
ACTIVE_SESSIONS = Gauge(
    'panda_active_sessions', 
    'Number of active conversation sessions'
)

# Get application tracer
tracer = trace.get_tracer("panda_agent_tracer")

def setup_telemetry(app):
    """Initialize OpenTelemetry tracer and instrument the FastAPI app."""
    # Configure Tracer Provider
    provider = TracerProvider()
    
    # Configure OTLP Exporter (Sending to localhost:4317 - Jaeger/Tempo)
    try:
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
    except Exception as e:
        print(f"Failed to configure OTLP Exporter: {e}")
        
    # Auto-instrument FastAPI routes
    FastAPIInstrumentor.instrument_app(app)
    
    return tracer

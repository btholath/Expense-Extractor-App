# observability.py — metrics, health, logs, optional OTEL tracing (idempotent)

import os
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# ---------- Config ----------
APP_NAME = os.getenv("APP_NAME", "receipt-tracker")
ENV = os.getenv("ENV", "dev")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9400"))   # /metrics
HEALTH_PORT  = int(os.getenv("HEALTH_PORT", "9401"))    # /healthz
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "false").lower() in ("1","true","yes")
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# ---------- Logging (JSON) ----------
logging.basicConfig(level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger().bind(app=APP_NAME, env=ENV)

# ---------- Prometheus ----------
RECEIPT_IMAGES_TOTAL = Counter(
    "receipt_images_total", "Uploaded receipt images", ["status"]
)
MODEL_CALLS_TOTAL = Counter(
    "gemini_calls_total", "Gemini generate_content calls", ["outcome","model"]
)
MODEL_LATENCY = Histogram(
    "gemini_call_seconds", "Latency of Gemini calls (s)", buckets=(0.1,0.2,0.5,1,2,5,10)
)
TOTAL_COMPUTED = Gauge(
    "receipt_total_last_computed", "Last computed final total (USD)"
)

# Idempotency flags to survive Streamlit reruns
_METRICS_STARTED = False
_HEALTH_STARTED  = False
_OTEL_STARTED    = False

def start_prometheus():
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    try:
        start_http_server(METRICS_PORT)  # non-blocking
        log.info("metrics_started", port=METRICS_PORT)
        _METRICS_STARTED = True
    except OSError as e:
        log.warning("metrics_port_in_use_skip_start", port=METRICS_PORT, error=str(e))
        _METRICS_STARTED = True  # prevent retry loops

# ---------- Health / readiness ----------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200); self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()

def start_health():
    global _HEALTH_STARTED
    if _HEALTH_STARTED:
        return

    def run():
        try:
            srv = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
            log.info("health_started", port=HEALTH_PORT)
            srv.serve_forever()
        except OSError as e:
            log.warning("health_port_in_use_skip_start", port=HEALTH_PORT, error=str(e))

    threading.Thread(target=run, daemon=True).start()
    _HEALTH_STARTED = True

# ---------- Tracing (optional, idempotent) ----------
def init_tracing():
    global _OTEL_STARTED
    if _OTEL_STARTED or not OTEL_ENABLED:
        if not OTEL_ENABLED:
            log.info("otel_disabled")
        return trace.get_tracer(APP_NAME) if _OTEL_STARTED else None

    # Try to set a provider; if one exists, reuse it and attach an exporter if possible
    resource = Resource.create({"service.name": APP_NAME, "deployment.environment": ENV})
    try:
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
    except Exception as e:
        log.warning("otel_provider_exists", error=str(e))
        provider = trace.get_tracer_provider()

    # Attach exporter/processor if possible
    try:
        exporter = OTLPSpanExporter(endpoint=f"{OTLP_ENDPOINT}/v1/traces")
        processor = BatchSpanProcessor(exporter)
        if hasattr(provider, "add_span_processor"):
            provider.add_span_processor(processor)
    except Exception as e:
        # Don’t crash if endpoint is missing; spans will just be dropped
        log.warning("otel_exporter_attach_failed", error=str(e), endpoint=OTLP_ENDPOINT)

    # Instrument requests only once
    try:
        RequestsInstrumentor().instrument()
    except Exception as e:
        log.warning("requests_already_instrumented", error=str(e))

    _OTEL_STARTED = True
    log.info("otel_enabled", endpoint=OTLP_ENDPOINT)
    return trace.get_tracer(APP_NAME)

# ---------- Entry ----------
def boot():
    start_prometheus()
    start_health()
    tracer = init_tracing()
    return log, tracer, {
        "counters": {"images": RECEIPT_IMAGES_TOTAL, "calls": MODEL_CALLS_TOTAL},
        "hists": {"latency": MODEL_LATENCY},
        "gauges": {"total": TOTAL_COMPUTED},
    }

# 📊 Observability & Dashboards Guide

Panda Agent comes with built-in **Prometheus** metrics collection and a pre-configured **Grafana** dashboard.

---

## 🚀 How to Run the Observability Stack

### Option 1: Running with Docker Compose (Recommended)

1. **Start all services (Panda API, Prometheus, and Grafana):**
   ```bash
   docker-compose up -d
   ```

2. **Access Grafana Dashboard:**
   - **URL:** [http://localhost:3000](http://localhost:3000)
   - **Username:** `admin`
   - **Password:** `admin` (or as configured in `docker-compose.yml`)

3. **View the Pre-configured Dashboard:**
   - Go to **Dashboards** $\rightarrow$ **Panda Agent** folder $\rightarrow$ **🐼 Panda Agent Overview**.
   - You will see live stats for:
     - **Active Sessions** (`panda_active_sessions`)
     - **Total Tokens Consumed** (`panda_tokens_total`)
     - **Tool Execution Errors** (`panda_tool_errors_total`)
     - **LLM Latency (p95 & Avg)** (`panda_llm_latency_seconds`)
     - **Tool & RAG Execution Latency** (`panda_tool_latency_seconds`, `panda_rag_latency_seconds`)

---

### Option 2: Accessing Local Prometheus Directly

If you prefer to inspect raw Prometheus metrics:
- **Prometheus UI:** [http://localhost:9090](http://localhost:9090)
- **API Metrics Endpoint:** [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

## 🔒 Securing `/metrics` with API Key

If `REQUIRE_API_KEY=true` is enabled in your `.env`:
1. Open `prometheus/prometheus.yml`.
2. Add your bearer token or header under `job_name: 'panda-agent'`:
   ```yaml
   scrape_configs:
     - job_name: 'panda-agent'
       metrics_path: '/metrics'
       bearer_token: 'your_api_key_here'
       static_configs:
         - targets: ['panda-api:8000']
   ```
3. Restart Prometheus:
   ```bash
   docker-compose restart prometheus
   ```

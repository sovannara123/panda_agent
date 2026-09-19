# 🔒 Security Hardening Plan

**Plan File**: `plans/06_security_hardening.md`  
**Status**: Completed ✅  
**Auditor**: Penetration Tester (`agency-penetration-tester`)

---

## 🎯 Objective
Address critical and high-severity security vulnerabilities identified during the penetration testing audit of the Panda Agent REST API and LLM tool execution pipeline.

---

## 📋 Hardening Tasks Breakdown

### Phase 1: API Authentication & Session Security
- [x] **1.1 API Key / Bearer Token Authentication**
  - Add API key authentication dependency in `panda_agent/api/app.py`.
  - Require `X-API-Key` or `Authorization: Bearer <token>` for `/chat`, `/chat/stream`, `/upload`, and `/documents`.
  - Add `REQUIRE_API_KEY` and `API_KEY` settings to `panda_agent/core/config.py`.
- [x] **1.2 Session ID Validation & Ownership**
  - Sanitize `session_id` parameters with `sanitize_session_id()` to prevent injection attacks and session spoofing.

### Phase 2: File Upload & Input Hardening
- [x] **2.1 File Size & Content-Length Enforcement**
  - Enforce a 10MB limit on `POST /upload` during streaming.
  - Return `413 Payload Too Large` for oversized uploads.
- [x] **2.2 Filename Sanitization & Path Traversal Guard**
  - Use `pathlib.Path(file.filename).name` to sanitize incoming filenames and strip relative path markers (`../`).
- [x] **2.3 Page / Chunk Bounds Validation**
  - Restrict maximum page count (max 50 pages) per PDF during extraction.

### Phase 3: Indirect Prompt Injection Defenses (RAG)
- [x] **3.1 RAG Context Fencing**
  - Wrap retrieved vector store chunks in strict XML tags (`<retrieved_context>...</retrieved_context>`) inside `search_knowledge_base()`.
- [x] **3.2 System Prompt Guardrails**
  - Instruct the model in `SYSTEM_PROMPT` to treat contents inside `<retrieved_context>` strictly as passive reference data and ignore any embedded command directives.

### Phase 4: Rate Limiting & Denial of Wallet (DoW) Protection
- [x] **4.1 API Rate Limiting**
  - Integrate sliding window IP rate limiter dependency (`check_rate_limit`) in FastAPI.
  - Set default rate limit (20 requests per minute per IP) in `Config`.

### Phase 5: Error Sanitization & Information Leak Prevention
- [x] **5.1 Exception Handling Cleanup**
  - Replace raw `detail=f"Failed to process PDF: {str(error)}"` outputs with sanitized error messages.
  - Log full exception tracebacks internally via `log_event()` while returning generic status messages to clients.

---

## 🧪 Retest & Verification Plan

1. **Authentication Retest**:
   - `curl -X POST http://localhost:8000/chat` without headers $\rightarrow$ verify `401 Unauthorized`.
   - `curl -H "X-API-Key: valid_key" -X POST http://localhost:8000/chat` $\rightarrow$ verify `200 OK`.
2. **File Size Retest**:
   - Send oversized dummy file (>10MB) $\rightarrow$ verify `413 Payload Too Large`.
3. **Prompt Injection Retest**:
   - Upload PDF with prompt injection test strings $\rightarrow$ verify LLM does not execute injected commands.
4. **Rate Limit Retest**:
   - Send 25 rapid requests $\rightarrow$ verify 429 Too Many Requests response.

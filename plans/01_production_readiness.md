# Plan: Production Readiness

This document tracks the tasks required to get the Panda Agent ready for a production environment.

## Goal
Secure the API, ensure data persistence for the vector database, and set up proper deployment pipelines.

## Tasks

- [ ] **Task 1: API Security (CORS & Secrets)**
  - Update `panda_agent/api/app.py` to restrict `allow_origins` to specific frontend domains instead of `["*"]`.
  - Ensure API keys are loaded securely and not committed to source control.

- [ ] **Task 2: Database Persistence**
  - Update `docker-compose.yml` or deployment scripts to mount a persistent volume to `/app/chroma_db`.
  - Ensure ChromaDB does not lose data on container restart.

- [ ] **Task 3: Production Server Configuration**
  - Modify `Dockerfile` or `docker-compose.yml` to run the application using `gunicorn` with `uvicorn` workers instead of raw `uvicorn`.

- [ ] **Task 4: CI/CD Pipeline**
  - Create a `.github/workflows/ci.yml` file.
  - Configure automated testing (`pytest`) on Pull Requests.
  - Configure Docker image build and push on merge to `main`.

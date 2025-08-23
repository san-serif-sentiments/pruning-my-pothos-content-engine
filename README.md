# Agentic Content Engine (M1-friendly)

Opinionated, small-model-first content engine that:
- Indexes your existing content (Chroma + bge-small),
- Generates grounded drafts (Ollama Llama-3.1-8B or Phi-3 Mini),
- Converts Markdown to HTML,
- Publishes **as draft** to WordPress via REST.

## Guardrails
1) **Duplicate protection**: cosine similarity ≥ 0.92 vs. recent posts → skip publish.
2) **Allowlist references**: only domains in `sources.allow` are permitted in “References”.
3) **Social snippets**: if `social: true`, writes `artifacts/<slug>/social.md` for LinkedIn/X.

## Quickstart
1. Install Docker Desktop (Apple Silicon).
2. Copy `.env.example` → `.env` and set WordPress creds (or leave unset to skip publishing).
3. Start services:
```
make up
make pull
```
4. Add Markdown to `content/` (for RAG), then:
```
make index
```
5. Generate a draft from the example brief:
```
make run brief=briefs/example.yaml
```
6. If WP creds are set and guardrails pass, a **draft** post is created. Otherwise, artifacts are saved to `artifacts/<slug>/`.

## Models
- Default GEN_MODEL: `llama3.1:8b-instruct` (Q4 via Ollama). For speed: `phi3:mini`.
- Embeddings: `BAAI/bge-small-en-v1.5` (fast, solid quality on M1).

## Next steps
- Add LangGraph once stable.
- Add metrics (PostHog) later if needed.

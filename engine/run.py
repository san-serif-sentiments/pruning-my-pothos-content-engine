import os
import sys
import re
import json
import yaml
import requests
import datetime
import frontmatter
import numpy as np
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer
from engine.rag.search import retrieve
from engine.tools.html import md_to_clean_html

# Env
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.1:8b-instruct")
EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")

# History store for duplicate checks
HIST_DIR = "artifacts/_history"
HIST_PATH = os.path.join(HIST_DIR, "embeddings.jsonl")

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def ollama_complete(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["response"]

def load_prompt(system_path, user_path, **vars):
    with open(system_path, "r", encoding="utf-8") as f:
        system = f.read()
    with open(user_path, "r", encoding="utf-8") as f:
        user = f.read().format(**vars)
    return f"{system}\n\n{user}"

def extract_references(md: str):
    """
    Very simple extraction: collect all http(s) URLs found anywhere,
    then focus policy on the References section if present.
    """
    urls = re.findall(r"https?://[^\s\)\]]+", md)
    # If there's a References section, prefer URLs appearing after it
    refs_idx = md.lower().find("\n## references")
    if refs_idx != -1:
        refs_md = md[refs_idx:]
        urls = re.findall(r"https?://[^\s\)\]]+", refs_md)
    return urls

def domain(host: str):
    host = host.lower()
    return host[4:] if host.startswith("www.") else host

def parse_domains(urls):
    ds = []
    for u in urls:
        try:
            d = domain(urlparse(u).netloc)
            if d: ds.append(d)
        except Exception:
            continue
    return list(dict.fromkeys(ds))

def cosine(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def load_recent_history(n=50):
    if not os.path.exists(HIST_PATH):
        return []
    items = []
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items[-n:]

def append_history(record):
    ensure_dir(HIST_DIR)
    with open(HIST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def first_sentences(text: str, max_chars: int = 400):
    # naive first 2-3 sentences truncation
    parts = re.split(r"(?<=[\.!?])\s+", text.strip())
    out = ""
    for p in parts:
        if not p: continue
        if len(out) + len(p) + 1 > max_chars: break
        out = (out + " " + p).strip()
        if out.count(".") >= 2: break
    return out if out else text[:max_chars]

def generate_social(brief, draft_md, out_path):
    title = brief.get("title", "")
    tags = brief.get("tags", [])
    # Grab a short summary from the draft
    md_no_h1 = re.sub(r"^# .*\n", "", draft_md).strip()
    summary = first_sentences(md_no_h1, 480)
    hashtags = " ".join("#" + t.replace(" ", "") for t in tags[:6])

    linkedin = f"""{title}

{summary}

{hashtags}"""
    # X/Twitter ~280 chars
    x_body = (title + " — " + first_sentences(md_no_h1, 180)).strip()
    x_post = (x_body + " " + hashtags).strip()
    if len(x_post) > 280:
        x_post = x_post[:277] + "..."

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Social Snippets\n\n")
        f.write("## LinkedIn\n\n")
        f.write(linkedin + "\n\n")
        f.write("## X/Twitter\n\n")
        f.write(x_post + "\n")

def main(brief_fp: str):
    with open(brief_fp, "r", encoding="utf-8") as f:
        brief = yaml.safe_load(f)

    slug = brief["slug"]
    artifacts_dir = f"artifacts/{slug}"
    ensure_dir(artifacts_dir)

    # Build context from local index (allowlist expansion later)
    query = f"{brief['title']} {brief.get('tags','')}"
    context, _sources = retrieve(query, k=8)

    # Prepare prompt
    prompt = load_prompt(
        "engine/prompts/post_system.txt",
        "engine/prompts/post_user.txt",
        content_type=brief.get("type", "blog"),
        title=brief["title"],
        audience=brief.get("audience", "general"),
        goal=brief.get("goal", "inform"),
        tone=brief.get("tone", "direct"),
        context=context,
    )

    draft_md = ollama_complete(prompt).strip()

    # Basic audit: require some references and minimum length
    if "References" not in draft_md or len(draft_md) < 500:
        with open(f"{artifacts_dir}/failure.json", "w") as f:
            json.dump({"reason": "audit_failed_min_requirements", "len": len(draft_md)}, f, indent=2)
        print("Audit fail: missing References or too short. Saved artifacts.")
        return

    # === Guardrail 1: Allowlist enforcement for references ===
    allow = [domain(d) for d in brief.get("sources", {}).get("allow", [])]
    if allow:
        urls = extract_references(draft_md)
        ref_domains = parse_domains(urls)
        violations = [d for d in ref_domains if d not in allow]
        if violations:
            with open(f"{artifacts_dir}/references_violation.json", "w") as f:
                json.dump({
                    "reason": "allowlist_violation",
                    "allowed": allow,
                    "found": ref_domains,
                    "violations": violations
                }, f, indent=2)
            print("Reference allowlist violation. Skipping publish. Artifacts written.")
            return

    # Save markdown artifact with frontmatter
    post = frontmatter.Post(
        draft_md,
        **{
            "title": brief["title"],
            "slug": slug,
            "date": datetime.date.today().isoformat(),
            "tags": brief.get("tags", []),
        },
    )
    md_path = f"{artifacts_dir}/draft.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(frontmatter.dumps(post))

    # === Guardrail 2: Duplicate protection via cosine similarity ===
    embedder = SentenceTransformer(EMB_MODEL)
    draft_vec = embedder.encode([draft_md]).tolist()[0]

    recents = load_recent_history(n=50)
    max_sim = 0.0
    nearest = None
    for r in recents:
        sim = cosine(draft_vec, r.get("vector", []))
        if sim > max_sim:
            max_sim = sim
            nearest = r

    if max_sim >= 0.92:
        with open(f"{artifacts_dir}/duplicate.json", "w") as f:
            json.dump({
                "reason": "duplicate_detected",
                "similarity": round(max_sim, 4),
                "nearest": {"slug": nearest.get("slug"), "title": nearest.get("title"), "date": nearest.get("date")}
            }, f, indent=2)
        print(f"Duplicate detected (cos={max_sim:.3f}). Skipping publish.")
        return

    # Convert to HTML
    content_html = md_to_clean_html(draft_md)
    with open(f"{artifacts_dir}/draft.html", "w", encoding="utf-8") as f:
        f.write(content_html)

    # Optional social snippets
    if brief.get("social", False):
        generate_social(brief, draft_md, os.path.join(artifacts_dir, "social.md"))

    # Optionally publish to WordPress if env is present
    base = os.getenv("WP_BASE_URL")
    user = os.getenv("WP_USER")
    apw  = os.getenv("WP_APP_PASSWORD")
    status = brief.get("publish", {}).get("status", "draft")
    date   = brief.get("publish", {}).get("date")
    cats   = brief.get("publish", {}).get("category_ids", [])

    if base and user and apw:
        from engine.tools.wp_publish import WPPublisher
        if status not in ["draft", "publish", "future"]:
            status = "draft"
        wp = WPPublisher(base, user, apw)
        res = wp.create_post(
            title=brief["title"],
            content_html=content_html,
            status=status,
            slug=slug,
            category_ids=cats,
            date=date,
            tags=brief.get("tags", []),
        )
        with open(f"{artifacts_dir}/publish.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"WordPress post created: id={res.get('id')}, status={res.get('status')}")
        # Append to history ONLY after successful publish draft/publish/future
        append_history({
            "slug": slug,
            "title": brief.get("title"),
            "date": datetime.datetime.now().isoformat(),
            "vector": draft_vec
        })
    else:
        print("WP creds not set; skipped publishing. Artifacts written.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python engine/run.py briefs/<file>.yaml")
        sys.exit(2)
    main(sys.argv[1])

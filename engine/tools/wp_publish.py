import base64
import requests
import mimetypes
import os

class WPPublisher:
    def __init__(self, base_url: str, user: str, app_password: str):
        token = base64.b64encode(f"{user}:{app_password}".encode()).decode()
        self.headers = {"Authorization": f"Basic {token}"}
        self.base = base_url.rstrip("/")

    def upload_media(self, image_path: str, title="featured"):
        url = f"{self.base}/wp-json/wp/v2/media"
        filename = os.path.basename(image_path)
        mime = mimetypes.guess_type(filename)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            data = {"title": title}
            r = requests.post(url, headers=self.headers, files=files, data=data, timeout=60)
        r.raise_for_status()
        return r.json()["id"]

    def create_post(self, *, title, content_html, status="draft", slug=None, category_ids=None, featured_media=None, date=None, tags=None):
        url = f"{self.base}/wp-json/wp/v2/posts"
        payload = {"title": title, "content": content_html, "status": status}
        if slug: payload["slug"] = slug
        if category_ids: payload["categories"] = category_ids
        if featured_media: payload["featured_media"] = featured_media
        if date: payload["date"] = date
        if tags: payload["tags"] = tags
        r = requests.post(url, headers=self.headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

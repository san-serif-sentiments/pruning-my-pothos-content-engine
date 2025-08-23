from markdown import markdown
from bs4 import BeautifulSoup

def md_to_clean_html(md_text: str) -> str:
    html = markdown(md_text, extensions=["extra", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")
    # Basic cleanup: ensure one h1, add rel="nofollow" later if you wish
    return str(soup)

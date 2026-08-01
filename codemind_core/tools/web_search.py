"""DuckDuckGo Web Search — no API key needed."""

import re
import json
import urllib.request
import urllib.parse
from typing import List, Dict, Any


class DuckDuckGoSearch:
    DDG = "https://api.duckduckgo.com/"
    HEADERS = {"User-Agent": "CodeMind-AI/2.0"}

    def search(self, query, max_results=5):
        try:
            return self._instant(query) or self._fallback(query, max_results)
        except Exception as e:
            return [{"title": "Search Error", "snippet": str(e), "url": ""}]

    def _instant(self, query):
        params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"})
        req = urllib.request.Request(f"{self.DDG}?{params}", headers=self.HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
        results = []
        if data.get("AbstractText"):
            results.append({"title": data.get("Heading", query), "snippet": data["AbstractText"][:400], "url": data.get("AbstractURL", "")})
        for t in data.get("RelatedTopics", [])[:3]:
            if isinstance(t, dict) and t.get("Text"):
                results.append({"title": t["Text"][:80], "snippet": t["Text"][:300], "url": t.get("FirstURL", "")})
        return results

    def _fallback(self, query, max_results):
        return [{"title": f"Search: {query}", "snippet": "Install requests for better search results.", "url": f"https://duckduckgo.com/?q={urllib.parse.quote(query)}"}]

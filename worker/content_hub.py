"""Minimal Sitecore Content Hub REST client: OAuth, entity read/update, original download, Upload API v2."""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import requests

from config import Settings

log = logging.getLogger(__name__)

CHUNK_LIMIT = 10 * 1024 * 1024  # Upload API: chunks must be <= 10 MB


class ContentHubClient:
    def __init__(self, s: Settings):
        self.s = s
        self.base = s.ch_base_url
        self._token: str | None = None
        self._token_exp = 0.0
        self.http = requests.Session()

    # ---------------- auth ----------------
    def _auth_header(self) -> dict[str, str]:
        if not self._token or time.time() > self._token_exp - 60:
            data = {"client_id": self.s.ch_client_id, "client_secret": self.s.ch_client_secret,
                    "grant_type": self.s.ch_grant_type}
            if self.s.ch_grant_type == "password":
                data |= {"username": self.s.ch_username, "password": self.s.ch_password}
            r = self.http.post(f"{self.base}/oauth/token", data=data, timeout=30)
            r.raise_for_status()
            body = r.json()
            self._token = body["access_token"]
            self._token_exp = time.time() + int(body.get("expires_in", 3600))
        return {"Authorization": f"Bearer {self._token}"}

    def _req(self, method: str, path_or_url: str, **kw) -> requests.Response:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base}{path_or_url}"
        headers = kw.pop("headers", {}) | self._auth_header()
        for attempt in range(5):
            r = self.http.request(method, url, headers=headers, timeout=kw.pop("timeout", 120), **kw)
            if r.status_code in (429, 502, 503, 504):  # CH throttling / transient
                wait = int(r.headers.get("Retry-After", 2 ** attempt))
                log.warning("CH %s %s -> %s, retry in %ss", method, url, r.status_code, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                log.error("CH %s %s -> %s %s", method, url, r.status_code, r.text[:500])
            r.raise_for_status()
            return r
        r.raise_for_status()
        return r

    # ---------------- entities ----------------
    def get_entity(self, entity_id: int) -> dict[str, Any]:
        return self._req("GET", f"/api/entities/{entity_id}").json()

    def update_entity(self, entity_id: int, properties: dict | None = None,
                      relations: dict | None = None) -> None:
        body: dict[str, Any] = {}
        if properties:
            body["properties"] = properties
        if relations:
            body["relations"] = relations
        self._req("PUT", f"/api/entities/{entity_id}", json=body,
                  headers={"Content-Type": "application/json"})

    def entity_href(self, entity_id: int) -> str:
        return f"{self.base}/api/entities/{entity_id}"

    def wait_for_original(self, entity_id: int, timeout_s: int) -> tuple[dict, str]:
        """Poll until the 'downloadOriginal' rendition exists (file processing finished)."""
        deadline = time.time() + timeout_s
        while True:
            ent = self.get_entity(entity_id)
            rend = (ent.get("renditions") or {}).get("downloadOriginal") or []
            if rend and rend[0].get("href"):
                return ent, rend[0]["href"]
            if time.time() > deadline:
                raise TimeoutError(f"Asset {entity_id}: downloadOriginal rendition not ready after {timeout_s}s")
            time.sleep(10)

    def download(self, url: str) -> bytes:
        return self._req("GET", url, timeout=300).content

    # ---------------- Upload API v2 ----------------
    def upload_new_asset(self, file_name: str, data: bytes) -> int:
        """request -> process (chunked if >10 MB) -> finalize. Returns new asset id."""
        create = {
            "file_name": file_name,
            "file_size": len(data),
            "upload_configuration": {"name": "AssetUploadConfiguration"},
            "action": {"name": "NewAsset"},
        }
        r = self._req("POST", "/api/v2.0/upload", json=create,
                      headers={"Content-Type": "application/json"})
        upload_url = r.headers.get("Location") or r.headers.get("location")
        if not upload_url:
            raise RuntimeError("Upload API returned no Location header")
        session = r.json()  # upload_identifier, storage_identifier, file_identifier

        if len(data) <= CHUNK_LIMIT:
            self._req("POST", upload_url, files={"file": (file_name, data, "image/png")})
        else:
            n = math.ceil(len(data) / CHUNK_LIMIT)
            sep = "&" if "?" in upload_url else "?"
            for i in range(n):
                part = data[i * CHUNK_LIMIT:(i + 1) * CHUNK_LIMIT]
                self._req("POST", f"{upload_url}{sep}chunks={n}&chunk={i}",
                          files={"file": (file_name, part, "application/octet-stream")})
            self._req("POST", f"{upload_url}{sep}chunks={n}")  # final "assemble" call

        fin = self._req("POST", "/api/v2.0/upload/finalize", json=session,
                        headers={"Content-Type": "application/json"}).json()
        if not fin.get("success", True) or not fin.get("asset_id"):
            raise RuntimeError(f"Finalize failed for {file_name}: {fin}")
        return int(fin["asset_id"])

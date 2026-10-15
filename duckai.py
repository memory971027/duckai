"""Headless Duck.ai chat client (real Chrome + UI-driven, no IP rotation).

WHY THIS DESIGN (root-cause analysis, not guesswork):
  The same machine/IP that the user browses Duck.ai with manually works fine,
  but our Playwright relay got 418 ERR_BN_LIMIT. Two findings from js-reverse:

  1. Playwright's *bundled Chromium* is fingerprinted (TLS/HTTP2/JA3 + UA carries
     "HeadlessChrome") and is hard-banned with ERR_BN_LIMIT on the FIRST request.
     Real Chrome (`channel="chrome"`) passes that layer and instead gets
     ERR_CHALLENGE - i.e. the server is willing to talk, just wants a valid
     client-generated challenge answer.

  2. Calling fetch('/duckchat/v1/chat') by hand (bypassing the app) yields
     ERR_CHALLENGE because x-fe-signals / x-vqd-hash-1 are only produced by the
     app's OWN send logic (real interaction telemetry + challenge JS). Driving
     the real UI - type into the textarea, trigger send - lets the app generate
     the full, valid request. That path returns the real answer ("PONG" in test).

  => The fix is NOT a new IP. It is: real Chrome + drive the app's own send,
     then read the assistant reply from the intercepted chat response.

  We intercept the /duckchat/v1/chat RESPONSE (SSE) and parse `message` chunks,
  which is far more robust than scraping the DOM.

Model catalog (live /duckchat/v1/models snapshot 2026-08-27) + alias map follow.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import AsyncIterator, List, Optional

from playwright.async_api import async_playwright

logger = logging.getLogger("duckai")

BASE = os.getenv("DUCKAI_BASE", "https://duck.ai")
# Browser resolution order:
#   1) DUCKAI_CHROME_PATH env override (explicit executable)
#   2) well-known system Chrome/Chromium install locations (Linux/macOS/Windows)
#   3) Playwright's bundled Chromium (no channel) - used in containers such as
#      Render/Docker where no system Chrome exists.
_CHROME_CANDIDATES = (
    "/opt/google/chrome/chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)

def resolve_browser() -> dict:
    """Return extra kwargs for chromium.launch(): executable_path if found, else {} (bundled Chromium)."""
    env_path = os.getenv("DUCKAI_CHROME_PATH", "").strip()
    if env_path and os.path.exists(env_path):
        return {"executable_path": env_path}
    for c in _CHROME_CANDIDATES:
        if os.path.exists(c):
            return {"executable_path": c}
    return {}
# A real desktop Chrome UA (no "HeadlessChrome" marker). Version pinned to a common
# stable release; must NOT contain "HeadlessChrome".
REAL_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)

# Real backend model ids served by Duck.ai (snapshot 2026-08-27).
MODEL_LABELS = {
    "gpt-5.6-luna": "GPT-6 Luna",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "mistral-small-2603": "Mistral Small 4",
    "tinfoil/gpt-oss-120b": "gpt-oss 120B",
    "tinfoil/gemma4-31b": "Gemma 4 31B",
}
DEFAULT_MODEL = "gpt-5.6-luna"

MODEL_ALIASES = {
    "gpt-5.6-sol": "gpt-5.6-luna",
    "gpt-5.6": "gpt-5.6-luna",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-5.5": "gpt-5.4",
    "gpt-5": "gpt-5.4",
    "gpt-4o": "gpt-5.4",
    "gpt-4o-mini": "gpt-5.4-mini",
    "o3-mini": "gpt-5.4-mini",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-3.7-sonnet": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-3-haiku": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-3-opus": "claude-opus-4-8",
    "claude-opus": "claude-opus-4-8",
    "claude-opus-4-8": "claude-opus-4-8",
    "mistral-small": "mistral-small-2603",
    "mistral-small-2603": "mistral-small-2603",
    "gpt-oss-120b": "tinfoil/gpt-oss-120b",
    "tinfoil/gpt-oss-120b": "tinfoil/gpt-oss-120b",
    "gemma4-31b": "tinfoil/gemma4-31b",
    "tinfoil/gemma4-31b": "tinfoil/gemma4-31b",
}

MODEL_FAMILY = {
    "gpt-5.6": "gpt-5.6-luna",
    "gpt-5.4": "gpt-5.4",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5",
    "claude-opus": "claude-opus-4-8",
    "claude": "claude-sonnet-4-6",
    "mistral": "mistral-small-2603",
    "gpt-oss": "tinfoil/gpt-oss-120b",
    "gemma": "tinfoil/gemma4-31b",
}


def resolve_model(name: str | None) -> str:
    """Map any user-facing model name to a real Duck.ai backend id."""
    if not name:
        return DEFAULT_MODEL
    n = name.strip()
    if n in MODEL_LABELS:
        return n
    low = n.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    for k, v in MODEL_ALIASES.items():
        if k.lower() == low:
            return v
    for fam in sorted(MODEL_FAMILY, key=len, reverse=True):
        if low.startswith(fam):
            return MODEL_FAMILY[fam]
    return n


class DuckAIError(Exception):
    pass


class DuckAIRateLimit(DuckAIError):
    pass


class _Null:
    banned = False


_NULL = _Null()


class _BrowserSession:
    """One real-Chrome tab, UI-driven send, response intercepted from the wire."""

    def __init__(self, model: str, proxy: Optional[str], timeout: float) -> None:
        self.model = model
        self.proxy = proxy
        self.timeout = timeout
        self._pw = None
        self.browser = None
        self.ctx = None
        self.page = None
        self._ready = False
        self.banned = False
        # SERIALIZE all UI operations on this page. Concurrent requests (the App
        # fires several at once: message + title + retries) must NOT type into the
        # same textarea simultaneously - that's exactly what made the App fail
        # while terminal curl worked.
        self._lock = asyncio.Lock()

    async def _dismiss_consent(self) -> None:
        """duck.ai shows a first-visit privacy consent dialog with a 'Continue'
        button. Until it's clicked the real send never fires, so the reply stays
        empty (last_len=0) forever. Dismiss it by clicking any consent button."""
        try:
            for label in ("Continue", "Agree", "Accept", "同意", "继续", "Got it"):
                for btn in await self.page.query_selector_all("button"):
                    t = (await btn.inner_text() or "").strip()
                    a = await btn.get_attribute("aria-label") or ""
                    if label in t or label in a:
                        logger.info("consent: clicking %r", t or a)
                        try:
                            await btn.click(force=True)
                        except Exception:
                            pass
                        await self.page.wait_for_timeout(1500)
                        return
        except Exception as e:
            logger.info("consent: dismiss failed (%s)", str(e)[:80])

    async def _page_alive(self, timeout: float = 8.0) -> bool:
        try:
            await asyncio.wait_for(self.page.evaluate("()=>1"), timeout=timeout)
            return True
        except Exception:
            return False

    async def _rebuild(self) -> None:
        logger.info("rebuilding browser session (page died)")
        try:
            if self.browser:
                await self.browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass
        self._pw = None
        self.browser = None
        self.ctx = None
        self.page = None
        self._ready = False
        last_err = None
        # Cloud datacenter IPs are often deliberately slowed by duck.ai; retry
        # the whole warm-up a few times with growing timeouts before giving up.
        for attempt, nav_timeout in enumerate((60000, 120000, 180000), start=1):
            try:
                await self._ensure_ready(nav_timeout=nav_timeout)
                return
            except Exception as e:
                last_err = e
                logger.info("warmup attempt %d failed: %s", attempt, str(e)[:120])
                try:
                    if self.browser:
                        await self.browser.close()
                except Exception:
                    pass
                try:
                    if self._pw:
                        await self._pw.stop()
                except Exception:
                    pass
                self._pw = None
                self.browser = None
                self.ctx = None
                self.page = None
                self._ready = False
        raise DuckAIError("browser launch failed: %s" % str(last_err)[:100])

    async def _ensure_ready(self, nav_timeout: int = 60000) -> None:
        if self._ready:
            return
        self._pw = await async_playwright().start()
        launch: dict = {"headless": True}
        launch.update(resolve_browser())
        launch["args"] = ["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
        self.browser = await self._pw.chromium.launch(**launch)
        self.ctx = await self.browser.new_context(user_agent=REAL_UA)
        # hide navigator.webdriver so the app can't tell it's automated, and
        # install a fetch hook that captures the chat SSE (cloned, non-intrusive).
        await self.ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            "(function(){const O=window.fetch;"
            "window.__duckai={last:null,err:null};"
            "window.fetch=async function(u,o){const r=await O.apply(this,arguments);"
            "if(typeof u==='string'&&u.includes('/duckchat/v1/chat')){"
            "try{const c=r.clone();const t=await c.text();"
            "if(t.includes('\"action\":\"error\"')){window.__duckai.err=t;}else{window.__duckai.last=t;}}catch(e){}"
            "}return r;};})();"
        )
        # SPEED PATCH: keep ONE warm page alive across requests instead of
        # reopening+reloading duck.ai (and re-running the 7s warm-up) every ask.
        self.page = await self.ctx.new_page()
        await self.page.goto(BASE, wait_until="commit", timeout=nav_timeout)
        # WAIT FOR THE APP TO ACTUALLY BE READY (not a blind 7s sleep).
        # On slow mobile networks duck.ai's JS can take far longer than 7s.
        try:
            await self.page.wait_for_selector("textarea", timeout=max(120000, nav_timeout))
            logger.info("warmup: textarea appeared")
        except Exception as e:
            logger.info("warmup: textarea did NOT appear in 60s: %s", str(e)[:80])
        await self.page.wait_for_timeout(2000)
        await self._dismiss_consent()
        self._ready = True

    async def send_ui(self, prompt: str, timeout: float = 120.0) -> str:
        """Wrapper: convert ANY browser/playwright exception into DuckAIError so the
        retry logic in send() can rebuild the session and try again. The lock
        serializes concurrent requests to the single shared page."""
        try:
            async with self._lock:
                return await self._send_ui_inner(prompt, timeout)
        except DuckAIError:
            raise
        except Exception as e:
            raise DuckAIError("browser action failed: %s" % str(e)[:120])

    async def _send_ui_inner(self, prompt: str, timeout: float = 120.0, _retried: bool = False) -> str:
        try:
            await self._ensure_ready()
        except Exception as e:
            raise DuckAIError("browser launch failed: %s" % str(e)[:100])
        if self.page is None or not await self._page_alive():
            await self._rebuild()
        page = self.page
        try:
            # wait (up to 45s) for the textarea: slow mobile nets need this
            try:
                await page.wait_for_selector("textarea", timeout=120000)
            except Exception:
                pass
            ta = await page.query_selector("textarea")
            if ta is None:
                # last resort: reload and wait again
                logger.info("textarea missing - reloading page")
                await page.reload(wait_until="commit", timeout=45000)
                try:
                    await page.wait_for_selector("textarea", timeout=120000)
                except Exception:
                    pass
                ta = await page.query_selector("textarea")
            if ta is None:
                raise DuckAIError("duck.ai textarea not found (page layout changed)")
            await ta.click()
            await self._dismiss_consent()
            await ta.fill(prompt)
            await page.wait_for_timeout(300)
            # Send. Clicking the Ask/Send button is the reliable path (verified by
            # screenshot: the blue "Ask" button exists). Enter in the textarea is
            # only a fallback because it sometimes just inserts a newline.
            sent = False
            _all_btns = await page.query_selector_all("button")
            for btn in _all_btns:
                t = (await btn.inner_text() or "").strip().lower()
                a = (await btn.get_attribute("aria-label") or "").strip().lower()
                if ("ask" in t or "send" in t or "发送" in t or "问" in t
                        or "ask" in a or "send" in a or "发送" in a):
                    logger.info("send: clicking button text=%r aria=%r", t, a)
                    try:
                        await btn.click(force=True)
                        sent = True
                        break
                    except Exception as e:
                        logger.info("send: click failed (%s)", str(e)[:60])
            if not sent:
                logger.info("send: no button matched (%d buttons on page), pressing Enter", len(_all_btns))
                await ta.press("Enter")

            # Poll the fetch hook for the captured SSE (or an error payload).
            raw = None
            deadline = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < deadline:
                await page.wait_for_timeout(1500)
                try:
                    err, last = await asyncio.wait_for(page.evaluate(
                        "()=>[window.__duckai.err, window.__duckai.last]"
                    ), timeout=15)
                except Exception:
                    raise DuckAIError("page died mid-poll")
                logger.info("poll: err=%s last_len=%s", bool(err), len(last or ""))
                if err:
                    self._raise_err(err)
                if last:
                    raw = last
                    # stream may still be appending; wait for it to stabilize
                    await page.wait_for_timeout(1500)
                    last2 = await page.evaluate("()=>window.__duckai.last")
                    if last2 == last:
                        break
                    raw = last2
            if not raw:
                try:
                    body = await page.evaluate("()=>document.body.innerText.replace(/\\s+/g,' ').slice(0,600)")
                    logger.info("DEBUG body: %s", body)
                    await page.screenshot(path="/opt/duckai/fail.png")
                    logger.info("DEBUG screenshot saved: /opt/duckai/fail.png")
                except Exception as _de:
                    logger.info("DEBUG capture failed: %s", _de)
                if _retried:
                    raise DuckAIError("timed out waiting for Duck.ai reply")
                # HEAL: page state is stale (expired VQD token / stuck UI).
                # Reload duck.ai and retry once inside this call.
                logger.info("no reply from duck.ai - reloading page and retrying once")
                try:
                    await self.page.reload(wait_until="commit", timeout=30000)
                    await self.page.wait_for_timeout(7000)
                except Exception:
                    pass
                return await self._send_ui_inner(prompt, timeout, _retried=True)
            return self._parse_messages(raw)
        finally:
            # SPEED PATCH: do NOT close the warm page; keep it for the next ask.
            try:
                await page.evaluate("()=>{window.__duckai.last=null;window.__duckai.err=null;}")
            except Exception:
                pass

    @staticmethod
    def _parse_messages(text: str) -> str:
        chunks = []
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                # SSE: data: {...}
                if line.startswith("data:"):
                    line = line[5:].strip()
                else:
                    continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                msg = obj.get("message")
                if msg:
                    chunks.append(msg)
        return "".join(chunks)

    @staticmethod
    def _raise_err(text: str) -> None:
        m = re.search(r'"type"\s*:\s*"([^"]+)"', text)
        typ = m.group(1) if m else "error"
        if "ERR_BN_LIMIT" in text:
            raise DuckAIRateLimit("Duck.ai ERR_BN_LIMIT (IP/fingerprint banned)")
        if "ERR_CHALLENGE" in text:
            raise DuckAIError("Duck.ai ERR_CHALLENGE (challenge required)")
        raise DuckAIError(f"Duck.ai error: {typ}")

    async def close(self) -> None:
        try:
            if self.browser:
                await self.browser.close()
            if self._pw:
                await self._pw.stop()
        except Exception:
            pass


class DuckAISession:
    """Thin pool over _BrowserSession; rotates on ban.

    The fundamental fix for the ERR_BN_LIMIT the user hit is real Chrome + UI-driven
    send (see module docstring) - NOT IP rotation. Proxy is optional, kept only for
    users who genuinely need it.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        proxies: Optional[List[str]] = None,
        timeout: float = 90.0,
    ) -> None:
        self.model = model
        self.timeout = timeout
        if not proxies:
            self.proxies: List[Optional[str]] = [None]
        elif isinstance(proxies, str):
            self.proxies = [p.strip() or None for p in proxies.split(",")]
        else:
            self.proxies = [p.strip() or None for p in proxies]
        self._sessions: dict = {}
        self._order = list(range(len(self.proxies)))
        self._idx = 0

    def _next_proxy_index(self) -> Optional[int]:
        healthy = [i for i in self._order if not self._sessions.get(i, _NULL).banned]
        if not healthy:
            return None
        for i in range(len(self._order)):
            cand = (self._idx + i) % len(self._order)
            if not self._sessions.get(cand, _NULL).banned:
                self._idx = cand
                return cand
        return healthy[0]

    async def _session_for(self, idx: int) -> _BrowserSession:
        if idx not in self._sessions:
            self._sessions[idx] = _BrowserSession(self.model, self.proxies[idx], self.timeout)
        return self._sessions[idx]

    async def send(self, prompt: str, max_rotations: int = 2) -> str:
        last_err: Optional[Exception] = None
        for _ in range(max(1, max_rotations)):
            idx = self._next_proxy_index()
            if idx is None:
                raise DuckAIRateLimit("all sessions failed (Duck.ai ban)")
            try:
                return await (await self._session_for(idx)).send_ui(prompt, self.timeout)
            except DuckAIRateLimit as e:
                last_err = e
                self._sessions.get(idx).banned = True
                continue
            except DuckAIError as e:
                # page 劣化/超时/浏览器异常：销毁坏 session，下一轮用全新 session 重试一次
                last_err = e
                old = self._sessions.pop(idx, None)
                if old is not None:
                    try:
                        await old.close()
                    except Exception:
                        pass
                continue
        raise last_err or DuckAIRateLimit("all sessions failed")

    async def send_stream(self, prompt: str, max_rotations: int = 2) -> AsyncIterator[str]:
        """Duck.ai UI is not natively streamable here; we return the full reply as
        one chunk. BUT the full reply can take 50-90s on a cold page, and streaming
        clients (the App) time out waiting for the first byte. So we run send() in a
        background task and emit empty keep-alive chunks every few seconds until the
        real reply is ready - this keeps the SSE connection alive the whole time."""
        task = asyncio.create_task(self.send(prompt, max_rotations))
        while not task.done():
            await asyncio.sleep(5)
            yield ""  # keep-alive: empty delta, client sees activity, no text shown
        text = await task
        yield text

    async def close(self) -> None:
        for s in self._sessions.values():
            await s.close()

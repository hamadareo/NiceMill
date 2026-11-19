"""The resumable uploader of the drop zone (phone-first lane, audit #4).

Shown in place of Streamlit's uploader when the server has the upload endpoint
(ui/upload_api.MOUNTED, i.e. the app runs from ui/serve.py). The browser sends each file in
8 MiB pieces to /pfb-upload with this session's token (services.chunked_upload). When the
connection drops, or the phone sleeps, it waits and goes on from the last piece the server
has; picking the same file again after a reload also goes on (the upload id and its token are
kept in the browser's localStorage per file name, size and date). While it sends, the screen
is kept awake where the browser allows it (Wake Lock, https only) and leaving the page asks
first. Finished files are reported to Python (`files` state); the drop zone claims them when
the coach presses its primary.

Nothing is sent before the coach picks a file; every word on it is Japanese; the component
draws in the theme's tokens (CSS variables cross the shadow root).
"""
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from pfb_app.config import Settings
from pfb_app.services import chunked_upload as cu
from pfb_app.ui import upload_api

TOKEN = "_pfb_upload_token"
RESET = "_pfb_upload_reset_{}"   # bumped after a drop: the next uploader starts empty
TOKEN_RENEW_SEC = 3600           # a token with less than this left is replaced

TEXTS = {
    "pick": "ファイルを選ぶ",
    "drop": "ここにファイルを置くか、「ファイルを選ぶ」を押してください。",
    "note": "大きな動画も送れます。途中で通信が切れても、つながると続きから送ります。",
    "sending": "送っています",
    "resuming": "続きから送っています",
    "done": "送りました",
    "offline": "通信が切れました。つながると続きから送ります。",
    "retry": "うまく送れませんでした。少し待って続きから送ります。",
    "failed": "送れませんでした",
    "remove": "取り消す",
    "left": "残り約{m}分",
    "leave": "まだ送っています。このページを離れると、あとで続きから送ることになります。",
    "too_many": "動画は4つまでです。",
}

def _css() -> str:
    """The uploader's CSS in the theme's tokens (CSS variables cross the shadow root; the
    fallbacks are the same tokens, from theme.TOKENS, for a page without the theme)."""
    from pfb_app.ui.wizard import theme   # noqa: PLC0415
    v = lambda name: f"var(--pfb-{name}, {theme.TOKENS[name]})"   # noqa: E731
    return f"""
.pfb-bu {{ font-family: var(--pfb-font-ui, sans-serif); color: {v("ink")}; }}
.pfb-bu__zone {{ border: 2px dashed {v("line")}; border-radius: 12px; padding: 1rem;
  background: transparent; display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: center; }}
.pfb-bu__zone--over {{ background: {v("mist")}; border-color: {v("board")}; }}
.pfb-bu__pick {{ font: inherit; font-weight: 700; color: {v("board")}; background: transparent;
  border: 1.5px solid {v("board")}; border-radius: 9px; min-height: 2.75rem;
  padding: 0.5rem 1.25rem; cursor: pointer; }}
.pfb-bu__pick:focus-visible, .pfb-bu__remove:focus-visible {{ outline: 3px solid {v("sky")};
  outline-offset: 2px; }}
.pfb-bu__hint {{ flex: 1 1 12rem; min-width: 0; color: {v("muted")}; font-size: 0.9375rem; }}
.pfb-bu__note {{ margin: 0.5rem 0 0; color: {v("muted")}; font-size: 0.875rem; }}
.pfb-bu__list {{ list-style: none; margin: 0.75rem 0 0; padding: 0; display: grid; gap: 0.5rem; }}
.pfb-bu__item {{ border: 1px solid {v("line")}; border-radius: 10px; padding: 0.6rem 0.75rem;
  background: transparent; display: grid; grid-template-columns: minmax(0, 1fr) auto;
  gap: 0.35rem 0.75rem; align-items: center; }}
.pfb-bu__name {{ font-weight: 700; overflow-wrap: anywhere; min-width: 0; }}
.pfb-bu__state {{ grid-column: 1 / -1; font-size: 0.875rem; color: {v("muted")}; }}
.pfb-bu__state--bad {{ color: {v("ink")}; font-weight: 700; }}
.pfb-bu__bar {{ grid-column: 1 / -1; height: 0.5rem; border-radius: 999px;
  background: {v("line")}; overflow: hidden; }}
.pfb-bu__fill {{ height: 100%; width: 0; background: {v("sky")}; transition: width 0.3s; }}
.pfb-bu__item--done .pfb-bu__fill {{ background: {v("board")}; }}
.pfb-bu__remove {{ font: inherit; font-size: 0.875rem; color: {v("ink")}; background: none;
  border: none; text-decoration: underline; cursor: pointer; min-height: 2.75rem;
  padding: 0 0.25rem; }}
@media (prefers-reduced-motion: reduce) {{ .pfb-bu__fill {{ transition: none; }} }}
"""

_HTML = """
<div class="pfb-bu">
  <div class="pfb-bu__zone" data-zone>
    <button type="button" class="pfb-bu__pick" data-pick></button>
    <span class="pfb-bu__hint" data-hint></span>
    <input type="file" data-input multiple hidden>
  </div>
  <p class="pfb-bu__note" data-note></p>
  <ul class="pfb-bu__list" data-list aria-live="polite"></ul>
</div>
"""

# One global store per browser tab (window.__pfbBigUpload[key]) keeps the uploads going across
# Streamlit reruns; the component only draws it and reports finished files.
_JS = r"""
export default function (component) {
  const { data, setStateValue, parentElement } = component;
  const T = data.texts;
  const root = parentElement.querySelector(".pfb-bu");
  const all = (window.__pfbBigUpload = window.__pfbBigUpload || {});
  const store = (all[data.key] = all[data.key] || { items: [], token: data.token, wake: null });
  store.token = data.token;
  const $ = (s) => root.querySelector(s);
  const input = $("[data-input]"), zone = $("[data-zone]"), list = $("[data-list]");
  $("[data-pick]").textContent = T.pick;
  $("[data-hint]").textContent = T.drop;
  $("[data-note]").textContent = T.note;
  input.accept = data.accept.join(",");
  input.multiple = data.multiple;

  const memo = (f) => "pfb-upload:" + f.name + ":" + f.size + ":" + f.lastModified;
  const remember = (it) => { try { localStorage.setItem(memo(it.file),
      JSON.stringify({ id: it.id, token: it.token })); } catch (e) {} };
  const forget = (it) => { try { localStorage.removeItem(memo(it.file)); } catch (e) {} };
  const recall = (f) => { try { return JSON.parse(localStorage.getItem(memo(f)) || "null"); }
                          catch (e) { return null; } };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const online = () => navigator.onLine !== false ? Promise.resolve() :
      new Promise((r) => window.addEventListener("online", r, { once: true }));
  // every request can be cut: by its own time limit (a piece gets at least 100 kB/s), or
  // at once when the device goes offline (a dropped connection may otherwise hang for minutes)
  const api = (it, method, path, body, headers) => {
    const ctrl = new AbortController();
    const size = body && body.size ? body.size : 0;
    const timer = setTimeout(() => ctrl.abort(), Math.max(60000, size / 100));
    it.ctrl = ctrl;
    return fetch(data.endpoint + path, {
      method, body, credentials: "same-origin", cache: "no-store", signal: ctrl.signal,
      headers: Object.assign({ "X-PFB-Token": it.token }, headers || {}) })
      .finally(() => { clearTimeout(timer); if (it.ctrl === ctrl) it.ctrl = null; });
  };

  function report() {
    const done = store.items.filter((it) => it.state === "done")
      .map((it) => ({ id: it.id, token: it.token, name: it.file.name, size: it.file.size }));
    const key = JSON.stringify(done);
    if (key !== store.reported) { store.reported = key; setStateValue("files", done); }
  }
  function busy() { return store.items.some((it) => it.state === "sending"); }
  async function wake(on) {
    try {
      if (on && !store.wake && navigator.wakeLock && window.isSecureContext) {
        store.wake = await navigator.wakeLock.request("screen");
        store.wake.addEventListener("release", () => { store.wake = null; });
      } else if (!on && store.wake) { await store.wake.release(); store.wake = null; }
    } catch (e) { store.wake = null; }
  }
  if (!store.hooked) {
    store.hooked = true;
    window.addEventListener("beforeunload", (e) => {
      if (busy()) { e.preventDefault(); e.returnValue = T.leave; return T.leave; }
    });
    window.addEventListener("offline", () => {
      for (const it of store.items) { if (it.ctrl) it.ctrl.abort(); }
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible" && busy()) wake(true);
    });
  }

  function minutesLeft(it) {
    if (!it.started || it.offset <= it.startOffset) return "";
    const rate = (it.offset - it.startOffset) / ((Date.now() - it.started) / 1000);
    if (rate <= 0) return "";
    return T.left.replace("{m}", Math.max(1, Math.ceil((it.file.size - it.offset) / rate / 60)));
  }
  function draw() {
    list.textContent = "";
    for (const it of store.items) {
      const li = document.createElement("li");
      li.className = "pfb-bu__item" + (it.state === "done" ? " pfb-bu__item--done" : "");
      const name = document.createElement("span");
      name.className = "pfb-bu__name"; name.textContent = it.file.name;
      const rm = document.createElement("button");
      rm.type = "button"; rm.className = "pfb-bu__remove"; rm.textContent = T.remove;
      rm.setAttribute("aria-label", it.file.name + " " + T.remove);
      rm.onclick = () => remove(it);
      const bar = document.createElement("div"); bar.className = "pfb-bu__bar";
      bar.setAttribute("role", "progressbar"); bar.setAttribute("aria-label", it.file.name);
      const pct = it.file.size ? Math.floor((100 * it.offset) / it.file.size) : 0;
      bar.setAttribute("aria-valuemin", "0"); bar.setAttribute("aria-valuemax", "100");
      bar.setAttribute("aria-valuenow", String(pct));
      const fill = document.createElement("div"); fill.className = "pfb-bu__fill";
      fill.style.width = pct + "%"; bar.appendChild(fill);
      const state = document.createElement("span");
      state.className = "pfb-bu__state" + (it.state === "failed" ? " pfb-bu__state--bad" : "");
      let text = "";
      if (it.state === "done") text = T.done;
      else if (it.state === "failed") text = T.failed + "：" + (it.error || "");
      else if (it.note) text = it.note + "（" + pct + "%）";
      else text = (it.resumed ? T.resuming : T.sending) + " " + pct + "%" +
                  (minutesLeft(it) ? "（" + minutesLeft(it) + "）" : "");
      state.textContent = text;
      li.append(name, rm, bar, state);
      list.appendChild(li);
    }
  }

  async function send(it) {
    let wait = 1000;
    it.state = "sending"; wake(true); store.draw();
    while (it.state === "sending") {
      try {
        await online();
        if (!it.id) {
          const r = await api(it, "POST", "", JSON.stringify({ name: it.file.name, size: it.file.size }),
                              { "Content-Type": "application/json" });
          const j = await r.json();
          if (r.status === 429 || r.status >= 500) throw new Error(j.error);
          if (!r.ok) { it.state = "failed"; it.error = j.error; break; }
          it.id = j.id; it.chunk = j.chunk; it.offset = j.offset; remember(it);
        } else if (it.offset === undefined || it.note) {
          const r = await api(it, "GET", "/" + it.id);
          const j = await r.json();
          if (r.status === 404 || r.status === 401) {     // gone or too old: start again
            forget(it); it.id = null; it.offset = 0; it.resumed = false; continue;
          }
          if (!r.ok) throw new Error(j.error);
          it.offset = j.offset;
          if (j.complete) { it.state = "done"; break; }
        }
        it.note = ""; it.started = it.started || Date.now();
        it.startOffset = it.startOffset === undefined ? it.offset : it.startOffset;
        store.draw();
        const piece = it.file.slice(it.offset, it.offset + (it.chunk || data.chunk));
        const r = await api(it, "PATCH", "/" + it.id, piece, { "Upload-Offset": String(it.offset) });
        const j = await r.json();
        if (r.status === 429 || r.status === 408 || r.status >= 500) throw new Error(j.error);
        if (r.status === 409) { it.offset = j.offset; continue; }
        if (r.status === 404 || r.status === 401) { forget(it); it.id = null; it.offset = 0; continue; }
        if (!r.ok) { it.state = "failed"; it.error = j.error; break; }
        it.offset = j.offset; wait = 1000;
        if (j.complete) { it.state = "done"; forget(it); }
        store.draw();
      } catch (e) {                  // the network: wait, then ask the server where to go on
        it.note = navigator.onLine === false ? T.offline : T.retry;
        store.draw();
        await sleep(wait); wait = Math.min(wait * 2, 30000);
      }
    }
    store.draw(); store.report();
    if (!busy()) wake(false);
  }

  async function remove(it) {
    const was = it.state; it.state = "removed";
    store.items = store.items.filter((x) => x !== it);
    forget(it); store.draw(); store.report();
    if (it.id && was !== "failed") {
      try { await api(it, "DELETE", "/" + it.id); } catch (e) {}
    }
    if (!busy()) wake(false);
  }

  function add(files) {
    let videos = store.items.filter((it) => data.videos.some((x) => it.file.name.toLowerCase().endsWith(x))).length;
    for (const f of files) {
      if (store.items.some((it) => memo(it.file) === memo(f))) continue;
      const isVideo = data.videos.some((x) => f.name.toLowerCase().endsWith(x));
      if (isVideo && videos >= data.maxVideos) { alert(T.too_many); continue; }
      videos += isVideo ? 1 : 0;
      const it = { file: f, token: store.token, state: "waiting", offset: undefined };
      const old = recall(f);
      if (old && old.id) { it.id = old.id; it.token = old.token; it.resumed = true; it.note = T.resuming; }
      store.items.push(it);
      send(it);
    }
    draw();
  }

  // the upload loops started by an earlier run of this function draw and report through
  // the store, so they always reach the component as it is now
  store.draw = draw;
  store.report = report;
  $("[data-pick]").onclick = () => input.click();
  input.onchange = () => { add(Array.from(input.files || [])); input.value = ""; };
  zone.ondragover = (e) => { e.preventDefault(); zone.classList.add("pfb-bu__zone--over"); };
  zone.ondragleave = () => zone.classList.remove("pfb-bu__zone--over");
  zone.ondrop = (e) => { e.preventDefault(); zone.classList.remove("pfb-bu__zone--over");
                         add(Array.from((e.dataTransfer && e.dataTransfer.files) || [])); };
  draw(); report();
}
"""

_component = None


def _renderer():
    global _component
    if _component is None:
        _component = st.components.v2.component("pfb_big_upload", html=_HTML, css=_css(),
                                                js=_JS)
    return _component


def available() -> bool:
    """The endpoint is served by this process (the app runs from ui/serve.py)."""
    return upload_api.MOUNTED


def session_token(settings: Settings) -> str:
    """This session's upload token, renewed when it is about to expire or the person's auth
    session changed (a new sign-in or password): it carries that session (docs/accounts.md)."""
    import time   # noqa: PLC0415
    from pfb_app.ui import access   # noqa: PLC0415
    token = st.session_state.get(TOKEN)
    issued = int(token.split(".")[0]) if isinstance(token, str) and token[:1].isdigit() else 0
    sid = access.session_id()
    if cu.check_token(settings, token) is None \
            or cu.token_session(settings, token) != sid \
            or time.time() - issued > cu.TOKEN_TTL_SEC - TOKEN_RENEW_SEC:
        token = cu.issue_token(settings, session_id=sid)
        st.session_state[TOKEN] = token
        cu.sweep(settings)   # once per new token: old parts and unclaimed files go
        from pfb_app.services import proxy   # noqa: PLC0415
        proxy.maybe_sweep(settings)
    return token


@dataclass
class BigPick:
    """A finished resumable upload, as a pick of the drop zone: `.name` like an uploaded file;
    `claim()` hands its file over (once) for registration."""
    id: str
    token: str
    name: str
    size: int

    def path(self, settings: Settings) -> Path:
        """Its finished file, still this upload's (a failed registration keeps it)."""
        return cu.finished_path(settings, self.token, self.id)

    def claim(self, settings: Settings) -> Path:
        return cu.claim(settings, self.token, self.id)

    def read_and_remove(self, settings: Settings) -> bytes:
        """A document: its bytes (the drop zone keeps them until the work exists)."""
        path = self.claim(settings)
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)


def finished(settings: Settings, entries) -> list[BigPick]:
    """The browser's finished files that are really finished on the server for their token
    (anything else, e.g. a forged entry, is left out)."""
    picks = []
    for e in entries or []:
        try:
            up = cu.status(settings, str(e["token"]), str(e["id"]))
        except (cu.UploadError, KeyError, TypeError):
            continue
        if up.complete:
            picks.append(BigPick(up.id, str(e["token"]), up.name, up.size))
    return picks


def uploader(settings: Settings, key: str, extensions: list[str], *, max_videos: int,
             videos: tuple[str, ...]) -> list[BigPick]:
    """Draw the uploader; returns the finished picks."""
    generation = st.session_state.get(RESET.format(key), 0)
    data = {"key": f"{key}:{generation}", "token": session_token(settings),
            "endpoint": upload_api.PREFIX, "chunk": upload_api.CHUNK_BYTES,
            "accept": list(extensions), "multiple": True, "maxVideos": max_videos,
            "videos": list(videos), "texts": TEXTS}
    result = _renderer()(key=f"{key}_big_{generation}", data=data,
                         on_files_change=lambda: None, default={"files": []})
    entries = getattr(result, "files", None) or []
    return finished(settings, entries)


def reset(key: str) -> None:
    """After a drop: the next uploader under `key` starts empty."""
    name = RESET.format(key)
    st.session_state[name] = st.session_state.get(name, 0) + 1


"""Regression guards for the Firefox extension's capture-frame invariants.

These are the invariants that were each broken once during development, and every break
produced the *same misleading symptom*: the popup reported

    "Không tìm thấy video nào (Hãy bấm Play video trước)"

even though the page was playing a video in a cross-origin player iframe (e.g.
`play.vlstream.net` embedded in `vlxx.phd`).

Each test below documents one of those hard-won constraints:

1.  The extension MUST hold host permission for the iframe's origin. `activeTab` only
    covers the active tab's TOP-LEVEL document, never child frames, so without it the
    player iframe is invisible to every discovery mechanism.
2.  `<all_urls>` MUST be in `optional_host_permissions` (not `host_permissions`) so that
    `permissions.request()` can be called from the popup. Declaring it in both places is
    a manifest error.
3.  The content scripts MUST be declared with `all_frames: true`, otherwise no frame other
    than the top one is instrumented.
4.  The popup MUST ask for host access BEFORE any `await` in the START click handler --
    Firefox only shows the permission prompt inside a user-gesture handler.
5.  The popup MUST be able to inject the content scripts itself, because declarative
    content scripts are not retroactively injected into documents that were already
    loaded when the permission was granted.
6.  `CONTENT_SCRIPT_FILES` in the popup MUST stay in sync with the manifest, otherwise the
    self-injection path silently loads an incomplete or stale script set.
7.  A frame with nothing to capture MUST stay silent when it receives a broadcast START.
    `tabs.sendMessage` without a frameId returns only the FIRST response, so a video-less
    top frame answering first would mask the frame that actually started.
8.  The per-frame discovery/targeting channel (`webNavigation.getAllFrames` +
    `tabs.sendMessage({frameId})`) MUST exist, because it is the only reliable way to find
    and reach a cross-origin iframe.
9.  The backend admission guard MUST stay "newest wins": rejecting a newcomer would let one
    stale session lock START out forever.
"""

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_DIR = PROJECT_ROOT / "extension_firefox"
MANIFEST_PATH = EXTENSION_DIR / "manifest.json"
POPUP_PATH = EXTENSION_DIR / "popup" / "popup.js"
CONTENT_SCRIPT_PATH = EXTENSION_DIR / "content" / "content-script.js"


@pytest.fixture(scope="module")
def manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def popup_source() -> str:
    return POPUP_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def content_script_source() -> str:
    return CONTENT_SCRIPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1-3. Manifest: permissions, optional host permissions, all_frames
# ---------------------------------------------------------------------------
def test_all_urls_is_an_optional_host_permission(manifest):
    """`<all_urls>` must be requestable at runtime, so it belongs in optional_host_permissions."""
    optional = manifest.get("optional_host_permissions", [])
    assert "<all_urls>" in optional, (
        "`<all_urls>` must be declared in optional_host_permissions so the popup can call "
        "permissions.request(); without it the extension cannot instrument a cross-origin "
        "player iframe."
    )


def test_all_urls_is_not_declared_twice(manifest):
    """Declaring the same origin in both host_permissions and optional_host_permissions is an error."""
    required = manifest.get("host_permissions", [])
    optional = manifest.get("optional_host_permissions", [])
    overlap = set(required) & set(optional)
    assert not overlap, f"origin declared in both host_permissions and optional_host_permissions: {overlap}"


def test_required_permissions_present(manifest):
    permissions = set(manifest.get("permissions", []))
    for needed in ("activeTab", "storage", "scripting", "webNavigation"):
        assert needed in permissions, (
            f"missing '{needed}' permission. webNavigation is required to enumerate frames "
            f"(the only way to reach a cross-origin iframe); scripting is required for the "
            f"self-injection fallback."
        )


def test_content_script_is_declared_for_every_frame(manifest):
    scripts = manifest.get("content_scripts", [])
    assert scripts, "no content_scripts declared"

    entry = scripts[0]
    assert entry.get("all_frames") is True, (
        "content_scripts[0].all_frames must be true: the capture owner is usually a "
        "cross-origin player iframe, and without this no frame other than the top one gets "
        "the content script."
    )
    assert "<all_urls>" in entry.get("matches", []), "content_scripts[0] must match <all_urls>"
    assert entry.get("js"), "content_scripts[0].js must list the script files"


# ---------------------------------------------------------------------------
# 4. Host access must be requested before any await in the START handler
# ---------------------------------------------------------------------------
def test_start_handler_requests_host_access_before_awaiting(popup_source):
    """Firefox only shows the permission prompt inside a user-gesture handler.

    The host-access request must therefore be the FIRST awaited operation in the START
    handler -- any earlier `await` consumes the user gesture and the prompt is refused.
    """
    handler_start = popup_source.index('btnStart.addEventListener("click"')
    handler_body = popup_source[handler_start:]

    first_await_index = handler_body.find("await ")
    assert first_await_index != -1, "the START handler should contain at least one await"

    first_awaited_call = handler_body[first_await_index:first_await_index + 80]
    assert "ensureHostAccess" in first_awaited_call, (
        "the FIRST await in the START handler must be the host-access request, but found: "
        f"{first_awaited_call.strip()!r}. Awaits consume the user gesture, after which Firefox "
        "refuses to show the permission prompt."
    )


# ---------------------------------------------------------------------------
# 5-6. Self-injection and manifest/popup script-list sync
# ---------------------------------------------------------------------------
def _extract_js_array(source: str, const_name: str) -> list:
    """Extract a `const NAME = ["a.js", "b.js"];` string array from JavaScript source."""
    match = re.search(rf"const\s+{const_name}\s*=\s*\[(.*?)\]\s*;", source, re.DOTALL)
    if not match:
        raise AssertionError(f"could not find '{const_name}' array in popup.js")
    return re.findall(r'"([^"]+)"', match.group(1))


def test_popup_can_inject_content_scripts(popup_source):
    """Declarative scripts are not retroactively injected, so the popup must inject them itself."""
    assert "async function injectContentScripts" in popup_source, (
        "injectContentScripts() is required: a page loaded before host access was granted has "
        "NO content script in any frame, and every discovery channel then returns nothing."
    )
    assert "injectContentScripts(tab)" in popup_source, "injectContentScripts() must actually be called"


def test_popup_file_list_matches_manifest(popup_source, manifest):
    """A drift between these two lists silently breaks the self-injection fallback."""
    declared = manifest["content_scripts"][0]["js"]
    used_by_popup = _extract_js_array(popup_source, "CONTENT_SCRIPT_FILES")
    assert used_by_popup == declared, (
        "CONTENT_SCRIPT_FILES in popup.js must match manifest.json content_scripts[0].js "
        f"in the same order.\n  manifest: {declared}\n  popup:    {used_by_popup}"
    )


# ---------------------------------------------------------------------------
# 7. A frame with nothing to capture must stay silent on broadcast
# ---------------------------------------------------------------------------
def test_content_script_stays_silent_without_a_video(content_script_source):
    """A broadcast returns only the FIRST response, so a no-op frame must not answer."""
    assert re.search(
        r"if\s*\(\s*!captureOwnerToken\s*&&\s*!findVideo\(\)\s*\)\s*\{\s*return false;",
        content_script_source,
    ), (
        "the START_TRANSLATION handler must `return false` (no response) when the frame has "
        "no owner token and no video. Otherwise a video-less top frame answers first and "
        "masks the frame that actually started."
    )


# ---------------------------------------------------------------------------
# 8. Per-frame messaging channel must exist on both sides
# ---------------------------------------------------------------------------
def test_per_frame_messaging_channel_exists(popup_source, content_script_source):
    assert "webNavigation" in popup_source and "getAllFrames" in popup_source, (
        "the popup must enumerate frames via webNavigation.getAllFrames to discover and target "
        "a cross-origin player iframe."
    )
    assert "sendToFrame" in popup_source, "the popup must be able to message a single frame"

    for action in ('"DISCOVER_CAPTURE"', '"RELEASE_CAPTURE_OWNER"'):
        assert action in content_script_source, f"content script must handle {action}"
        assert action in popup_source, f"popup must send {action}"

    assert "__ownerToken" in popup_source and "__ownerToken" in content_script_source, (
        "the capture-owner token handshake is how a targeted frame accepts a START while "
        "other frames refuse."
    )


# ---------------------------------------------------------------------------
# 9. Backend admission guard must stay newest-wins
# ---------------------------------------------------------------------------
def test_backend_admission_guard_is_newest_wins():
    import backend_cpp.ws.ws_handler as ws_handler
    from backend_cpp.config import config

    assert config.ws.max_sessions == 1, "the single-session product invariant must stay on by default"
    assert hasattr(ws_handler, "_supersede_excess_sessions"), (
        "the admission guard must SUPERSEDE the oldest session rather than reject the "
        "newcomer; rejecting a newcomer lets one stale session lock START out forever."
    )
    assert hasattr(ws_handler, "_active_sessions")
    with ws_handler._active_sessions_lock:
        assert ws_handler._active_sessions == {} or hasattr(ws_handler._active_sessions, "popitem"), (
            "_active_sessions must be an insertion-ordered mapping so the OLDEST session can be evicted"
        )

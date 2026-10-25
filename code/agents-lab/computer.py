"""Part 9: computer-use and browser agents on top of the lab harness.

Two client toolsets from the Claude API, each executed against a headless Chromium (Playwright):

  computer_toolset_20260801  pixels only: screenshots in, mouse and keyboard actions out
  browser_toolset_20260801   a browser: navigate, read_page (an accessibility tree with element refs),
                             find, form_input, clicks by ref or by coordinate, and (if enabled) screenshots

The loop is Part 1's loop with three changes: the tools are one toolset entry, each tool_use block
carries a `toolset_name` that its tool_result must echo, and the blocks of one turn run in order
and stop at the first failure. harness.py and ledger.py are imported, not changed.
"""

import base64
import io
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
import ledger  # noqa: E402
from harness import Result, Step, Transcript, _TimedClient  # noqa: E402

W, H = 1280, 800  # viewport = screenshot size, so Claude's coordinates need no scaling
HALT = {"computer": "Not executed: an earlier computer action in this turn failed.",
        "browser": "Not executed: an earlier action in this turn failed."}
TABS = ["new_tab", "list_tabs", "switch_tab", "close_tab"]


def computer_toolset(zoom=True):
    entry = {"type": "computer_toolset_20260801"}
    if not zoom:
        entry["configs"] = {"zoom": {"enabled": False}}
    return entry


def browser_toolset(screenshots=True):
    """Default members minus the four tab members (one tab only); optionally minus screenshot and zoom."""
    off = TABS + ([] if screenshots else ["screenshot", "zoom"])
    return {"type": "browser_toolset_20260801", "configs": {m: {"enabled": False} for m in off}}


# ---- the browser under test ---------------------------------------------------------------------

class Page:
    """One fresh browser context per run, locked to the local site."""

    def __init__(self, browser, base, run_id, start="/"):
        self.base = base.rstrip("/")
        host = urlparse(self.base)
        self.allowed = (host.hostname, host.port)
        self.ctx = browser.new_context(viewport={"width": W, "height": H}, device_scale_factor=2)
        self.ctx.add_cookies([{"name": "run", "value": run_id, "url": self.base}])
        self.ctx.route("**/*", self._guard)  # every request that leaves 127.0.0.1:<port> is aborted
        self.page = self.ctx.new_page()
        self.page.goto(self.base + start)
        self.mouse = (W // 2, H // 2)
        self.blocked = []

    def _guard(self, route):
        u = urlparse(route.request.url)
        if u.scheme in ("http", "https") and (u.hostname, u.port) == self.allowed:
            return route.continue_()
        self.blocked.append(route.request.url)
        return route.abort()

    def settle(self, url_before=None):
        """Wait until the document is complete and the URL has stopped changing (max 6 s)."""
        time.sleep(0.15)
        end, stable, last = time.time() + 6, 0, None
        while time.time() < end:
            try:
                state, cur = self.page.evaluate("document.readyState"), self.page.url
            except Exception:  # mid-navigation: the old document is gone
                stable, last = 0, None
                time.sleep(0.1)
                continue
            stable = stable + 1 if (state == "complete" and cur == last) else 0
            last = cur
            if stable >= 2:
                break
            time.sleep(0.1)
        return url_before is not None and self.page.url != url_before

    def png(self):
        return self.page.screenshot(type="png", scale="css")  # W x H pixels

    def zoom_png(self, region):
        x0, y0, x1, y1 = [float(v) for v in region]
        x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1))
        x1, y1 = max(x0 + 1, min(x1, W)), max(y0 + 1, min(y1, H))
        raw = self.page.screenshot(type="png", scale="device", clip={"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0})
        from PIL import Image
        im = Image.open(io.BytesIO(raw))
        f = min(W / im.width, H / im.height)  # fit the usual screenshot size, aspect ratio kept
        im = im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def close(self):
        self.ctx.close()


def image_block(png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(png).decode()}}


KEYS = {"return": "Enter", "enter": "Enter", "backspace": "Backspace", "page_down": "PageDown", "pagedown": "PageDown",
        "page_up": "PageUp", "pageup": "PageUp", "escape": "Escape", "esc": "Escape", "tab": "Tab", "home": "Home",
        "end": "End", "delete": "Delete", "up": "ArrowUp", "down": "ArrowDown", "left": "ArrowLeft",
        "right": "ArrowRight", "space": "Space", "ctrl": "Control", "control": "Control", "alt": "Alt",
        "shift": "Shift", "super": "Meta", "cmd": "Meta", "meta": "Meta"}


def pw_key(combo):
    """xdotool-style 'ctrl+a' or 'Page_Down' to Playwright's 'Control+a' / 'PageDown'."""
    out = []
    for k in combo.split("+"):
        kl = k.strip().lower()
        out.append(KEYS.get(kl, k.strip() if len(k.strip()) > 1 else k.strip()))
    return "+".join(out)


class ActionError(Exception):
    pass


# ---- executor 1: the computer toolset (pixels) -----------------------------------------------------

def downscale(png, f):
    """Shrink a screenshot by factor f (2 = half width and height) before it goes to the model."""
    if f == 1:
        return png
    from PIL import Image
    im = Image.open(io.BytesIO(png))
    im = im.resize((round(im.width / f), round(im.height / f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class ComputerExecutor:
    toolset = "computer"

    def __init__(self, page, scale=1):
        self.p = page
        self.pg = page.page
        self.f = scale  # screenshots are sent at 1/scale size; Claude's coordinates are multiplied back up

    def _xy(self, inp, key="coordinate"):
        c = inp.get(key)
        if c is None:
            return self.p.mouse
        x, y = round(c[0] * self.f), round(c[1] * self.f)
        if not (0 <= x < W and 0 <= y < H):
            raise ActionError(f"Coordinates ({x}, {y}) are outside the {W}x{H} screen.")
        return x, y

    def _mods(self, text):
        return [pw_key(m) for m in (text or "").split("+") if m]

    def run(self, name, inp):
        pg, before = self.pg, self.pg.url
        if name == "screenshot":
            return [image_block(downscale(self.p.png(), self.f))]
        if name == "zoom":
            return [image_block(downscale(self.p.zoom_png([v * self.f for v in inp["region"]]), self.f))]
        if name in ("left_click", "right_click", "middle_click", "double_click", "triple_click"):
            x, y = self._xy(inp)
            mods = self._mods(inp.get("text"))
            for m in mods:
                pg.keyboard.down(m)
            button = {"right_click": "right", "middle_click": "middle"}.get(name, "left")
            count = {"double_click": 2, "triple_click": 3}.get(name, 1)
            pg.mouse.click(x, y, button=button, click_count=count)
            for m in mods:
                pg.keyboard.up(m)
            self.p.mouse = (x, y)
        elif name == "mouse_move":
            x, y = self._xy(inp)
            pg.mouse.move(x, y)
            self.p.mouse = (x, y)
        elif name == "left_click_drag":
            x0, y0 = self._xy(inp, "start_coordinate")
            x1, y1 = self._xy(inp)
            pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(x1, y1, steps=8); pg.mouse.up()
            self.p.mouse = (x1, y1)
        elif name == "left_mouse_down":
            pg.mouse.down()
        elif name == "left_mouse_up":
            pg.mouse.up()
        elif name == "cursor_position":
            return [{"type": "text", "text": f"X={round(self.p.mouse[0] / self.f)}, Y={round(self.p.mouse[1] / self.f)}"}]
        elif name == "scroll":
            x, y = self._xy(inp)
            pg.mouse.move(x, y)
            self.p.mouse = (x, y)
            n = int(inp.get("scroll_amount", 3)) * 100
            d = inp["scroll_direction"]
            pg.mouse.wheel(n if d == "right" else -n if d == "left" else 0, n if d == "down" else -n if d == "up" else 0)
        elif name == "type":
            pg.keyboard.type(inp["text"], delay=5)
        elif name == "key":
            k = inp["text"]
            if k.lower() in ("alt+left", "alt+arrowleft"):
                pg.go_back()
            else:
                for _ in range(int(inp.get("repeat", 1))):
                    pg.keyboard.press(pw_key(k))
        elif name == "hold_key":
            k = pw_key(inp["text"])
            pg.keyboard.down(k); time.sleep(min(float(inp["duration"]), 5)); pg.keyboard.up(k)
        elif name == "wait":
            time.sleep(min(float(inp["duration"]), 5))
        else:
            raise ActionError(f"Unknown action {name}")
        self.p.settle(before)
        return [{"type": "text", "text": "OK"}]


# ---- executor 2: the browser toolset (accessibility tree + optional pixels) -----------------------------

INTERACTIVE = {"link", "button", "textbox", "searchbox", "checkbox", "radio", "combobox", "listbox", "option",
               "menuitem", "tab", "switch", "slider", "spinbutton", "menuitemcheckbox", "menuitemradio", "treeitem"}
SKIP = {"generic", "none", "InlineTextBox", "LineBreak", "paragraph", "LayoutTableCell", "LayoutTableRow",
        "LayoutTable", "group", "Section", "list", "listitem", "rowgroup", "form", "main", "banner", "navigation",
        "contentinfo", "region", "strong", "emphasis", "DescriptionList", "radiogroup", "ListMarker",
        "superscript", "subscript", "time", "Abbr", "labelText", "LabelText", "RootWebArea", "document",
        "presentation", "table", "cell", "gridcell", "columnheader", "rowheader", "blockquote"}


class BrowserExecutor:
    toolset = "browser"

    def __init__(self, page, disabled=()):
        self.p = page
        self.pg = page.page
        self.disabled = set(disabled)  # members switched off in configs: answer with an error if named
        self.cdp = page.ctx.new_cdp_session(self.pg)
        self.refs, self.ref_of, self.n = {}, {}, 0
        self.url = self.pg.url

    # refs are scoped to one page load; numbering never restarts
    def _check_nav(self):
        if self.pg.url != self.url:
            self.url = self.pg.url
            self.refs.clear(); self.ref_of.clear()

    def _ref(self, backend):
        if backend not in self.ref_of:
            self.n += 1
            self.ref_of[backend] = f"ref_{self.n}"
            self.refs[f"ref_{self.n}"] = backend
        return self.ref_of[backend]

    def _quad(self, backend):
        try:
            q = self.cdp.send("DOM.getContentQuads", {"backendNodeId": backend})["quads"]
        except Exception:
            return None
        if not q:
            return None
        xs, ys = q[0][0::2], q[0][1::2]
        return min(xs), min(ys), max(xs), max(ys)

    def _visible(self, backend):
        b = self._quad(backend)
        return b is not None and b[2] > 0 and b[3] > 0 and b[0] < W and b[1] < H and b[2] - b[0] > 0

    def tree(self, filt=None, depth=15, ref=None):
        self._check_nav()
        nodes = self.cdp.send("Accessibility.getFullAXTree")["nodes"]
        by = {n["nodeId"]: n for n in nodes}
        root = nodes[0]
        if ref:
            if ref not in self.refs:
                raise ActionError(f"Error: {ref} is stale or not found on the current page. Re-read the page to get fresh references.")
            root = next((n for n in nodes if n.get("backendDOMNodeId") == self.refs[ref]), None)
            if root is None:
                raise ActionError(f"Error: {ref} is stale or not found on the current page. Re-read the page to get fresh references.")
        lines = []

        def text_of(n):
            if n.get("ignored"):
                return " ".join(text_of(by[c]) for c in n.get("childIds", []) if c in by).strip()
            if n["role"]["value"] == "StaticText":
                return n.get("name", {}).get("value", "")
            if n["role"]["value"] in ("image", "img") and n.get("name", {}).get("value"):
                return "[image: " + n["name"]["value"] + "]"  # an icon's label (aria-label or alt)
            return " ".join(t for t in (text_of(by[c]) for c in n.get("childIds", []) if c in by) if t).strip()

        def props(n):
            out = {}
            for pr in n.get("properties", []):
                out[pr["name"]] = pr["value"].get("value")
            return out

        def walk(n, d, indent, parent_name):
            if d > depth:
                return
            role = n.get("role", {}).get("value", "")
            name = (n.get("name", {}) or {}).get("value", "") or ""
            kids = [by[c] for c in n.get("childIds", []) if c in by]
            backend = n.get("backendDOMNodeId")
            if n.get("ignored") or (role in SKIP and role not in ("table",)):
                for k in kids:
                    walk(k, d, indent, parent_name)
                return
            line = None
            if role == "StaticText":
                if name.strip() and name.strip() != parent_name.strip():
                    line = f'text "{name.strip()}"'
            elif role == "row":
                cells = [text_of(k) for k in kids]
                line = "row: " + " | ".join(c for c in cells)
            elif role == "heading":
                line = f'heading "{name}" level={props(n).get("level", "")}'
            else:
                pr = props(n)
                line = f'{role} "{name}"' if name else role
                v = (n.get("value") or {}).get("value")
                if v not in (None, "") and role in ("textbox", "combobox", "searchbox", "spinbutton", "slider"):
                    line += f' value="{v}"'
                for k in ("checked", "pressed", "expanded", "selected", "disabled", "readonly", "required"):
                    if pr.get(k) not in (None, False, "false"):
                        line += f" {k}" if pr[k] in (True, "true") else f" {k}={pr[k]}"
                desc = (n.get("description") or {}).get("value")
                if desc:
                    line += f' description="{desc}"'
                if role == "combobox":  # a <select>: list its options inline, they are never "visible"
                    opts = []

                    def collect(m):
                        r = m.get("role", {}).get("value", "")
                        if r in ("option", "MenuListOption"):
                            opts.append((m.get("name", {}) or {}).get("value", ""))
                        for c in m.get("childIds", []):
                            if c in by:
                                collect(by[c])
                    collect(n)
                    if opts:
                        line += " options: " + " | ".join(opts)
                    kids = []
            if line is not None and backend is not None:
                interactive = role in INTERACTIVE
                show = True
                if filt != "all" and not self._visible(backend):
                    show = False
                if filt == "interactive" and not interactive:
                    show = False
                if show:
                    lines.append("  " * indent + f"{line} [{self._ref(backend)}]")
                nxt = indent + 1 if show else indent
            else:
                nxt = indent
            if role == "row":  # cells already summarised; show only interactive descendants
                for k in kids:
                    walk_interactive(k, d + 1, nxt)
                return
            for k in kids:
                walk(k, d + 1, nxt, name or parent_name)

        def walk_interactive(n, d, indent):
            role = n.get("role", {}).get("value", "")
            if not n.get("ignored") and (role in INTERACTIVE or role in ("img", "image")) and n.get("backendDOMNodeId"):
                name = (n.get("name", {}) or {}).get("value", "")
                if filt == "all" or self._visible(n["backendDOMNodeId"]):
                    lines.append("  " * indent + f'{role} "{name}" [{self._ref(n["backendDOMNodeId"])}]')
                return
            for c in n.get("childIds", []):
                if c in by:
                    walk_interactive(by[c], d + 1, indent)

        walk(root, 1, 0, "")
        sy = self.pg.evaluate("[window.scrollY, document.documentElement.scrollHeight]")
        head = f"Page: {self.pg.title()} ({self.pg.url}). Viewport shows y={sy[0]}..{sy[0] + H} of {sy[1]} px.\n"
        out = head + "\n".join(lines)
        if len(out) > 50000:
            out = out[:50000] + "\n[truncated at 50,000 characters: use a smaller depth or a ref]"
        return out

    def _backend(self, target):
        if target.get("type") != "ref":
            raise ActionError("Expected a ref target")
        self._check_nav()
        r = target.get("ref")
        if r not in self.refs:
            raise ActionError(f"Error: {r} is stale or not found on the current page. Re-read the page to get fresh references.")
        try:
            self.cdp.send("DOM.resolveNode", {"backendNodeId": self.refs[r]})
        except Exception:
            raise ActionError(f"Error: {r} is stale or not found on the current page. Re-read the page to get fresh references.")
        return self.refs[r]

    def _point(self, target):
        if target.get("type") == "coordinate":
            x, y = int(target["x"]), int(target["y"])
            if not (0 <= x < W and 0 <= y < H):
                raise ActionError(f"Coordinates ({x}, {y}) are outside the {W}x{H} viewport.")
            return x, y
        b = self._backend(target)
        try:
            self.cdp.send("DOM.scrollIntoViewIfNeeded", {"backendNodeId": b})
        except Exception:
            pass
        q = self._quad(b)
        if q is None:
            raise ActionError(f"Error: {target['ref']} is not visible on the page, so it can't be clicked.")
        return (q[0] + q[2]) / 2, (q[1] + q[3]) / 2

    def _state(self):
        return {"type": "browser_state", "tabs": [{"tab_id": "tab-1", "title": self.pg.title() or self.pg.url,
                                                   "url": self.pg.url, "active": True}]}

    def run(self, name, inp):
        pg, before = self.pg, self.pg.url
        if name in self.disabled:
            raise ActionError(f"Error: {name} is not available in this browser.")
        if name == "navigate":
            u = inp["url"]
            if u in ("back", "forward", "reload"):
                {"back": pg.go_back, "forward": pg.go_forward, "reload": pg.reload}[u]()
            else:
                if "://" not in u:
                    u = "https://" + u
                pu = urlparse(u)
                if pu.scheme not in ("http", "https") or (pu.hostname, pu.port) != self.p.allowed:
                    raise ActionError("Navigation blocked: only the local intranet at " + self.p.base + " is reachable.")
                pg.goto(u)
            self.p.settle()
            self._check_nav()
            return [{"type": "text", "text": f"Navigated to {pg.url}"}, self._state()]
        if name == "screenshot":
            return [image_block(self.p.png())]
        if name == "zoom":
            return [image_block(self.p.zoom_png(inp["region"]))]
        if name == "read_page":
            return [{"type": "text", "text": self.tree(inp.get("filter"), int(inp.get("depth", 15)), inp.get("ref"))}]
        if name == "find":
            words = [w for w in re.findall(r"\w+", inp["query"].lower()) if len(w) > 1]
            lines = self.tree("all").splitlines()[1:]
            scored = sorted(((sum(w in l.lower() for w in words), i, l.strip()) for i, l in enumerate(lines)), key=lambda t: (-t[0], t[1]))
            hits = [l for s, i, l in scored if s > 0][:20]
            return [{"type": "text", "text": "\n".join(hits) or "No matching elements."}]
        if name == "get_page_text":
            self._check_nav()
            txt = pg.inner_text("main") if pg.query_selector("main") else pg.inner_text("body")
            return [{"type": "text", "text": f"{pg.title()}\n\n{txt[:50000]}"}]
        if name == "form_input":
            b = self._backend(inp["target"])
            obj = self.cdp.send("DOM.resolveNode", {"backendNodeId": b})["object"]["objectId"]
            fn = """function(v){ const el=this;
              if(!el.tagName) return 'Error: the target is text, not a form field';
              if(el.disabled) return 'Error: the element is disabled';
              const t=el.tagName;
              if(t==='SELECT'){ const o=[...el.options].find(o=>o.value==String(v)||o.text.trim()==String(v));
                 if(!o) return 'Error: no option '+v; el.value=o.value; }
              else if(t==='INPUT'&&(el.type==='checkbox'||el.type==='radio')){ el.checked=(v===true||v==='true'||v===1); }
              else if(t==='INPUT'||t==='TEXTAREA'){ if(el.readOnly) return 'Error: the element is read-only'; el.value=String(v); }
              else return 'Error: the element is not a form field (use left_click)';
              el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true})); return 'OK'; }"""
            r = self.cdp.send("Runtime.callFunctionOn", {"objectId": obj, "functionDeclaration": fn,
                                                        "arguments": [{"value": inp["value"]}], "returnByValue": True})
            msg = r["result"].get("value", "OK")
            if msg.startswith("Error"):
                raise ActionError(msg)
            return [{"type": "text", "text": f"Set {inp['target'].get('ref')} to {json.dumps(inp['value'])}."}]
        if name == "scroll_to":
            b = self._backend(inp["target"])
            self.cdp.send("DOM.scrollIntoViewIfNeeded", {"backendNodeId": b})
            return [{"type": "text", "text": f"Scrolled to {inp['target']['ref']}."}]
        if name in ("left_click", "right_click", "middle_click", "double_click", "triple_click", "hover",
                    "mouse_move", "left_mouse_down", "left_mouse_up"):
            x, y = self._point(inp["target"])
            mods = [pw_key(m) for m in (inp.get("modifiers") or "").split("+") if m]
            if name == "hover" or name == "mouse_move":
                pg.mouse.move(x, y)
            elif name == "left_mouse_down":
                pg.mouse.move(x, y); pg.mouse.down()
            elif name == "left_mouse_up":
                pg.mouse.move(x, y); pg.mouse.up()
            else:
                for m in mods:
                    pg.keyboard.down(m)
                pg.mouse.click(x, y, button={"right_click": "right", "middle_click": "middle"}.get(name, "left"),
                               click_count={"double_click": 2, "triple_click": 3}.get(name, 1))
                for m in mods:
                    pg.keyboard.up(m)
            self.p.mouse = (x, y)
        elif name == "left_click_drag":
            x0, y0 = self._point(inp["from"])
            x1, y1 = self._point(inp["target"])
            pg.mouse.move(x0, y0); pg.mouse.down(); pg.mouse.move(x1, y1, steps=8); pg.mouse.up()
        elif name == "scroll":
            x, y = self._point(inp["target"])
            pg.mouse.move(x, y)
            n = int(inp.get("scroll_amount", 3)) * 100
            d = inp["scroll_direction"]
            pg.mouse.wheel(n if d == "right" else -n if d == "left" else 0, n if d == "down" else -n if d == "up" else 0)
        elif name == "type":
            pg.keyboard.type(inp["text"], delay=5)
        elif name == "key":
            for _ in range(int(inp.get("repeat", 1))):
                pg.keyboard.press(pw_key(inp["text"]))
        elif name == "hold_key":
            k = pw_key(inp["text"])
            pg.keyboard.down(k); time.sleep(min(float(inp["duration"]), 5)); pg.keyboard.up(k)
        elif name == "wait":
            time.sleep(min(float(inp["duration"]), 5))
        else:
            raise ActionError(f"Unknown or disabled member {name}")
        changed = self.p.settle(before)
        what = inp.get("target", {}).get("ref") if isinstance(inp.get("target"), dict) else None
        ack = {"left_click": "Clicked", "hover": "Hovered over", "double_click": "Double-clicked"}.get(name)
        text = f"{ack} element {what}." if ack and what else "OK"
        out = [{"type": "text", "text": text}]
        if changed:
            self._check_nav()
            out.append(self._state())
        return out


# ---- the loop ---------------------------------------------------------------------------------------------

def _strip_images(blocks):
    out = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "image":
            out.append({"type": "image", "bytes": len(b["source"]["data"]) * 3 // 4})
        elif isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            out.append({**b, "content": _strip_images(b["content"])})
        else:
            out.append(b)
    return out


def run_browser_agent(task, executor, tool_entry, system, *, part, model="claude-haiku-5-5", effort="medium",
                      max_steps=30, max_cost=0.05, max_tokens=8000, transcript=None, client=None, shots_dir=None):
    """Part 1's loop for a client toolset. Returns a harness Result with per-step records and counters."""
    request = dict(model=model, max_tokens=max_tokens, tools=[tool_entry],
                   system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   output_config={"effort": effort}, thinking={"type": "adaptive", "display": "summarized"},
                   cache_control={"type": "ephemeral"})
    cl = _TimedClient(client or ledger.client())
    log = Transcript(transcript)
    log.write(event="task", task=task, **request)
    messages = [{"role": "user", "content": task}]
    result = Result(answer="", stop="max_steps")
    result.images_per_step, result.actions, result.errors = [], [], 0
    n_images = 0
    for n in range(1, max_steps + 1):
        if result.cost >= max_cost:
            result.stop = "max_cost"
            break
        result.images_per_step.append(n_images)  # images carried in this request
        resp = ledger.create(cl, part, note=f"a09 step {n}", messages=messages, **request)
        u, calls = resp.usage, [b for b in resp.content if b.type == "tool_use"]
        step = Step(n, u.input_tokens, u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0,
                    u.output_tokens, round(cl.last, 3), ledger.cost_of(model, u), resp.stop_reason,
                    [f"{b.name}({json.dumps(b.input)})" for b in calls])
        result.steps.append(step)
        log.write(event="step", **step.__dict__, content=[b.model_dump(exclude_none=True) for b in resp.content])
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            result.stop = resp.stop_reason
            result.answer = "".join(b.text for b in resp.content if b.type == "text")
            break
        results, failed = [], False
        for b in calls:
            ts = getattr(b, "toolset_name", None) or executor.toolset
            r = {"type": "tool_result", "tool_use_id": b.id, "toolset_name": ts}
            if failed:
                r.update(is_error=True, content=HALT[executor.toolset])
            else:
                try:
                    r["content"] = executor.run(b.name, dict(b.input))
                    result.actions.append(b.name)
                except Exception as e:  # the model sees the error; later blocks in this turn are halted
                    msg = str(e) if isinstance(e, ActionError) else f"Error: {type(e).__name__}: {str(e)[:300]}"
                    r.update(is_error=True, content=msg if msg.startswith("Error") or msg.startswith("Not") else "Error: " + msg)
                    result.errors += 1
                    result.actions.append(b.name + "!")
                    failed = True
            if isinstance(r.get("content"), list):
                imgs = [c for c in r["content"] if c.get("type") == "image"]
                n_images += len(imgs)
                if shots_dir and imgs:
                    Path(shots_dir).mkdir(parents=True, exist_ok=True)
                    for k, im in enumerate(imgs):
                        Path(shots_dir, f"step{n:02d}-{k}.png").write_bytes(base64.b64decode(im["source"]["data"]))
            results.append(r)
        messages.append({"role": "user", "content": results})
        log.write(event="tool_results", n=n, results=_strip_images(results))
    result.messages_len = len(messages)
    log.write(event="result", answer=result.answer, stop=result.stop, steps=len(result.steps), cost=result.cost,
              actions=result.actions, errors=result.errors, images=n_images)
    result.n_images = n_images
    return result

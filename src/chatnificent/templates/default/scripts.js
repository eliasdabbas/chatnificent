// Defaults for the chatnificent namespace. Devs may assign properties
// (e.g. window.chatnificent.renderMarkdown, .beforeSend) BEFORE this
// script runs; the `||` guards preserve any pre-set override.
window.chatnificent = window.chatnificent || {};
window.chatnificent.beforeSend = window.chatnificent.beforeSend || function (message) { return message; };
window.chatnificent.afterSend = window.chatnificent.afterSend || function (convoId) { };
window.chatnificent.onDelta = window.chatnificent.onDelta || function (token, accumulated) { };
window.chatnificent.messageSlots = window.chatnificent.messageSlots
    || { "pre-user": "", "post-user": "", "pre-assistant": "", "post-assistant": "" };
function chatInteraction(el, data) {
    var apiBase = window.__CHATNIFICENT_ROOT__ || "";
    fetch(apiBase + "/api/interactions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: el.id, data: data !== undefined ? data : el.value })
    }).catch(function (e) { console.warn("interaction error", e); });
}
(function () {
    "use strict";
    var $ = function (s) { return document.querySelector(s) };
    var msgs = $("#messages"), input = $("#input"), sendBtn = $("#send");
    var sidebar = $("#sidebar"), convoList = $("#convo-list");
    var chatWrap = $("#chat-wrap"), welcome = $("#welcome");
    var convoId = null, sidebarOpen = false;
    var apiBase = window.__CHATNIFICENT_ROOT__ || "";

    // Single source of truth for RTL detection — textarea, addMsg, stream.
    var RTL_RE = /[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb1d-\ufb4f\ufb50-\ufdff\ufe70-\ufeff]/;
    function isRtl(text) { return !!text && RTL_RE.test(text); }

    // breaks: single newlines render as <br> so malformed/streaming output
    // doesn't collapse into a wall of text. gfm: tables, autolinks, strikethrough.
    // silent: marked logs parse errors instead of throwing — assistant
    // output can be malformed mid-stream; we don't want a thrown error
    // to break rendering.
    marked.setOptions({ breaks: true, gfm: true, silent: true });
    DOMPurify.addHook("afterSanitizeAttributes", function (node) {
        if (node.tagName === "A") {
            node.setAttribute("target", "_blank");
            node.setAttribute("rel", "noopener");
        }
    });

    // Markdown rendering is exposed as an overridable seam. Devs can
    // swap implementations (e.g. markdown-it, add highlight.js) by
    // assigning window.chatnificent.renderMarkdown BEFORE this IIFE
    // runs — the `||` guard preserves an earlier override.
    window.chatnificent = window.chatnificent || {};
    window.chatnificent.renderMarkdown = window.chatnificent.renderMarkdown
        || function (text) {
            try { return DOMPurify.sanitize(marked.parse(text)); }
            catch (e) { return DOMPurify.sanitize(text); }
        };
    function renderMarkdown(text) { return window.chatnificent.renderMarkdown(text); }

    function setTheme(dark) {
        document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
        try { localStorage.setItem("chatnificent-theme", dark ? "dark" : "light") } catch (e) { }
    }
    (function () {
        var saved;
        try { saved = localStorage.getItem("chatnificent-theme") } catch (e) { }
        if (saved === "dark") setTheme(true);
        else if (saved === "light") setTheme(false);
        else if (window.matchMedia && window.matchMedia("(prefers-color-scheme:dark)").matches) setTheme(true);
    })();
    $("#theme-toggle").addEventListener("click", function () {
        setTheme(document.documentElement.getAttribute("data-theme") !== "dark");
        input.focus();
    });

    function toggleSidebar() {
        sidebarOpen = !sidebarOpen;
        sidebar.classList.toggle("open", sidebarOpen);
    }
    $("#sidebar-toggle").addEventListener("click", toggleSidebar);

    // Close sidebar on outside click only when the sidebar visually
    // overlaps the centered chat column.
    function sidebarOverlapsContent() {
        var cs = getComputedStyle(document.documentElement);
        var sw = parseFloat(cs.getPropertyValue("--sidebar-w")) || 300;
        var cw = parseFloat(cs.getPropertyValue("--chat-max-w")) || 760;
        return window.innerWidth < (2 * sw + cw);
    }
    document.addEventListener("click", function (e) {
        if (!sidebarOpen) return;
        if (!sidebarOverlapsContent()) return;
        if (sidebar.contains(e.target)) return;
        if (e.target.closest("#sidebar-toggle")) return;
        toggleSidebar();
    });

    $("#new-chat-btn").addEventListener("click", function () {
        convoId = null; msgs.innerHTML = "";
        msgs.appendChild(welcome); welcome.style.display = "";
        history.pushState({ convoId: null }, "", apiBase + "/");
        refreshConvoList();
        input.focus();
    });

    input.addEventListener("input", function () {
        this.style.height = "auto";

        this.style.height = this.scrollHeight + "px";
        this.dir = isRtl(this.value) ? "rtl" : "ltr";
    });

    // SVG icons for default message actions.
    var ICON_COPY = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    var ICON_CHECK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    var ICON_PENCIL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>';
    var ICON_CHEVRON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';

    // Inject the language label + copy button at parse time, so fenced
    // code blocks come out fully decorated on the first render — no
    // post-walk of the DOM needed, and decorations appear during
    // streaming (not just on `done`). Defers to marked's default
    // ``code`` renderer for the actual <pre><code> escaping.
    marked.use({
        renderer: {
            code: function (token) {
                var html = marked.Renderer.prototype.code.call(this, token);
                var m = (token.lang || "").match(/^[\w.+-]+/);
                if (!m) return html;
                var header = '<div class="code-header">' +
                    '<span class="code-lang">' + m[0] + '</span>' +
                    '<button class="msg-action-btn code-copy" type="button" ' +
                    'data-action="copy-code" data-tooltip="Copy" aria-label="Copy code">' +
                    ICON_COPY + '</button></div>';
                return html.replace(/^<pre>/, '<pre class="has-lang">' + header);
            }
        }
    });

    function defaultActionsHTML(role) {
        if (role === "user") {
            return '<button class="msg-action-btn" data-action="copy" data-tooltip="Copy" aria-label="Copy">' + ICON_COPY + '</button>' +
                '<button class="msg-action-btn" data-action="edit" data-tooltip="Edit" aria-label="Edit prompt">' + ICON_PENCIL + '</button>';
        }
        return '<button class="msg-action-btn" data-action="copy" data-tooltip="Copy" aria-label="Copy as markdown">' + ICON_COPY + '</button>';
    }

    function applyCollapse(bubble) {
        if (!bubble.classList.contains("user")) return;
        // Bubble has no max-height yet, so scrollHeight === clientHeight
        // until we collapse it — compare against the token directly.
        var raw = getComputedStyle(document.documentElement)
            .getPropertyValue("--user-pill-max-h").trim();
        var maxH = parseFloat(raw);
        if (!maxH || isNaN(maxH)) return;
        if (bubble.scrollHeight <= maxH) return;
        if (bubble.classList.contains("is-collapsed")) return;
        // Wrap the bubble so the chevron centers on the pill, not the
        // full-width row. The wrapper shrinks to the bubble's width.
        var wrap = document.createElement("div");
        wrap.className = "msg-pill-wrap is-collapsed-wrap";
        bubble.parentNode.insertBefore(wrap, bubble);
        wrap.appendChild(bubble);
        bubble.classList.add("is-collapsed");
        var toggle = document.createElement("button");
        toggle.className = "msg-collapse-toggle";
        toggle.type = "button";
        toggle.setAttribute("aria-label", "Show more");
        toggle.setAttribute("aria-expanded", "false");
        toggle.innerHTML = ICON_CHEVRON;
        toggle.addEventListener("click", function () {
            // On collapse, anchor the toggle's viewport position so
            // the page doesn't yank to the bottom. On expand, leave
            // scroll alone so the user keeps reading where they are.
            var willCollapse = !bubble.classList.contains("is-collapsed");
            var beforeTop = willCollapse ? toggle.getBoundingClientRect().top : 0;
            var collapsed = bubble.classList.toggle("is-collapsed");
            wrap.classList.toggle("is-collapsed-wrap", collapsed);
            toggle.classList.toggle("is-expanded", !collapsed);
            toggle.setAttribute("aria-expanded", String(!collapsed));
            toggle.setAttribute("aria-label", collapsed ? "Show more" : "Show less");
            if (willCollapse) {
                var afterTop = toggle.getBoundingClientRect().top;
                msgs.scrollTop += afterTop - beforeTop;
            }
        });
        wrap.appendChild(toggle);
    }

    function addMsg(role, text) {
        if (welcome.style.display !== "none") welcome.style.display = "none";
        var slots = (window.chatnificent && window.chatnificent.messageSlots) || {};
        var row = document.createElement("div");
        row.className = "msg-row msg-row-" + role;
        var pre = document.createElement("div");
        pre.className = "msg-pre";
        pre.innerHTML = slots["pre-" + role] || "";
        var d = document.createElement("div");
        d.className = "msg " + role + (role === "assistant" ? " md-content" : "");
        if (role === "assistant") {
            d.innerHTML = renderMarkdown(text);
            d.dataset.source = text || "";
        } else {
            d.textContent = text;
        }
        if (isRtl(text)) d.dir = "rtl";
        var post = document.createElement("div");
        post.className = "msg-post";
        post.innerHTML = defaultActionsHTML(role) + (slots["post-" + role] || "");
        row.appendChild(pre);
        row.appendChild(d);
        row.appendChild(post);
        msgs.appendChild(row);
        applyCollapse(d);
        msgs.scrollTop = msgs.scrollHeight;
        return d;
    }
    function renderMessageList(messages) {
        msgs.innerHTML = "";
        if (!messages || !messages.length) {
            msgs.appendChild(welcome);
            welcome.style.display = "";
            return;
        }
        welcome.style.display = "none";
        messages.forEach(function (m) {
            addMsg(m.role, m.content || "");
        });
    }
    function showLoading() {
        var el = document.createElement("div");
        el.id = "loading"; el.className = "visible";
        el.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
        msgs.appendChild(el); msgs.scrollTop = msgs.scrollHeight;
    }
    function hideLoading() { var el = $("#loading"); if (el) el.remove(); }
    function scrollToBottom() {
        if (msgs.scrollHeight - msgs.scrollTop - msgs.clientHeight < 100) {
            msgs.scrollTop = msgs.scrollHeight;
        }
    }

    async function send() {
        var text = input.value.trim();
        if (!text) return;
        text = window.chatnificent.beforeSend(text);
        if (text === null || text === undefined) return;
        addMsg("user", text); input.value = ""; input.style.height = "auto";
        sendBtn.disabled = true; showLoading();
        try {
            var r = await fetch(apiBase + "/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream"
                },
                body: JSON.stringify({ message: text, conversation_id: convoId })
            });
            var ct = (r.headers.get("Content-Type") || "");
            if (ct.indexOf("text/event-stream") !== -1) {
                var bubble = null;
                var accumulated = "";
                var bubbleRtl = false;
                var reader = r.body.getReader();
                var decoder = new TextDecoder();
                var buf = "";
                while (true) {
                    var result = await reader.read();
                    if (result.done) break;
                    buf += decoder.decode(result.value, { stream: true });
                    var lines = buf.split("\n");
                    buf = lines.pop();
                    for (var i = 0; i < lines.length; i++) {
                        var line = lines[i];
                        if (line.indexOf("data: ") !== 0) continue;
                        var payload;
                        try { payload = JSON.parse(line.slice(6)); } catch (_) { continue; }
                        if (payload.event === "delta") {
                            if (!bubble) { hideLoading(); bubble = addMsg("assistant", ""); }
                            accumulated += payload.data;
                            bubble.innerHTML = renderMarkdown(accumulated);
                            bubble.dataset.source = accumulated;
                            if (!bubbleRtl && isRtl(payload.data)) { bubbleRtl = true; bubble.dir = "rtl"; }
                            scrollToBottom();
                            window.chatnificent.onDelta(payload.data, accumulated);
                        } else if (payload.event === "status") {
                            if (!bubble) { hideLoading(); bubble = addMsg("assistant", ""); }
                            bubble.innerHTML = renderMarkdown(accumulated || "") +
                                '<div class="status-indicator">' + DOMPurify.sanitize(payload.data) + '</div>';
                            scrollToBottom();
                        } else if (payload.event === "done") {
                            if (!bubble) { hideLoading(); bubble = addMsg("assistant", ""); }
                            if (payload.data && payload.data.conversation_id) {
                                convoId = payload.data.conversation_id;
                                window.chatnificent.afterSend(convoId);
                                history.replaceState({ convoId: convoId }, "", payload.data.path || apiBase + "/" + convoId);
                                upsertSidebarEntry(convoId, text.length > 30 ? text.slice(0, 30) + "…" : text);
                            }
                            bubble.innerHTML = renderMarkdown(accumulated);
                            bubble.dataset.source = accumulated;
                        } else if (payload.event === "error") {
                            if (!bubble) { hideLoading(); bubble = addMsg("assistant", ""); }
                            bubble.innerHTML = renderMarkdown(accumulated || "Error: " + payload.data);
                            bubble.dataset.source = accumulated || ("Error: " + payload.data);
                        }
                    }
                }
            } else {
                var data = await r.json(); hideLoading();
                if (data.error) addMsg("assistant", "Error: " + data.error);
                else {
                    renderMessageList(data.messages || []);
                    convoId = data.conversation_id;
                    window.chatnificent.afterSend(convoId);
                    history.replaceState({ convoId: convoId }, "", data.path || apiBase + "/" + convoId);
                    upsertSidebarEntry(convoId, text.length > 30 ? text.slice(0, 30) + "…" : text);
                }
            }
        } catch (e) { hideLoading(); addMsg("assistant", "Error: " + e.message); }
        sendBtn.disabled = false; input.focus();
    }
    sendBtn.addEventListener("click", send);
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });

    async function refreshConvoList() {
        try {
            var r = await fetch(apiBase + "/api/conversations"); var data = await r.json();
            convoList.innerHTML = "";
            (data.conversations || []).forEach(function (c) {
                var id = c.id || c, title = c.title || id;
                var el = document.createElement("div");
                el.className = "convo-item" + (id === convoId ? " active" : "");
                el.dataset.convoId = id;
                el.textContent = title;
                el.addEventListener("click", function () { loadConvo(id) });
                convoList.appendChild(el);
            });
        } catch (e) { }
    }
    function upsertSidebarEntry(id, title) {
        var items = convoList.querySelectorAll(".convo-item");
        var existing = null;
        for (var i = 0; i < items.length; i++) {
            items[i].classList.remove("active");
            if (items[i].dataset.convoId === id) existing = items[i];
        }
        if (existing) {
            existing.classList.add("active");
        } else {
            var el = document.createElement("div");
            el.className = "convo-item active";
            el.dataset.convoId = id;
            el.textContent = title || id;
            el.addEventListener("click", function () { loadConvo(id) });
            convoList.prepend(el);
        }
    }
    async function loadConvo(id, replaceHistory) {
        try {
            var r = await fetch(apiBase + "/api/conversations/" + id); var data = await r.json();
            if (data.error) return;
            convoId = id;
            if (replaceHistory) history.replaceState({ convoId: id }, "", data.path || apiBase + "/" + id);
            else history.pushState({ convoId: id }, "", data.path || apiBase + "/" + id);
            renderMessageList(data.messages || []);
            refreshConvoList();
            input.focus();
        } catch (e) { }
    }
    window.addEventListener("popstate", function (e) {
        var cid = e.state && e.state.convoId;
        if (cid) loadConvo(cid); else { convoId = null; msgs.innerHTML = ""; msgs.appendChild(welcome); welcome.style.display = ""; refreshConvoList(); input.focus(); }
    });
    (function initFromURL() {
        var initConvo = window.__CHATNIFICENT_CONVO__;
        if (initConvo) loadConvo(initConvo); else refreshConvoList();
    })();

    // Suggestion chips contract: any element with [data-insert-prompt="..."]
    // fills the textarea and focuses it. Delegated so HTML rendered later
    // (e.g. via marked into #welcome-message) still works without rebinding.
    document.addEventListener("click", function (e) {
        var target = e.target.closest("[data-insert-prompt]");
        if (!target) return;
        var prompt = target.getAttribute("data-insert-prompt");
        if (prompt === null) return;
        input.value = prompt;
        input.dispatchEvent(new Event("input"));
        input.focus();
    });

    // Shared copy-feedback: icon flip + tooltip swap for ~1.4s, with
    // a re-click guard so the feedback window can't be interrupted.
    function copyWithFeedback(btn, text) {
        if (btn.classList.contains("is-copied")) return;
        var prevTip = btn.getAttribute("data-tooltip");
        var done = function () {
            btn.classList.add("is-copied");
            btn.setAttribute("data-tooltip", "Copied");
            var icon = btn.querySelector("svg");
            if (icon) icon.outerHTML = ICON_CHECK;
            setTimeout(function () {
                btn.classList.remove("is-copied");
                btn.setAttribute("data-tooltip", prevTip || "Copy");
                var newIcon = btn.querySelector("svg");
                if (newIcon) newIcon.outerHTML = ICON_COPY;
            }, 1400);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(done, function () { });
        } else {
            var ta = document.createElement("textarea");
            ta.value = text; document.body.appendChild(ta);
            ta.select();
            try { document.execCommand("copy"); done(); } catch (_) { }
            document.body.removeChild(ta);
        }
    }

    // Default Copy action on message rows. Assistant copies raw markdown
    // source (stashed on .msg.assistant[data-source]); user copies plain text.
    document.addEventListener("click", function (e) {
        var btn = e.target.closest('.msg-action-btn[data-action="copy"]');
        if (!btn) return;
        var row = btn.closest(".msg-row");
        var bubble = row && row.querySelector(".msg");
        if (!bubble) return;
        var text = bubble.classList.contains("assistant")
            ? (bubble.dataset.source || bubble.textContent)
            : bubble.textContent;
        copyWithFeedback(btn, text);
    });

    // Per-code-block Copy on fenced blocks that carry a language tag.
    document.addEventListener("click", function (e) {
        var btn = e.target.closest('.msg-action-btn[data-action="copy-code"]');
        if (!btn) return;
        var pre = btn.closest("pre");
        var code = pre && pre.querySelector("code");
        if (!code) return;
        copyWithFeedback(btn, code.textContent);
    });

    // Edit button on user pills reuses the [data-insert-prompt] contract.
    document.addEventListener("click", function (e) {
        var btn = e.target.closest('.msg-action-btn[data-action="edit"]');
        if (!btn) return;
        var row = btn.closest(".msg-row");
        var bubble = row && row.querySelector(".msg.user");
        if (!bubble) return;
        input.value = bubble.textContent;
        input.dispatchEvent(new Event("input"));
        input.focus();
    });

    input.focus();
})();

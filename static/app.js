(function () {
  const $ = (id) => document.getElementById(id);
  const levelOf = (category) => (category === "safe" ? "safe" : category === "toxic" ? "toxic" : "mild");
  const pct = (x) => Math.round(x * 100) + "%";

  /* ---------- tabs ---------- */
  const tabs = [$("tab-check"), $("tab-batch")];
  function showTab(active) {
    tabs.forEach((t) => {
      const on = t === active;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !on;
    });
  }
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => showTab(t));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const next = tabs[(i + 1) % tabs.length];
        showTab(next);
        next.focus();
      }
    });
  });

  /* ---------- single comment ---------- */
  const comment = $("comment");
  const checkBtn = $("check-btn");

  document.querySelectorAll("[data-example]").forEach((b) =>
    b.addEventListener("click", () => {
      comment.value = b.dataset.example;
      comment.focus();
    })
  );

  function showError(el, msg) {
    el.textContent = msg;
    el.hidden = !msg;
  }

  async function check() {
    showError($("check-error"), "");
    const text = comment.value.trim();
    if (!text) {
      showError($("check-error"), "Enter a comment to check.");
      comment.focus();
      return;
    }
    checkBtn.disabled = true;
    checkBtn.textContent = "Checking";
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong. Try again.");
      render(data);
    } catch (err) {
      showError($("check-error"), err.message);
    } finally {
      checkBtn.disabled = false;
      checkBtn.textContent = "Check comment";
    }
  }

  function render(r) {
    const level = levelOf(r.category);

    // marked-up comment
    const marked = $("marked");
    marked.textContent = "";
    marked.dataset.level = level;
    marked.classList.remove("drawn");
    let n = 0;
    r.highlighted.forEach(([word, hit]) => {
      if (hit) {
        const s = document.createElement("span");
        s.className = "mark";
        s.style.setProperty("--i", n++);
        s.textContent = word;
        marked.appendChild(s);
      } else {
        marked.appendChild(document.createTextNode(word));
      }
    });
    $("mark-note").textContent = r.keywords.length
      ? "Marked words pushed the comment toward this verdict."
      : level === "safe"
      ? "Nothing in this comment was marked."
      : "No single word stood out, so the whole tone was weighed.";

    $("editor").hidden = true;
    $("result-text").hidden = false;

    // verdict
    const v = $("verdict");
    v.dataset.level = level;
    $("verdict-empty").hidden = true;
    $("verdict-body").hidden = false;
    $("verdict-word").textContent = r.category.charAt(0).toUpperCase() + r.category.slice(1);
    $("verdict-prob").textContent = pct(r.toxicity_probability) + " chance this comment is abusive";
    $("score-num").textContent = Math.round(r.severity_score) + " out of 100";
    $("why").textContent = r.explanation;
    $("action").textContent = r.action;

    const probs = $("probs");
    probs.textContent = "";
    Object.entries(r.probabilities).forEach(([label, p]) => {
      const row = document.createElement("div");
      row.className = "prob";
      const dt = document.createElement("dt");
      dt.textContent = label.charAt(0).toUpperCase() + label.slice(1);
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("i");
      fill.style.width = pct(p);
      bar.appendChild(fill);
      const dd = document.createElement("dd");
      dd.textContent = pct(p);
      row.append(dt, bar, dd);
      probs.appendChild(row);
    });

    // animate once the new state has painted
    const pin = $("pin");
    pin.style.left = "0%";
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        marked.classList.add("drawn");
        pin.style.left = Math.min(Math.max(r.severity_score, 0), 100) + "%";
      })
    );
  }

  checkBtn.addEventListener("click", check);
  comment.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) check();
  });
  $("edit-btn").addEventListener("click", () => {
    $("result-text").hidden = true;
    $("editor").hidden = false;
    comment.focus();
  });

  /* ---------- batch ---------- */
  let rows = [];

  $("file").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("file-name").textContent = file.name;
    showError($("batch-error"), "");
    $("batch-out").hidden = true;
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/api/batch", { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong. Try again.");
      rows = data.rows.sort((a, b) => b.severity_score - a.severity_score);
      $("filter").value = "all";
      drawRows();
      $("batch-out").hidden = false;
    } catch (err) {
      showError($("batch-error"), err.message);
    }
  });

  function drawRows() {
    const f = $("filter").value;
    const shown = rows.filter((r) => f === "all" || r.category === f);
    const count = (c) => rows.filter((r) => r.category === c).length;
    $("batch-summary").textContent =
      rows.length + " comments checked: " + count("toxic") + " toxic, " + count("mildly rude") + " mildly rude, " + count("safe") + " safe.";

    const body = $("rows");
    body.textContent = "";
    shown.forEach((r) => {
      const tr = document.createElement("tr");
      const c1 = document.createElement("td");
      c1.className = "comment";
      c1.textContent = r.text;
      const c2 = document.createElement("td");
      c2.className = "verdict-cell";
      c2.dataset.level = levelOf(r.category);
      c2.textContent = r.category.charAt(0).toUpperCase() + r.category.slice(1);
      const c3 = document.createElement("td");
      c3.className = "sev";
      c3.textContent = Math.round(r.severity_score);
      const bar = document.createElement("i");
      bar.style.width = Math.round(r.severity_score * 0.6) + "px";
      c3.appendChild(bar);
      const c4 = document.createElement("td");
      c4.className = "words";
      c4.textContent = r.keywords.join(", ") || "None";
      tr.append(c1, c2, c3, c4);
      body.appendChild(tr);
    });
  }
  $("filter").addEventListener("change", drawRows);

  $("download").addEventListener("click", () => {
    const esc = (v) => '"' + String(v).replace(/"/g, '""') + '"';
    const lines = [["text", "verdict", "severity", "flagged_words", "suggested_action"].join(",")];
    rows.forEach((r) =>
      lines.push([esc(r.text), esc(r.category), r.severity_score, esc(r.keywords.join("; ")), esc(r.action)].join(","))
    );
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "moderation-results.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  });
})();

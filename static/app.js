const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");
const preview = document.getElementById("preview");
const hint = document.getElementById("hint");
const statusEl = document.getElementById("status");
const results = document.getElementById("results");

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handle(fileInput.files[0]);
});

["dragenter", "dragover"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e =>
  drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", ev => {
  const f = ev.dataTransfer.files[0];
  if (f) handle(f);
});

async function handle(file) {
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  hint.hidden = true;
  statusEl.textContent = "Ищу похожие вещи…";
  results.innerHTML = "";

  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/search?top=12", { method: "POST", body: form });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    render(data.results);
    statusEl.textContent = `Найдено: ${data.results.length}`;
  } catch (err) {
    statusEl.textContent = "Ошибка: " + err.message;
  }
}

function render(items) {
  for (const r of items) {
    const card = document.createElement("div");
    card.className = "card";
    const title = r.title || r.item_id;
    const price = r.price ? `<div>${r.price}</div>` : "";
    const link = r.url ? `<a href="${r.url}" target="_blank" rel="noopener">В магазин</a>` : "";
    card.innerHTML = `
      <img src="${r.image}" alt="${title}" loading="lazy">
      <div class="info">
        <div class="title">${title}</div>
        <div class="score">сходство ${(r.score * 100).toFixed(1)}%</div>
        ${price}${link}
      </div>`;
    results.appendChild(card);
  }
}

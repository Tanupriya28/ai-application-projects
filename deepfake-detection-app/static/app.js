function showTab(btn, id) {
  const parent = btn.closest(".result-card");

  parent
    .querySelectorAll(".tab-content")
    .forEach((t) => t.classList.remove("active"));
  parent
    .querySelectorAll(".tabs button")
    .forEach((b) => b.classList.remove("active"));

  parent.querySelector("#" + id).classList.add("active");
  btn.classList.add("active");
}
const form = document.querySelector("form");
const loader = document.getElementById("loader");

if (form) {
  form.addEventListener("submit", () => {
    loader.classList.remove("hidden");
  });
}
function updateCount(input) {
  const label = document.getElementById("fileLabel");
  if (input.files.length === 1) {
    label.innerText = "1 image selected";
  } else {
    label.innerText = input.files.length + " images selected";
  }
}

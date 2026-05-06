(function () {
  var year = document.getElementById("shellYear");
  if (year) {
    year.textContent = new Date().getFullYear();
  }

  var btn = document.querySelector(".art-info-bar-btn");
  var bar = document.querySelector(".art-info-bar");
  var curtain = document.getElementById("shellMobileCurtain");
  if (!btn || !bar) return;

  function setOpen(open) {
    bar.classList.toggle("art-active", open);
    btn.classList.toggle("art-active", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    if (curtain) {
      curtain.classList.toggle("is-open", open);
      curtain.setAttribute("aria-hidden", open ? "false" : "true");
    }
  }

  btn.addEventListener("click", function () {
    setOpen(!bar.classList.contains("art-active"));
  });

  if (curtain) {
    curtain.addEventListener("click", function () {
      setOpen(false);
    });
  }
})();

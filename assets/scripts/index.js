/* eslint-env es6 */
/* eslint-disable no-console */

const successColor = "#28a745";
const copyHTML =
  '<svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-copy js-clipboard-copy-icon m-2"><path fill="#fff" d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z"></path><path fill="#fff" d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z"></path></svg>';
const copiedHTML = `<svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none m-2"><path fill="${successColor}" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"></path></svg>`;

const includeFileCodeBlocks = document.querySelectorAll("[include-file");
includeFileCodeBlocks.forEach(async (codeBlock) => {
  const fileName = codeBlock.getAttribute("include-file");
  console.log(fileName);
  const fileContent = await fetch(fileName).then((response) => response.text());
  codeBlock.innerText = fileContent;
});

const spoilerCodeBlocks = document.querySelectorAll('[spoiler="true"]');
spoilerCodeBlocks.forEach(async (codeBlock) => {
  const old_html = codeBlock.innerHTML;
  const new_html =
    '<details><summary class="code-summary">Click to show/hide code</summary>' +
    old_html +
    "</details>";
  codeBlock.innerHTML = new_html;
});

const copyableCodeBlocks = document.querySelectorAll("pre.highlight");
copyableCodeBlocks.forEach((codeBlock) => {
  var copyButton = document.createElement("button");
  copyButton.className = "copy";
  copyButton.type = "button";
  copyButton.ariaLabel = "Copy code to clipboard";
  copyButton.innerHTML = copyHTML;

  codeBlock.append(copyButton);

  copyButton.addEventListener("click", function () {
    var code = codeBlock.querySelector("code").innerText.trim();
    window.navigator.clipboard.writeText(code);

    copyButton.innerHTML = copiedHTML;
    copyButton.style.borderColor = successColor;
    var twoSeconds = 2000;

    setTimeout(function () {
      copyButton.innerHTML = copyHTML;
      copyButton.style.borderColor = "#fff";
    }, twoSeconds);
  });
});

const copyToClipboardButtonStrings = {
    default: 'Copy',
    ariaLabel: 'Copy to clipboard',
    copied: 'Copied',
};

const includeFileCodeBlocks = document.querySelectorAll('[include-file');
includeFileCodeBlocks.forEach(async (codeBlock) => {
    const fileName = codeBlock.getAttribute('include-file');
    console.log(fileName);
    const fileContent = await fetch(fileName).then(response => response.text());
    codeBlock.innerText = fileContent;

});

const spoilerCodeBlocks = document.querySelectorAll('[spoiler="true"]');
spoilerCodeBlocks.forEach(async (codeBlock) => {
   const old_html = codeBlock.innerHTML;
    const new_html = '<details><summary class="code-summary">Click to show/hide code</summary>' + old_html + '</details>';
    codeBlock.innerHTML = new_html;

});

const copyableCodeBlocks = document.querySelectorAll('pre.highlight');
copyableCodeBlocks.forEach((codeBlock) => {
    var copyButton = document.createElement('button');
    copyButton.className = 'copy';
    copyButton.type = 'button';
    copyButton.ariaLabel = 'Copy code to clipboard';
    copyButton.innerText = 'Copy';

    codeBlock.append(copyButton);

    copyButton.addEventListener('click', function () {
        var code = codeBlock.querySelector('code').innerText.trim();
        window.navigator.clipboard.writeText(code);

        copyButton.innerText = 'Copied';
        var fourSeconds = 4000;

        setTimeout(function () {
            copyButton.innerText = 'Copy';
        }, fourSeconds);
    });
});

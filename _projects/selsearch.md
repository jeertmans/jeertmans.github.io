---
title: SelSearch
github: jeertmans/selsearch
date: 2022-02-21
---

Grab text selected in any application and open a web browser to search for.

<!--more-->

# Story

As part of my research, I often need to search authors, paper names, or even
translate text from a document (e.g., PDF) that I'm reading.

Because I'm kind of lazy, I decided to see if I could automate a bit this
process. This is how I created
[SelSearch](https://github.com/jeertmans/selsearch).

SelSearch simply reads any text you select, in any application[^1], and allow
to search for it on your favorite websites.

I think one image is worth a thousands words, so here is an example:

![Example usage of SelSearch](https://raw.githubusercontent.com/jeertmans/selsearch/main/static/demo.gif)

Above, you can see me select some word, i.e., *Linux*, in the terminal and
search it on Google via a predefined shortcut.
Then, I select some sentence on Google, and look up for the
translation on DeepL.

As explained on the [Github page](https://github.com/jeertmans/selsearch), SelSearch
is fully configurable using a TOML config file:

```toml
[urls]
# List of urls
# You can add / remove / edit any number of lines
google = "https://www.google.com/search?q="
wordreference = "https://www.wordreference.com/enfr/"
deepl = "https://www.deepl.com/translator#en/fr/"
googlescholar = "https://scholar.google.com/scholar?q="

[shortcuts]
# List of shotcuts
# You can add / remove / edit any number of lines
"<ctrl>+0" = "google"
"<ctrl>+1" = "deepl"
"<ctrl>+2" = "wordreference"
```

[^1]: Depending on whether you have XSel installed, some application may not work. But for most, like any web page or PDF reader, it works fine!

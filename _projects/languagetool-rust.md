---
title: LanguageTool-Rust
order: 2
github: jeertmans/languagetool-rust
date: 2023-03-25
---

LanguageTool API in Rust.

<!--more-->

In 2022, I developed a strong *love* towards the Rust programming language.
As I was looking for a first Rust project, and I realized that [LanguageTool](https://languagetool.org/)[^1]
was lacking API bindings[^2] in RustAPI, I decided to implement my own API bindings, [LanguageTool-Rust](https://github.com/jeertmans/languagetool-rust).

This project provides bindings for the whole public LanguageTool HTTP public, and is continously tested against different versions of LanguageTool.

[^1]: LanguageTool is a open source grammar checker server written in Java.

[^2]: Actually, there aready was a [languagetool](https://github.com/patronus-checker/languagetool-rs) crate, but it seemed to be abandoned and only implemented basic bindings.

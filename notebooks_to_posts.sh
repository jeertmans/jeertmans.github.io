#!/bin/bash

[ -d _notebooks ] && [ "$(find _notebooks -type f -iname '*.ipynb')" ] && python3 -m jupyter nbconvert _notebooks/*.ipynb --ExecutePreprocessor.kernel_name=python3 --execute --to markdown --output-dir assets/images || echo "No notebook found"
find assets/images -type f -iname '*.md' | xargs -n 1 sed -i -E "s/(!\[svg\])\((.*)\)/\1(\/assets\/images\/\2)/"
mv assets/images/*.md _posts/

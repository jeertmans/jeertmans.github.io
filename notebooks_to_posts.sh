#!/bin/bash

[ -d _notebooks ] && [ "$(find _notebooks -type f -iname '*.ipynb')" ] && python3 -m jupyter nbconvert _notebooks/*.ipynb --ExecutePreprocessor.kernel_name=python3 --execute --to markdown --output-dir assets/images || echo "No notebook found"
find assets/images -type f -iname '*.md' | xargs -n 1 sed -i -E "s/(!\[svg\])\((.*)\)/\1(\/assets\/images\/\2)/"
find assets/images -type f -iname '*.md' | xargs -n 1 python3 -c "import sys,os;file=sys.argv[1];lines=open(file).readlines();index=lines.index('---\n');basename=os.path.basename(file);path=os.path.join('_notebooks', basename[:-2] + 'ipynb\n');lines.insert(index + 1, f'source: {path}');open(file, 'w').writelines(lines);"
mv assets/images/*.md _posts/

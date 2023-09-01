rm -rf ../../assets/slides/
find -iname "main.py" | xargs -I {} bash -c 'cd $(dirname {}) && manim "$(basename {})" --media_dir="../media/$(dirname {})" && manim-slides convert Main "../../assets/slides/$(dirname {}).html"'


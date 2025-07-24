## Image Diff Configuration

This branch uses a custom Git diff driver to visually compare image cache files using ImageMagick tools.

### Enabling the Image Diff Driver

To enable this locally, you need to do two things:

1. **Configure the `image` diff driver command**

Set the custom diff command in your local Git configuration by running:

```
git config diff.image.command "sh -c 'compare -highlight-color black -compose src \$4 \$1 png:- | montage -geometry +4+4 \$4 - \$1 png:- | display -title \"\$1\" -'"
```

This command uses ImageMagick's `compare`, `montage`, and `display` utilities to visually highlight image differences during `git diff`.

2. **Assign the diff driver to image files**

Add the following lines to the local `.git/info/attributes` file:

```
*.gif diff=image
*.jpg diff=image
*.png diff=image
```

### Requirements

- [ImageMagick](https://imagemagick.org) installed and available in your PATH.
- X11 or an image viewer compatible with `display`.

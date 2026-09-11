"""Data-exploration tools: listing files, peeking at CSV schema/samples, and
describing product images for the agent."""

from __future__ import annotations

from typing import Any

from .base import Tool


class ListFilesTool(Tool):
    name = "list_files"
    doc = """\
### list_files()
List all items in the data directory, showing whether each is a file or folder.

```python
list_files()
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self) -> str:
        lines = []
        for p in sorted(self._data_dir.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            lines.append(f"  [{kind}] {p.name}")
        return "Data directory contents:\n" + "\n".join(lines)


class ExploreSchemaT(Tool):
    name = "explore_schema"
    doc = """\
### explore_schema(filename)
Show the column names and dtypes of a CSV file in the data directory.

```python
explore_schema("Reviews.csv")
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self, filename: str) -> str:
        import pandas as pd
        df = pd.read_csv(self._data_dir / filename, nrows=0)
        lines = [f"  {col}: {dtype}" for col, dtype in df.dtypes.items()]
        return f"{filename} schema:\n" + "\n".join(lines)


class ExploreSampleTool(Tool):
    name = "explore_sample"

    # Free-text columns (contracts, reviews, articles) can hold tens of thousands of characters per
    # cell, so a raw `to_string` dump would flood the agent's context with a single row. Cap both the
    # per-cell width and the whole block; `explore_data` is the way to get untruncated values.
    MAX_CELL_CHARS = 300
    MAX_OUTPUT_CHARS = 4000

    doc = """\
### explore_sample(filename, n=5)
Show the first `n` rows of a CSV file in the data directory. Returns a formatted STRING (not a
DataFrame); long cells are truncated — use `explore_data(filename)` when you need full values.

```python
explore_sample("Reviews.csv", n=3)
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self, filename: str, n: int = 5) -> str:
        import pandas as pd
        df = pd.read_csv(self._data_dir / filename, nrows=n)
        body = df.to_string(index=False, max_colwidth=self.MAX_CELL_CHARS)
        if len(body) > self.MAX_OUTPUT_CHARS:
            omitted = len(body) - self.MAX_OUTPUT_CHARS
            body = body[: self.MAX_OUTPUT_CHARS] + f"\n... [{omitted} chars omitted]"
        return (
            f"{filename} sample ({n} rows; cells truncated to {self.MAX_CELL_CHARS} chars, "
            f"block to {self.MAX_OUTPUT_CHARS}):\n{body}"
        )


class ExploreImagesTool(Tool):
    name = "explore_images"
    MAX_IMAGES = 5

    # Rendered per-instance in __init__ so the prompt names this dataset's actual id column/folder.
    _DOC_TEMPLATE = """\
### explore_images(ids, question=None)
Inspect up to {max_images} images (found in the data directory's `{subdir}/` folder, named by
`{id_col}`) during data exploration. Pass a list of ids taken from the `{id_col}` column of a CSV.
THIS step's observation returns an auto-generated TEXTUAL description of each image (a cheap vision
model reads the pixels for you); the images themselves are not attached to your context. Use this a
couple of times at most — images are expensive; do not stream the whole dataset through it.

`question` is an optional natural-language question the describer must address for every image, on
top of its default description — ask whatever would inform your plan (e.g. whether the attribute
your query filters on is visible at all, or how cluttered/ambiguous the pictures are).

```python
ids = explore_data("items.csv")["{id_col}"].head(3).tolist()
explore_images(ids, question="Is the garment's sleeve length clearly visible?")
```"""

    def __init__(
        self,
        data_dir: str,
        pending_images: list,
        *,
        subdir: str = "images",
        id_col: str = "idx",
    ) -> None:
        import pathlib
        self._images_dir = pathlib.Path(data_dir) / subdir
        self._id_col = id_col
        self._exts = (".jpg",)
        self._pending = pending_images  # shared buffer drained by the run loop
        self.doc = self._DOC_TEMPLATE.format(max_images=self.MAX_IMAGES, subdir=subdir, id_col=self._id_col)

    def _find_image(self, image_id: Any):
        for ext in self._exts:
            p = self._images_dir / f"{image_id}{ext}"
            if p.is_file():
                return p
        return None

    def __call__(self, ids: Any, question: str | None = None) -> str:
        import base64
        import mimetypes

        if not self._images_dir.is_dir():
            return f"No {self._images_dir.name}/ directory found at {self._images_dir} — this dataset has no images."
        if not isinstance(ids, (list, tuple)):
            ids = [ids]
        note = ""
        if len(ids) > self.MAX_IMAGES:
            note = f" (capped at {self.MAX_IMAGES}; ignored {len(ids) - self.MAX_IMAGES} extra id(s))"
            ids = list(ids)[: self.MAX_IMAGES]

        attached, missing = [], []
        for image_id in ids:
            path = self._find_image(image_id)
            if path is None:
                missing.append(str(image_id))
                continue
            data = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
            b64 = base64.b64encode(data).decode("ascii")
            self._pending.append({
                "id": str(image_id),
                "url": f"data:{mime};base64,{b64}",
                "question": question,
            })
            attached.append(str(image_id))

        lines = [f"Attaching {len(attached)} image(s){note}; they appear below as vision inputs in your next step."]
        if attached:
            lines.append(f"  shown {self._id_col}s: {attached}")
        if question:
            lines.append(f"  describer asked: {question}")
        if missing:
            lines.append(f"  no image file found for {self._id_col}s: {missing}")
        return "\n".join(lines)

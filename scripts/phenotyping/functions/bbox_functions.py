from pathlib import Path
import re


def find_bbox_json_file(bboxes_folder: Path, id_date: str) -> Path:
    """
    Look for exactly one JSON file in `bboxes_folder` whose filename
    matches the pattern:  ^{id_date}_.*_bboxes\.json$
    If found, return its Path. Otherwise, raise FileNotFoundError.

    Args:
        bboxes_folder (Path): folder to search in
        id_date (str): prefix (e.g. "461_1707") to match at the start of the filename

    Returns:
        Path: full path to the matching .json file

    Raises:
        FileNotFoundError: if no matching file is found
        RuntimeError: if more than one matching file is found
    """
    # build a regex like r"^461_1707_.*_bboxes\.json$"
    pattern = re.compile(rf"^{re.escape(id_date)}_.*_bboxes\.json$")
    matches = []
    for f in bboxes_folder.iterdir():
        if f.is_file() and pattern.match(f.name):
            matches.append(f)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) == 0:
        raise FileNotFoundError(
            f"No JSON file starting with '{id_date}_' and ending with '_bboxes.json' found in {bboxes_folder}")
    else:
        raise RuntimeError(f"More than one JSON file matched '{id_date}_*_bboxes.json' in {bboxes_folder}: {matches}")

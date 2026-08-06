from __future__ import annotations

from pathlib import Path
import shutil
import sys


def find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False

    for index in range(open_index, len(text)):
        char = text[index]

        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index

    raise ValueError("Could not find the end of SeparableConv1D(...).")


def patch_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    marker = "layers.SeparableConv1D("
    search_from = 0
    replacements = 0

    while True:
        call_start = text.find(marker, search_from)
        if call_start == -1:
            break

        open_paren = call_start + len("layers.SeparableConv1D")
        close_paren = find_matching_paren(text, open_paren)
        block = text[call_start : close_paren + 1]

        old_double = 'padding="causal"'
        old_single = "padding='causal'"

        if old_double in block:
            new_block = block.replace(old_double, 'padding="same"', 1)
        elif old_single in block:
            new_block = block.replace(old_single, 'padding="same"', 1)
        else:
            search_from = close_paren + 1
            continue

        text = text[:call_start] + new_block + text[close_paren + 1 :]
        replacements += 1
        search_from = call_start + len(new_block)

    if replacements == 0:
        return 0

    backup = path.with_suffix(path.suffix + ".before_padding_fix.bak")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    print(f"Backup created: {backup}")
    return replacements


def main() -> int:
    path = Path("src/models/lstm/train.py")
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return 1

    replacements = patch_file(path)
    if replacements == 0:
        print(
            "No SeparableConv1D padding='causal' setting was found. "
            "The file may already be fixed or a different file is running."
        )
        return 2

    print(f"Patched {replacements} SeparableConv1D layer(s) in: {path}")
    print("Changed padding='causal' to padding='same'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
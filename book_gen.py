import json
import pyrev
import numpy as np
from pyrev import Position

BOOK_FILE = "opening_lines.json"


# ==========================================
# 定石一覧
# ==========================================
# 各要素:
# [
#     "F5", "D6", ...
# ]
#
# 必要に応じて追加してください
# ==========================================
RAW_OPENING_LINES = """
f5
f5d6
f5d6c3g5
f5d6c3g5c6c5
f5d6c3g5c6c5c4b6
f5d6c3g5c6c5c4b6f6f4
f5d6c3g5c6c5c4b6f6f4e6d7
f5d6c3g5c6c5c4b6f6f4e6d7c7g6
f5d6c3g5c6c5c4b6f6f4e6d7c7g6d8b5
f5d6c3g5c6c5c4b6f6f4e6d7c7g6d8b5e7b3
f5d6c3g5c6c5c4b6f6f4e6d7c7g6d8b5e7b3a6e3
f5d6c3g5c6c5c4b6f6f4e6d7c7g6d8b5e7b3a6e3a5d3
f5d6c3g5f6d3
f5d6c3g5f6d3e3c2
f5d6c3g5f6d3e3c2c1e6
f5d6c3g5f6d3e3c2c1e6f4f3
f5d6c3g5f6d3e3c2c1e6f4f3f2g4
f5d6c3g5f6d3e3c2c1e6f4f3f2g4g6d2
f5d6c3g5f6d3e3c2c1e6f4f3f2g4g6d2h3h4
f5d6c3g5f6d3e3c2c1e6f4f3f2g4g6d2h3h4h5f7
f5d6c3g5f6d3e3c2c1e6f4f3f2g4g6d2h3h4h5f7e7g3
f5d6c3g5g6d3
f5d6c3g5g6d3c4e3
f5d6c3g5g6d3c4e3f3b4
f5d6c3g5g6d3c4e3f3b4f6e6
f5d6c3g5g6d3c4e3f3b4f6e6f4g4
f5d6c3g5g6d3c4e3f3b4f6e6f4g4h4h5
f5d6c3g5g6d3c4e3f3b4f6e6f4g4h4h5h6g3
f5d6c3g5g6d3c4e3f3b4f6e6f4g4h4h5h6g3h3f7
f5d6c3g5g6d3c4e3f3b4f6e6f4g4h4h5h6g3h3f7f8c2
f5d6c4b3
f5d6c4b3b4f4
f5d6c4b3b4f4f6g5
f5d6c4b3b4f4f6g5f3e7
f5d6c4b3b4f4f6g5f3e7c5e6
f5d6c4b3b4f4f6g5f3e7c5e6c3g4
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3f2b6
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3f2b6h4d3
f5d6c5b4
f5d6c5b4d7e7
f5d6c5b4d7e7c7d8
f5d6c5b4d7e7c7d8c3d3
f5d6c5b4d7e7c7d8c3d3c4b3
f5d6c5b4d7e7c7d8c3d3c4b3d2e2
f5d6c5b4d7e7c7d8c3d3c4b3d2e2c2e3
f5d6c5b4d7e7c7d8c3d3c4b3d2e2c2e3f4f2
f5d6c5b4d7e7c7d8c3d3c4b3d2e2c2e3f4f2c6b5
f5d6c5b4d7e7c7d8c3d3c4b3d2e2c2e3f4f2c6b5f3c8
f5d6c4
f5d6c4b3b4
f5d6c4b3b4f4f6
f5d6c4b3b4f4f6g5f3
f5d6c4b3b4f4f6g5f3e7c5
f5d6c4b3b4f4f6g5f3e7c5e6c3
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3f2
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3f2b6h4
f5d6c4b3b4f4f6g5f3e7c5e6c3g4c6g3h3e3f2b6h4d3e2
f5d6c4d3c3
f5d6c4d3c3b3d2
f5d6c4d3c3b3d2e1b5
f5d6c4d3c3b3d2e1b5c5b4
f5d6c4d3c3b3d2e1b5c5b4e3c2
f5d6c4d3c3b3d2e1b5c5b4e3c2a4c6
f5d6c4d3c3b3d2e1b5c5b4e3c2a4c6d1e2
f5d6c4d3c3b3d2e1b5c5b4e3c2a4c6d1e2c7b6
f5d6c4d3c3b3d2e1b5c5b4e3c2a4c6d1e2c7b6f1e6
f5d6c4d3c3b3d2e1b5c5b4e3c2a4c6d1e2c7b6f1e6f3f2
f5d6c4d3c3f4f6
f5d6c4d3c3f4f6f3e6
f5d6c4d3c3f4f6f3e6e7f7
f5d6c4d3c3f4f6f3e6e7f7c5b6
f5d6c4d3c3f4f6f3e6e7f7c5b6g5e3
f5d6c4d3c3f4f6f3e6e7f7c5b6g5e3d7c6
f5d6c4d3c3f4f6f3e6e7f7c5b6g5e3d7c6e2g4
f5d6c4d3c3f4f6f3e6e7f7c5b6g5e3d7c6e2g4h3d2
f5d6c4d3c3f4f6f3e6e7f7c5b6g5e3d7c6e2g4h3d2g3f1
f5d6c4d3c3f4f6f3e6e7f7c5b6g6e3
f5d6c4d3c3f4f6f3e6e7f7c5b6g6e3e2f1
f5d6c4d3c3f4f6f3e6e7f7c5b6g6e3e2f1d1g5
f5d6c4d3c3f4f6f3e6e7f7c5b6g6e3e2f1d1g5c6d8
f5d6c4d3c3f4f6f3e6e7f7c5b6g6e3e2f1d1g5c6d8g4h6
f5d6c4d3c3f4f6b4c2
f5d6c4d3c3f4f6b4c2f3e3
f5d6c4d3c3f4f6b4c2f3e3e2c6
f5d6c4d3c3f4f6b4c2f3e3e2c6f2c5
f5d6c4d3c3f4f6b4c2f3e3e2c6f2c5e6d2
f5d6c4d3c3f4f6b4c2f3e3e2c6f2c5e6d2g4d7
f5d6c4d3c3f4f6b4c2f3e3e2c6f2c5e6d2g4d7b3g5
f5d6c4d3c3f4f6b4c2f3e3e2c6f2c5e6d2g4d7b3g5c8h4
f5d6c4d3c3f4f6g5e3
f5d6c4d3c3f4f6g5e3f3g6
f5d6c4d3c3f4f6g5e3f3g6e2h5
f5d6c4d3c3f4f6g5e3f3g6e2h5c5g4
f5d6c4d3c3f4f6g5e3f3g6e2h5c5g4g3f2
f5d6c4d3c3b5b4
f5d6c4d3c3b5b4f4c5
f5d6c4d3c3b5b4f4c5a4b3
f5d6c4d3c3b5b4f4c5a4b3d2a6
f5d6c4d3c3b5b4f4c5a4b3d2a6a3e3
f5d6c4d3c3b5b4f4c5a4b3d2a6a3e3f3g4
f5d6c4d3c3b5b4f4c5a4b3d2a6a3e3f3g4e6f6
f5d6c4d3c3b5b4f4c5a4b3d2a6a3e3f3g4e6f6g3e2
f5d6c4d3c3b5b4f4c5a4b3d2a6a3e3f3g4e6f6g3e2c2f2
f5d6c4g5f6
f5d6c4g5f6f4f3
f5d6c4g5f6f4f3d3c3
f5d6c4g5f6f4f3d3c3g6e3
f5d6c4g5f6f4f3d3c3g6e3e6h5
f5d6c4g5f6f4f3d3c3g6e3e6h5d2e2
f5d6c4g5f6f4f3d3c3g6e3e6h5d2e2c2c6
f5d6c4g5f6f4f3d3c3g6e3e6h5d2e2c2c6c5b6
f5d6c4g5f6f4f3d3c3g6e3e6h5d2e2c2c6c5b6b4b3
f5d6c4g5f6f4f3d3c3g6e3e6h5d2e2c2c6c5b6b4b3c7a4
f5f6e6
f5f6e6f4g6
f5f6e6f4g6c5f3
f5f6e6f4g6c5f3g4e3
f5f6e6f4g6c5f3g4e3d6g5
f5f6e6f4g6c5f3g4e3d6g5g3c3
f5f6e6f4g6c5f3g4e3d6g5g3c3h5c4
f5f6e6f4g6c5f3g4e3d6g5g3c3h5c4d7h6
f5f6e6f4g6c5f3g4e3d6g5g3c3h5c4d7h6h7h3
f5f6e6f4g6c5f3g4e3d6g5g3c3h5c4d7h6h7h3f7e7
f5f6e6f4g6c5f3g4e3d6g5g3c3h5c4d7h6h7h3f7e7f8h4
f5f6e6f4g6c5f3g5d6
f5f6e6f4g6c5f3g5d6e3h4
f5f6e6f4g6c5f3g5d6e3h4g3g4
f5f6e6f4g6c5f3g5d6e3h4g3g4h6e2
f5f6e6f4g6c5f3g5d6e3h4g3g4h6e2d3h5
f5f6e6f4g6c5f3g5d6e3h4g3g4h6e2d3h5h3c6
f5f6e6f4g6c5f3g5d6e3h4g3g4h6e2d3h5h3c6e7f2
f5f6e6f4g6c5f3g5d6e3h4g3g4h6e2d3h5h3c6e7f2c4d2
f5f6e6f4g6d6g4
f5f6e6f4g6d6g4g5h4
f5f6e6f4g6d6g4g5h4e7f3
f5f6e6f4g6d6g4g5h4e7f3h6f7
f5f6e6f4g6d6g4g5h4e7f3h6f7e8f8
f5f6e6f4g6d6g4g5h4e7f3h6f7e8f8g8d3
f5f6e6f4g6d6g4g5h4e7f3h6f7e8f8g8d3h5h7
f5f6e6f4g6d6g4g5h4e7f3h6f7e8f8g8d3h5h7e3c5
f5f6e6f4g6d6g4g5h4e7f3h6f7e8f8g8d3h5h7e3c5c4g3
f5f6e6d6f7
f5f6e6d6f7e3c6
f5f6e6d6f7e3c6e7f4
f5f6e6d6f7e3c6e7f4c5d8
f5f6e6d6f7e3c6e7f4c5d8c7d7
f5f6e6d6f7e3c6e7f4c5d8c7d7f8b5
f5f6e6d6f7e3c6e7f4c5d8c7d7f8b5c4e8
f5f6e6d6f7e3c6e7f4c5d8c7d7f8b5c4e8c8f3
f5f6e6d6f7e3c6e7f4c5d8c7d7f8b5c4e8c8f3g5b6
f5f6e6d6f7e3c6e7f4c5d8c7d7f8b5c4e8c8f3g5b6d3b4
f5f6e6d6f7f4d7
f5f6e6d6f7f4d7e7d8
f5f6e6d6f7f4d7e7d8g5c6
f5f6e6d6f7f4d7e7d8g5c6f8g6
f5f6e6d6f7f4d7e7d8g5c6f8g6h5h6
f5f6e6d6f7f4d7e7d8g5c6f8g6h5h6h7c4
f5f6e6d6f7f4d7e7d8g5c6f8g6h5h6h7c4e8g8
f5f6e6d6f7f4d7e7d8g5c6f8g6h5h6h7c4e8g8c5e3
f5f6e6d6f7f4d7e7d8g5c6f8g6h5h6h7c4e8g8c5e3d3c7
""".strip()


def parse_opening_line(raw_line):
    raw_line = raw_line.strip().upper()

    if len(raw_line) % 2 != 0:
        raise ValueError(f"invalid opening line: {raw_line}")

    return [
        raw_line[i:i + 2]
        for i in range(0, len(raw_line), 2)
    ]


def parse_opening_lines(raw_text):
    lines = []

    for raw_line in raw_text.splitlines():
        raw_line = raw_line.strip()

        if not raw_line:
            continue

        lines.append(parse_opening_line(raw_line))

    return lines

OPENING_LINES = parse_opening_lines(RAW_OPENING_LINES)

def coord_to_xy(coord):
    s = pyrev.coord_to_str(np.int8(coord))
    x = ord(s[0]) - ord("A")
    y = int(s[1]) - 1
    return x, y


def xy_to_coord(x, y):
    s = chr(ord("A") + x) + str(y + 1)
    return int(pyrev.parse_coord_str(s))


def transform_coord(coord, t):
    x, y = coord_to_xy(coord)

    if t == 0:
        nx, ny = x, y
    elif t == 1:
        nx, ny = 7 - y, x
    elif t == 2:
        nx, ny = 7 - x, 7 - y
    elif t == 3:
        nx, ny = y, 7 - x
    elif t == 4:
        nx, ny = 7 - x, y
    elif t == 5:
        nx, ny = x, 7 - y
    elif t == 6:
        nx, ny = y, x
    elif t == 7:
        nx, ny = 7 - y, 7 - x
    else:
        raise ValueError("invalid transform id")

    return xy_to_coord(nx, ny)


def is_legal_line(line):
    pos = Position()

    for move_str in line:
        move = np.int8(pyrev.parse_coord_str(move_str))

        if not pos.is_legal(move):
            return False

        pos.do_move_at(move)

    return True


def generate_lines():
    saved_lines = []

    for line in OPENING_LINES:
        base_moves = [
            int(pyrev.parse_coord_str(s))
            for s in line
        ]

        for t in range(8):
            transformed_line = [
                pyrev.coord_to_str(np.int8(transform_coord(move, t)))
                for move in base_moves
            ]

            if is_legal_line(transformed_line):
                print(f"[add ] transform {t}: {transformed_line}")
                saved_lines.append(transformed_line)
            else:
                print(f"[skip] transform {t}: {transformed_line}")

    with open(BOOK_FILE, "w", encoding="utf-8") as f:
        json.dump(saved_lines, f, indent=2, ensure_ascii=False)

    print(f"saved file: {BOOK_FILE}")
    print(f"black_lines: {len(saved_lines)}")


if __name__ == "__main__":
    generate_lines()
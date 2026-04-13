import pyrev
from pyrev import Position
import ab_ex


def color_name(color) -> str:
    if color == pyrev.BLACK:
        return "Black"
    if color == pyrev.WHITE:
        return "White"
    return "Unknown"


def input_human_color():
    while True:
        s = input("あなたの色を選んでください (B/W): ").strip().upper()
        if s == "B":
            return pyrev.BLACK
        if s == "W":
            return pyrev.WHITE
        print("B か W を入力してください。")


def legal_moves_as_strings(pos: Position):
    moves = list(pos.get_legal_moves())
    return moves, [pyrev.coord_to_str(m) for m in moves]


def human_turn(pos: Position):
    moves, move_strs = legal_moves_as_strings(pos)
    print(f"合法手: {', '.join(move_strs)}")

    while True:
        s = input("着手座標を入力してください (例: F5): ").strip()

        if not s:
            print("入力が空です。")
            continue

        try:
            coord = pyrev.parse_coord_str(s)
        except Exception:
            print("座標の形式が不正です。例: D3, F5")
            continue

        if pos.do_move_at(coord):
            print(f"あなたの着手: {pyrev.coord_to_str(coord)}")
            return

        print("その手は打てません。もう一度入力してください。")


def cpu_turn(pos: Position):
    move = ab_ex.alpha_beta(pos, depth=3)  # 深さ3で探索

    pos.do_move_at(move)
    print(f"エージェントの着手: {pyrev.coord_to_str(move)}")


def print_status(pos: Position, human_color):
    print()
    print(pos)
    print(f"現在の手番: {color_name(pos.side_to_move)}")
    print(f"あなた: {color_name(human_color)} / 相手: {color_name(pyrev.to_opponent_color(human_color))}")
    print()


def print_result(pos: Position, human_color):
    human_diff = int(pos.get_score_from(human_color))
    agent_color = pyrev.to_opponent_color(human_color)

    print("\n===== 終局 =====")
    print(pos)
    print()

    if human_diff > 0:
        print(f"あなたの勝ちです。石差: +{human_diff}")
    elif human_diff < 0:
        print(f"エージェント ({color_name(agent_color)}) の勝ちです。石差: {human_diff}")
    else:
        print("引き分けです。石差: 0")


def main():
    print("=== PyRev エージェント対戦 ===")
    print("入力例: D3, F5")
    print()

    human_color = input_human_color()
    pos = Position()

    while not pos.is_gameover():
        print_status(pos, human_color)

        legal_moves = list(pos.get_legal_moves())

        if not legal_moves:
            if pos.can_pass():
                if pos.side_to_move == human_color:
                    print("あなたはパスです。")
                else:
                    print("エージェントはパスです。")
                pos.do_pass()
                continue
            else:
                break

        if pos.side_to_move == human_color:
            human_turn(pos)
        else:
            cpu_turn(pos)

    print_result(pos, human_color)


if __name__ == "__main__":
    main()
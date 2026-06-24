"""
generate_edb.py - 終盤データベース (EDB) 並列生成スクリプト

【GPU について】
EDB 生成の中核はαβ探索（再帰的な木探索）であり、GPU には不向きです。
  - 各ノードの評価結果が子ノードに依存する逐次依存構造
  - ノードごとに分岐数が異なる不規則な制御フロー
  - キャッシュ効率の悪いランダムメモリアクセス
  - 深さ優先の再帰はGPUの命令並列性と相性が悪い
GPU が得意とするのは「同じ演算を大量のデータに並列適用」するもの
（行列積・畳み込みなど）であり、木探索とは根本的に異なります。
→ CPU マルチプロセスが EDB 生成に適した並列化手段です。

【CPU 並列化の設計】
旧実装:  ゲーム数 ÷ ワーカー数 の大きなタスクを 1ワーカー 1タスクで処理
          → 早く終わったワーカーが後半アイドル状態になる

新実装:  GAMES_PER_TASK (デフォルト500) ゲームの小タスクを動的に配布
          → 全ワーカーが最後まで均等に稼働する (Futures + as_completed)
          → 長時間ゲームが多い局面でも遊びプロセスが出ない

【キー設計】
  ファイルのキー: (player_bits, opponent_bits) 整数タプル
  - Python の hash() はプログラム実行ごとに変わるため不使用
  - PyRev の calc_hash_code() も Zobrist テーブルの安定性が実装依存のため不使用
  - 整数ビット列は常に同じ値が得られる安定したキー

  エージェント起動時に calc_hash_code() キーへ変換する場合は
  load_edb_as_hash_dict() を使用する（PyRev の Position 再構築 API が必要）

【値設計】
  値は現在の手番側から見たスコア (agent の endgame_search と同じスケール)
    WIN_SCORE  ( 100000): 勝ち
    0                  : 引き分け
    -WIN_SCORE (-100000): 負け

使い方:
  python generate_edb.py                             # デフォルト (6マス / 10万ゲーム)
  python generate_edb.py --empty 8 --games 500000    # 8マス 50万ゲーム
  python generate_edb.py --workers 16                # プロセス数を指定
  python generate_edb.py --append edb_6.pkl          # 既存 EDB に追記
  python generate_edb.py --stats   edb_6.pkl         # 統計のみ表示
"""

import argparse
import os
import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pyrev

# ==============================
# 設定定数
# ==============================

WIN_SCORE = 100_000

# 1タスクあたりのゲーム数。
# 大きいほどタスク生成・転送オーバーヘッドが減る。
# 小さいほど負荷分散が均等になる。
# 目安: max_workers × 20 以上のタスク数になる値を推奨。
GAMES_PER_TASK = 500

# 何タスク完了ごとに中間保存するか (0 = 中間保存なし)
SAVE_INTERVAL = 20


# ==============================
# ビットボード操作
# ==============================

def _get_bitboards(pos) -> tuple:
    """
    pos から (player_bits, opponent_bits) のタプルを生成する。

    player_bits  : pos.side_to_move の石を 64bit 整数で表現
    opponent_bits: 相手の石を 64bit 整数で表現
    """
    p = o = 0
    for coord in pos.get_player_disc_coords():
        p |= (1 << int(coord))
    for coord in pos.get_opponent_disc_coords():
        o |= (1 << int(coord))
    return (p, o)


# ==============================
# 終盤完全読み (EDB 生成用)
# ==============================

def _solve(pos, alpha: int, beta: int, memo: dict) -> int:
    """
    局面 pos の最善スコアを完全読みして memo に格納する。

    訪問した全局面 (pos 以下の全子局面) を memo に登録するため、
    1回の呼び出しで多数の局面を一括登録できる。

    Returns
    -------
    int : pos.side_to_move 視点のスコア (WIN_SCORE / 0 / -WIN_SCORE)
    """
    key = _get_bitboards(pos)
    cached = memo.get(key)
    if cached is not None:
        return cached

    if pos.is_gameover():
        p = pos.player_disc_count
        o = pos.opponent_disc_count
        score = WIN_SCORE if p > o else (-WIN_SCORE if p < o else 0)
        memo[key] = score
        return score

    actions = list(pos.get_legal_moves())

    if not actions:
        child = pos.copy()
        child.do_pass()
        score = -_solve(child, -beta, -alpha, memo)
        memo[key] = score
        return score

    best = -(WIN_SCORE + 1)
    for action in actions:
        child = pos.copy()
        child.do_move_at(np.int8(action))
        score = -_solve(child, -beta, -alpha, memo)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    memo[key] = best
    return best


# ==============================
# ワーカー関数 (小粒度タスク)
# ==============================

def _worker(task: tuple) -> dict:
    """
    ProcessPoolExecutor ワーカー。

    GAMES_PER_TASK 局のランダムゲームを実行し、
    empty_square_count <= edb_limit になった時点で _solve を呼んで
    局面を収集する。

    【小粒度タスクキュー方式の利点】
    旧実装では num_games / max_workers ゲームを1ワーカーに固定割り当て
    していたが、それだと早く終わったワーカーがアイドル状態になる。
    本実装では GAMES_PER_TASK ゲームの小タスクを動的に配布するため、
    全ワーカーが最後まで均等に稼働する。

    Returns
    -------
    dict : {(player_bits, opponent_bits): score} の部分 EDB
    """
    num_games, edb_limit, seed = task
    random.seed(seed)
    np.random.seed(seed % (2**32))

    local_edb: dict = {}

    for _ in range(num_games):
        pos = pyrev.Position()

        while not pos.is_gameover():
            if pos.empty_square_count <= edb_limit:
                _solve(pos, -WIN_SCORE, WIN_SCORE, local_edb)
                break

            actions = list(pos.get_legal_moves())
            if actions:
                pos.do_move_at(np.int8(random.choice(actions)))
            elif pos.can_pass():
                pos.do_pass()
            else:
                break

    return local_edb


# ==============================
# EDB 生成メイン
# ==============================

def generate_edb(
    num_games:    int  = 100_000,
    edb_limit:    int  = 6,
    max_workers:  int  = None,
    output:       str  = None,
    append_file:  str  = None,
    save_interval: int = SAVE_INTERVAL,
    games_per_task: int = GAMES_PER_TASK,
) -> dict:
    """
    ランダム自己対局で終盤局面を収集し、EDB をファイルに保存する。

    Parameters
    ----------
    num_games     : 総ゲーム数
    edb_limit     : 収集対象の空きマス数上限
    max_workers   : 並列プロセス数 (None = CPU コア数)
    output        : 保存先ファイルパス
    append_file   : 追記元ファイルパス (指定時は読み込んで追記)
    save_interval : 何タスク完了ごとに中間保存するか (0 = しない)

    Returns
    -------
    dict : {(player_bits, opponent_bits): score}
    """
    if output is None:
        output = f"edb_{edb_limit}.pkl"

    max_workers = max(1, max_workers or os.cpu_count() or 1)

    # 既存 EDB を読み込んで追記
    edb: dict = {}
    if append_file and os.path.exists(append_file):
        print(f"[EDB] 既存ファイル読み込み中: {append_file}")
        with open(append_file, "rb") as f:
            edb = pickle.load(f)
        print(f"  既存エントリ数: {len(edb):,}")

    # タスク生成: GAMES_PER_TASK ゲームの小タスクに分割
    # 多数の小タスクにすることで、全ワーカーが最後まで均等に稼働する
    task_list = []
    total_assigned = 0
    seed = 0
    while total_assigned < num_games:
        batch = min(games_per_task, num_games - total_assigned)
        task_list.append((batch, edb_limit, seed))
        total_assigned += batch
        seed += 1

    num_tasks = len(task_list)

    print(f"[EDB] 生成開始 (CPU 並列)")
    print(f"  対象空きマス数   : {edb_limit}")
    print(f"  総ゲーム数       : {num_games:,}")
    print(f"  タスク数         : {num_tasks:,}  ({games_per_task} games/task)")
    print(f"  並列プロセス数   : {max_workers}")
    print(f"  出力ファイル     : {output}")
    print()

    start         = time.perf_counter()
    completed     = 0
    games_done    = 0
    next_save_at  = save_interval  # 次の中間保存タイミング

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, task): task for task in task_list}

        for future in as_completed(futures):
            local_edb          = future.result()
            task_games, _, _   = futures[future]
            games_done        += task_games
            edb.update(local_edb)
            completed         += 1

            # 進捗表示
            elapsed   = time.perf_counter() - start
            speed     = games_done / elapsed if elapsed > 0 else 0
            remaining = (num_games - games_done) / speed if speed > 0 else 0

            print(f"  [{completed:>{len(str(num_tasks))}}/{num_tasks}]"
                  f"  games: {games_done:>{len(str(num_games))},}/{num_games:,}"
                  f"  EDB: {len(edb):>10,}"
                  f"  {speed:6.0f} g/s"
                  f"  残り: {remaining:5.0f}s"
                  f"  経過: {elapsed:6.1f}s")

            # 中間保存
            if save_interval > 0 and completed >= next_save_at:
                _save_edb(edb, output, verbose=False)
                print(f"  ↑ 中間保存完了 ({len(edb):,} エントリ)")
                next_save_at += save_interval

    elapsed = time.perf_counter() - start
    print()
    print(f"[EDB] 生成完了")
    print(f"  総局面数  : {len(edb):,}")
    print(f"  総時間    : {elapsed:.1f}s")
    print(f"  速度      : {num_games / elapsed:,.0f} games/s")

    _save_edb(edb, output, verbose=True)
    return edb


# ==============================
# 保存・読み込み
# ==============================

def _save_edb(edb: dict, filepath: str, verbose: bool = True) -> None:
    """EDB を pickle 形式で保存する。"""
    with open(filepath, "wb") as f:
        pickle.dump(edb, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        size_mb = os.path.getsize(filepath) / (1024 ** 2)
        print(f"  保存先        : {filepath}")
        print(f"  ファイルサイズ: {size_mb:.1f} MB")


def load_edb(filepath: str) -> dict:
    """
    EDB ファイルを読み込む。

    Returns
    -------
    dict : {(player_bits, opponent_bits): score}
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_edb_as_hash_dict(filepath: str) -> dict:
    """
    EDB を読み込み、calc_hash_code() キーの辞書に変換して返す。

    (player_bits, opponent_bits) → calc_hash_code() への変換を
    エージェント起動時に1回だけ行い、検索時のビットボード計算コストを
    ゼロにする。

    PyRev の Position(player_bits, opponent_bits) 直接構築 API が必要。
    API が存在しない場合は None を返す。

    Note
    ----
    calc_hash_code() は Zobrist ハッシュで実装されており、
    PyRev バイナリが同じであれば実行ごとに同じ値を返す。
    Python の hash() と異なり実行ごとに変わらない。

    Returns
    -------
    dict または None
    """
    edb_raw = load_edb(filepath)
    edb_hash = {}
    failed = 0

    for (player_bits, opponent_bits), score in edb_raw.items():
        pos = _reconstruct_position(player_bits, opponent_bits)
        if pos is None:
            failed += 1
            continue
        key = int(pos.calc_hash_code()) * 2 + int(pos.side_to_move)
        edb_hash[key] = score

    if failed > 0:
        print(f"[EDB] 警告: {failed} 局面の再構築に失敗 "
              f"(PyRev が Position 直接構築 API を持たない可能性)")
        return None

    return edb_hash


def _reconstruct_position(player_bits: int, opponent_bits: int):
    """
    (player_bits, opponent_bits) から PyRev の Position を再構築する。
    PyRev がビットボード直接構築 API を持たない場合は None を返す。
    """
    try:
        return pyrev.Position(player_bits, opponent_bits)
    except TypeError:
        pass
    try:
        pos = pyrev.Position()
        pos.set_bitboards(player_bits, opponent_bits)
        return pos
    except AttributeError:
        pass
    return None


# ==============================
# 統計
# ==============================

def print_stats(edb: dict) -> None:
    total  = len(edb)
    wins   = sum(1 for v in edb.values() if v > 0)
    draws  = sum(1 for v in edb.values() if v == 0)
    losses = sum(1 for v in edb.values() if v < 0)
    print(f"\n[EDB 統計]")
    print(f"  総局面数 : {total:,}")
    print(f"  勝ち     : {wins:,}  ({wins  / total * 100:.1f}%)")
    print(f"  引き分け : {draws:,}  ({draws / total * 100:.1f}%)")
    print(f"  負け     : {losses:,}  ({losses / total * 100:.1f}%)")


# ==============================
# エントリポイント
# ==============================

def main():
    parser = argparse.ArgumentParser(
        description="終盤データベース (EDB) 並列生成スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python generate_edb.py                              # デフォルト (6マス/10万ゲーム)
  python generate_edb.py --empty 8 --games 500000     # 8マスEDB 50万ゲーム
  python generate_edb.py --workers 16                 # プロセス数指定
  python generate_edb.py --append edb_6.pkl           # 既存EDBに追記
  python generate_edb.py --stats   edb_6.pkl          # 統計のみ表示
        """,
    )
    parser.add_argument("--empty",    type=int, default=6,
                        help="収集対象の空きマス数上限 (default: 6)")
    parser.add_argument("--games",    type=int, default=100_000,
                        help="総ゲーム数 (default: 100000)")
    parser.add_argument("--workers",  type=int, default=None,
                        help="並列プロセス数 (default: CPU コア数)")
    parser.add_argument("--output",   type=str, default=None,
                        help="出力ファイル名 (default: edb_{empty}.pkl)")
    parser.add_argument("--append",   type=str, default=None,
                        help="追記元の既存 EDB ファイルパス")
    parser.add_argument("--batch",    type=int, default=GAMES_PER_TASK,
                        help=f"1タスクあたりのゲーム数 (default: {GAMES_PER_TASK})")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                        help=f"中間保存間隔 (タスク完了数) (default: {SAVE_INTERVAL}, 0=無効)")
    parser.add_argument("--stats",    type=str, default=None,
                        help="指定ファイルの統計を表示して終了")
    args = parser.parse_args()

    if args.stats:
        print(f"[EDB] 読み込み中: {args.stats}")
        edb = load_edb(args.stats)
        print_stats(edb)
        return

    edb = generate_edb(
        num_games=args.games,
        edb_limit=args.empty,
        max_workers=args.workers,
        output=args.output,
        append_file=args.append,
        save_interval=args.save_interval,
        games_per_task=args.batch,
    )
    print_stats(edb)


if __name__ == "__main__":
    # Mac / Windows では ProcessPoolExecutor に spawn が使われるため
    # このガードが必須。ないとワーカー起動時に main() が再実行される。
    main()

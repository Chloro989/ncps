"""
センチュリオンの入口。基本の操作はすべてここから呼べる。

    python main.py                          何ができるかを出す
    python main.py read                     原稿を読む(窓が開く)
    python main.py read 第五稿.txt
    python main.py ask --mode 発想           問いを組む(出すだけ)
    python main.py ask --mode 発想 --api     その場で論評させる
    python main.py ask --mode 発想 --api --all   原稿を最初から最後まで
    python main.py check 答え.txt --chunk 2  段落番号を検査する
    python main.py write 青色にまつわる話を聞かせて
    python main.py test                     試験を全部走らせる

原稿のパスを省くと、手元のPCならファイル選択の窓、
Colab ならアップロードの窓が開く。
manuscripts/ に置いた原稿はファイル名だけで呼べる(あの中身は git に入らない)。

read / check はモデルを使わないので、GPUが無くても動く。
ask も既定ではプロンプトを出すだけで、それを好きなチャットへ貼れば
性能の高いモデルで読ませられる。

その場で論評まで出したいときは ask に次のどちらかを付ける。
  --api  Claude の API に解かせる。鍵は環境変数 ANTHROPIC_API_KEY から読む。
         文芸の論評に耐える質が要るならこちら
  --run  手元(か Colab)のモデルに解かせる。無料だが 3B級では質が出ない
どちらも、答えの段落番号が実在するかの検査まで自動で通す。

write だけは必ずモデルを読み込む。
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

USAGE = """センチュリオン

  web                  ブラウザから使う (127.0.0.1 にだけ開く)
                       llama-server の起動・停止もここからできる
  read   [原稿]        章・段落・切り出し・反復の一覧を見る
  ask    [原稿]        原稿を読ませる (--mode 発想/査読/採点/接続/連想)
  check  答え [原稿]   答えの段落番号が実在するかを検査する
  write  お題...       小説を書かせる (モデルが要る)
  prompts              プロンプトの文面を prompts/ に書き出す (編集用)
  test                 試験を全部走らせる

原稿のパスは省ける。手元のPCなら選択の窓、Colab ならアップロードの窓が開く。
manuscripts/ に置いた原稿はファイル名だけで呼べる。

論評させる:
  python main.py ask 第五稿.txt --mode 発想 --api --out 添削.txt
  python main.py ask 第五稿.txt --mode 発想 --api --all  最初から最後まで
  python main.py ask 第五稿.txt --mode 接続 --api --dream

  --out を付けると、本文の各段落の下にその段落あての指摘を貼った
  添削ファイルを書く。引用が本文と食い違う指摘には × が付く。

厳しさを選ぶ (査読と採点):
  python main.py ask 第五稿.txt --mode 査読 --severity 育成
  python main.py ask 第五稿.txt --mode 採点 --severity 厳格

  育成  良かった点を先に述べる。評価3が「アマチュアとして十分健闘」
  標準  良い点と問題点を同じ精度で述べる (既定)
  厳格  卓越性の有無だけを問う。評価1が「商業出版可能な一般的作品」

  採点は7観点 (構成/人物/文体/描写/対話/主題/世界観) の5段階評価。
  査読は観点を回して自由に論じる。同じ厳しさの指定が両方に効く。

プロンプトを書き換える:
  python main.py prompts               既定の文面を prompts/ に書き出す
  (prompts/査読-厳格.txt などを編集する)
  python main.py ask 第五稿.txt --mode 査読 --severity 厳格

  prompts/ にファイルがあればそれが使われる。消せば既定値に戻る。
  prompts/観点.txt を編集すれば観点そのものを足せる。

手元のモデルを見る:
  python main.py ask --models

  落としてある GGUF と、立っている llama-server に載っているモデルを
  実際に調べて並べる。書き並べた一覧ではないので、増減がそのまま出る。

観点の決め方:
  既定は原稿を実測して、足りていないところへ問いを向ける。
  python main.py ask 第五稿.txt --survey            何が測られたかを見る
  python main.py ask 第五稿.txt --lens 視点,熱量     観点を名指しする
  python main.py ask 第五稿.txt --random-lenses     くじ引きに戻す

  --api には鍵が要る。console.anthropic.com で作って環境変数に置く。
    Windows: setx ANTHROPIC_API_KEY "自分の鍵"   (設定後に端末を開き直す)
    Colab:   import os; os.environ['ANTHROPIC_API_KEY'] = '自分の鍵'
  この道具は鍵を保存も表示もしない。

鍵を使わない道:
  python main.py ask 第五稿.txt --mode 発想 > 問い.txt   問いを出して
  (問い.txt の中身を好きなチャットへ貼り、答えを 答え.txt に保存して)
  python main.py check 答え.txt 第五稿.txt --out 添削.txt

  check は段落番号の実在と、引用の中身が本文と一致するかを見る。
  番号だけの検査では、3Bモデルが引用14件中8件を取り違えた答えを
  素通りさせた。うち3件は本文に存在しない文だった。

そのほか:
  python main.py read
  python main.py write 朝の匂いについて書いて 沈黙について書いて --turns

各命令の詳しい引数は -h で見られる。

  python main.py ask -h
"""

TESTS = ["test_centurion", "test_manuscript", "test_review",
         "test_rubric", "test_connect", "test_answer", "test_verify",
         "test_critique", "test_server", "test_web", "test_main"]


def cmd_read(rest):
    """原稿の姿を見る。critique の --list と同じ中身"""
    from centurion.critique import main as critique_main
    return critique_main(rest + ["--list"])


def cmd_ask(rest):
    from centurion.critique import main as critique_main
    return critique_main(rest)


def cmd_check(rest):
    """答えのファイルを先に取り、残りを critique へ渡す"""
    if not rest:
        print("答えのファイルを渡すこと", file=sys.stderr)
        return 1
    answer, rest = rest[0], rest[1:]
    from centurion.critique import main as critique_main
    return critique_main(rest + ["--check", answer])


def cmd_write(rest):
    from centurion.__main__ import main as write_main
    return write_main(rest)


def cmd_web(rest):
    """ブラウザから使う。127.0.0.1 にだけ開く"""
    from centurion.web import main as web_main
    return web_main(rest)


def cmd_test(rest):
    """試験を全部走らせる。モデルは使わないので手元で完走する"""
    names = rest or TESTS
    total = failures = 0
    for name in names:
        path = HERE / "tests" / f"{name}.py"
        if not path.exists():
            print(f"{name}: 見つからない", file=sys.stderr)
            failures += 1
            continue
        done = subprocess.run([sys.executable, str(path)],
                              capture_output=True, text=True,
                              encoding="utf-8")
        last = (done.stdout.strip().splitlines() or ["出力なし"])[-1]
        print(f"{name:<18} {last}")
        if done.returncode:
            failures += 1
            for line in done.stdout.splitlines():
                if line.strip().startswith("×"):
                    print("   " + line.strip())
            if done.stderr.strip():
                print("   " + done.stderr.strip().splitlines()[-1])
        else:
            total += int(last.split("件")[0])
    if failures:
        print(f"\n{failures}件の試験が失敗した")
        return 1
    print(f"\n合計 {total}件すべて通過")
    return 0


def cmd_prompts(rest):
    """プロンプトの文面を prompts/ に書き出す。

    書き出したファイルを編集すれば、そのまま次回から使われる。
    ソースを書き換えずに文面を試せるようにするため"""
    from centurion import review, wording

    parser = argparse.ArgumentParser(
        prog="main.py prompts",
        description="プロンプトの文面を書き出して、編集できるようにする")
    parser.add_argument("--dir", metavar="置き場",
                        help=f"書き出す先 (既定 {wording.HOME.name}/)")
    parser.add_argument("--force", action="store_true",
                        help="既にあるファイルも既定値で上書きする")
    args = parser.parse_args(rest)

    written, skipped = review.export_wording(args.dir, args.force)
    folder = wording.home(args.dir)
    for path in written:
        print(f"書いた: {path.name}")
    for path in skipped:
        print(f"すでにある (触っていない): {path.name}")
    if skipped and not args.force:
        print()
        print("既定値に戻したいものは消すか、--force で上書きする")
    print()
    print(f"{folder} の中を編集すれば、次回からその文面が使われる。")
    print("ファイルを消せば組み込みの既定値に戻る。")
    return 0


COMMANDS = {
    "read": cmd_read,
    "prompts": cmd_prompts,
    "ask": cmd_ask,
    "check": cmd_check,
    "web": cmd_web,
    "write": cmd_write,
    "test": cmd_test,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"知らない命令: {name}\n", file=sys.stderr)
        print(USAGE)
        return 1
    return COMMANDS[name](rest)


if __name__ == "__main__":
    raise SystemExit(main())

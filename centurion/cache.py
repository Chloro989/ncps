"""
一度読んだものを覚えておく。

## なぜ

同じ原稿を何度も読み直す。モードを変えるたび、塊を変えるたび、
画面を開き直すたびに、同じ解析を最初からやり直していた。

解析のうち重いのは反復語の探索で、形態素解析を使うとさらに重くなる。
104段落の原稿で、正規表現なら一瞬だが形態素解析では数秒かかる。
一文字も変わっていない原稿に、毎回それを払う理由がない。

## 何を鍵にするか

**ファイルなら、道と更新時刻と大きさ。** 原稿を書き直せば更新時刻が
変わるので、覚えていたものは自然に捨てられる。中身の要約を取るより
速く、書き直しを取りこぼさない。

**貼り付けた本文なら、中身そのものの要約。** 貼り付けにはファイルが
無いので、中身から鍵を作るしかない。

## 何を覚えないか

モデルの答えは覚えない。同じ問いでも読みは毎回変わるべきもので、
覚えてしまうと「二度目も同じ答えが返る」ことに気づけない。
覚えるのは、原稿から機械的に決まるものだけにする。
"""

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

# 何件まで覚えておくか。原稿は数十件も同時に扱わないので、
# これで足りる。溢れたら古いものから捨てる
LIMIT = 24


class Memo:
    """鍵で引ける覚え書き。古いものから溢れる。

    画面は複数の求めを同時に捌くので、錠を掛ける"""

    def __init__(self, limit=LIMIT):
        self.limit = limit
        self.items = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                self.hits += 1
                return self.items[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.limit:
                self.items.popitem(last=False)
            return value

    def fetch(self, key, make):
        """覚えていればそれを、無ければ作って覚える。

        make はここで呼ぶ。錠の外で呼ぶのは、
        重い解析の間ほかの求めを止めないため"""
        found = self.get(key)
        if found is not None:
            return found, True
        return self.put(key, make()), False

    def clear(self):
        with self.lock:
            self.items.clear()

    def state(self):
        with self.lock:
            return {"覚えている数": len(self.items),
                    "上限": self.limit,
                    "当たり": self.hits,
                    "外れ": self.misses}


def digest(text):
    """本文から鍵を作る。貼り付けにはファイルが無いので中身で見分ける"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_key(path):
    """ファイルの鍵。書き直せば更新時刻が変わるので、自然に捨てられる。

    中身の要約を取るより速く、書き直しを取りこぼさない。
    ただし更新時刻を保ったまま中身を変える細工には気づけない —
    自分の原稿を相手にする道具なので、そこは見ない"""
    path = Path(path)
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_mtime_ns), stat.st_size)


def key_for(body):
    """画面からの求めに対応する鍵。貼り付けとファイルを見分ける"""
    pasted = (body.get("text") or "").strip()
    if pasted:
        return ("貼り付け", digest(pasted))
    return ("置き場", body.get("name", ""))


# 原稿そのもの。段落分けと章の切り出しを覚える
manuscripts = Memo()

# 原稿から機械的に決まるもの。塊分け・実測・反復語。
# 反復語の探索が重く、形態素解析を使うとさらに重い
structures = Memo()


def forget():
    """全部忘れる。prompts/ や語の取り出し方を変えたときに使う"""
    manuscripts.clear()
    structures.clear()


def state():
    return {"原稿": manuscripts.state(), "解析": structures.state()}

"""センチュリオン — 小説を書く人に伴走する道具と、
文章を保ったまま常套から外れた語りを返す生成AI。

    from centurion import Centurion
    print(Centurion().say("青色にまつわる話を聞かせて"))

原稿を読む側(read / ask / check)は torch も transformers も要らない。
そのため、小説を書く側は**呼ばれたときに初めて読み込む**。
ここで generate を素直に import すると、原稿を読むだけの人にも
torch を強いることになる — 実際にそうなっていて、
「pip install は要らない」と書いた案内と食い違っていた。
"""

import importlib

# 名前 → それがどのモジュールにあるか。触られたときだけ読み込む
_LAZY = {
    "Centurion": "generate",
    "Reply": "generate",
    "BranchDiverter": "generate",
}

from .prompts import build_fluid          # torch を必要としない

__all__ = ["Centurion", "Reply", "BranchDiverter", "build_fluid"]


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"centurion に {name} は無い")


def __dir__():
    return sorted(__all__)

"""センチュリオン — 文章を保ったまま、常套から少し外れた語りを返す。

    from centurion import Centurion
    print(Centurion().say("青色にまつわる話を聞かせて"))
"""

from .generate import BranchDiverter, Centurion, Reply
from .prompts import build_fluid

__all__ = ["Centurion", "Reply", "BranchDiverter", "build_fluid"]

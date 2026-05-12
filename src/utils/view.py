"""Interactive PNG viewer — zoomable/pannable matplotlib window.

Usage:
    python -m src.utils.view figures/gruvae/run_001/generated_S.png
    python -m src.utils.view figures/*.png
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def view(path: str) -> None:
    img = mpimg.imread(path)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.manager.set_window_title(path)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.view <file.png> [file2.png ...]")
        sys.exit(1)
    for p in sys.argv[1:]:
        view(p)

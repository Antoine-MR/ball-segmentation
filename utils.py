from pathlib import Path
from IPython.display import display, Markdown
class DisplayPath(Path):
    def display(self):
        display(Markdown(f"[{self}]({self})") if self.exists() else str(self))
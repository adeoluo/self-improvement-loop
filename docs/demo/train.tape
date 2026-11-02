# Records docs/demo/train.gif from a real run: `vhs docs/demo/train.tape` from the repo root.
Output docs/demo/train.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `python -m self_improvement_loop train --iterations 10` Enter
Sleep 22s

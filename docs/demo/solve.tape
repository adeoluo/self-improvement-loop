# Records docs/demo/solve.gif from a real run: `vhs docs/demo/solve.tape` from the repo root.
Output docs/demo/solve.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 420
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m self_improvement_loop solve 11 65 --budget 6` Enter
Sleep 20s
Type `python -m self_improvement_loop evaluate` Enter
Sleep 22s

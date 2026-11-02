# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 480
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m self_improvement_loop serve --port 18326 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 22s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18326/solve -d '{"start": 7, "target": 50, "budget": 6}' | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18326/history | jq -c '.iterations[] | {iteration, test_accuracy}'` Enter
Sleep 3.5s
Hide
Type `kill %1` Enter
Sleep 1s

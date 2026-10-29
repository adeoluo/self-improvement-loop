# Case Study - What the ablations show

This walk-through reads the loop through its ablations, which is where the design justifies
itself.

## 1. The untrained policy is genuinely uniform
With zero-initialized weights the softmax is uniform over in-bounds actions, so greedy decoding
solves only 0.050 of held-out puzzles, essentially the puzzles already near their target. Any
later greedy success is therefore attributable to learning, not to a lucky initialization.

## 2. One iteration does most of the work
After the first iteration greedy accuracy jumps to 0.717. The task is close to linearly
separable in the features (progress, admissibility, overshoot), so a single round of
behavior cloning on verified traces already recovers a strong strategy. This is the expected
shape for self-training on a well-featured task: the first batch of verified traces is the most
informative.

## 3. The model comes to teach itself
Early on the policy solves few puzzles by sampling, so rationalization supplies 137 of the
first iteration's traces. As the policy improves, sampling solves nearly every training puzzle
and hint recoveries fall to 13. The share of self-generated supervision rises over time, which
is the defining behavior of a self-improvement loop.

## 4. The verifier is the reason it works
Removing the verifier and cloning every rollout collapses the result: accuracy never exceeds
0.55 and swings between 0.017 and 0.55 across iterations. The lesson is that self-training is
not "train on more of your own output"; it is "train on your own output that a verifier
accepts." The filter is the mechanism.

## 5. Rationalization is a cold-start tool, not free accuracy
Rationalization makes the first iteration stronger (0.717 versus 0.675) and guarantees coverage
of every solvable puzzle immediately. On this task its oracle-shortest hints bias slightly
toward multiply-heavy paths, which marginally lowers the final greedy asymptote. That trade is
reported rather than hidden, and it is the honest shape of the technique: hints help most when
the model is weak and matter less once it can solve problems on its own.

## Takeaway
The absolute numbers belong to a toy task, but the three findings generalize: verifier
filtering is load-bearing, rationalization accelerates the cold start, and the proportion of
self-generated supervision should rise as the model improves.

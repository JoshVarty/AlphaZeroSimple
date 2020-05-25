# AlphaZeroSimple

Blog: http://joshvarty.github.io/AlphaZero/

Video: https://www.youtube.com/watch?v=62nq4Zsn8vc

The absolute most basic example of AlphaZero and Monte Carlo Tree Search I could come up with. This implementation plays a game called Connect2:

![](http://joshvarty.github.io/AlphaZero/connect2.gif)

In Connect2 players alternate between playing pieces on a tiny 1x4 board with the goal of placing two of their pieces side-by-side. It’s comically easy to win as the first player, but Connect2 still has the interesting property that either player technically has the chance to win, lose or draw.


## Dependencies
 - PyTorch
 - kaggle_environments (Optional: Allows you to play against the model)
 
 
 ## Based On:
  - [MuZero General](https://github.com/werner-duvaud/muzero-general/)
  - [Alpha Zero General](https://github.com/suragnair/alpha-zero-general)

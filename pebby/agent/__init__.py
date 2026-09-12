"""The LS20 policy: network, oracle-guided data, GPU training and rollout scoring.

The pipeline is behaviour cloning from an exact planner. :mod:`data` runs the
real game with an epsilon-greedy oracle and labels every visited state with the
oracle's optimal action -- DAgger with a perfect expert, which is what teaches
the policy to recover once it has drifted off the optimal path. :mod:`train`
fits :class:`model.Ls20Policy` on those shards, and :mod:`evaluate` rolls the
trained policy out in the same game and reports completion, which is the only
number that counts.

Nothing is imported here on purpose. :mod:`data` needs the game but not torch,
and :mod:`model` needs torch but not the game, so importing the package must not
drag in either.
"""

VERSION = "0.1.0"

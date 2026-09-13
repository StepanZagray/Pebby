# Pebby domain language

Pebby is a learned controller specialized to the known LS20 rules. The vendored
engine defines the game. This context names the contracts used when discussing
training and evaluation; checkpoint-specific results belong in model status and
source-bound research evidence.

| Term | Meaning and boundary |
|---|---|
| Public history | The observations and actions available to the controller. A bounded frame window is distinct from persistent memory across windows or life resets. |
| Root | One decision point together with its causal public history. Multiple roots from one level are correlated. |
| Branch | One candidate action's successor from a root. Counterfactual branches are teacher supervision, not extra actions executed by the policy. |
| Teacher | Exact game planning used to create labels or diagnostic controls. Teacher success is not learned-controller success. |
| Pretrained controller | A model with prior learned capabilities used to select actions; its state adapter, memory and tool access are part of the evaluated system. |
| Curriculum designer | Proposes training situations or intermediate objectives. Proposal validity and teacher-label correctness are distinct responsibilities. |
| Tool-assisted controller | Chooses or executes external procedures such as exact search. System success does not by itself establish that its neural component learned the procedure. |
| Action horizon | Number of future action transitions explicitly composed during prediction or planning. |
| Refinement depth | Repeated computation on a representation. More refinement does not by itself extend the action horizon. |
| Remaining-route value | Predicted cost or distance to eventual completion. A one-action predictor can receive long-horizon supervision through this target. |
| Goal-directed competence | Choosing actions and completing episodes according to the actual goal; sensitive to valid changes in goal placement or identity. A readable goal representation alone does not establish this competence. |
| Learner-state teaching | Execute the current learner on training levels, then label states it actually visits with an independent teacher. Repeating collection after policy updates is distinct from repeatedly fitting a fixed cache. |
| Compositional competence | Reusing learned navigation and mechanics on combinations, orders, or layouts excluded from the fitting examples. |
| Belief memory | A learned or explicit record of previously observed information needed when the current observation window is insufficient. |
| Sequential completion | One game session progresses through all seven shipped levels under the declared life and action budgets. |
| Isolated completion | One level is evaluated from a fresh start. Isolated wins cannot be added together and reported as sequential completion. |
| Development panel | Evaluation examples exposed during diagnosis, checkpoint selection, or design iteration. |
| Confirmation panel | Examples excluded from training and development decisions, opened only after a candidate and acceptance rule are fixed. |

The current effort is indexed by [Find the training path to all seven LS20 levels](.scratch/ls20-seven-levels/map.md).
The durable game and teacher boundaries are in [Game and proof](docs/game-and-proof.md).

# Pebby domain language

Pebby's current controller is multi-game architecture v2, Run B: a recurrent
imitation policy trained on generated games from 24 known families. The vendored
engines define each game's rules. Checkpoint-specific claims belong in
[model status](MODEL_STATUS.md).

| Term | Meaning and boundary |
|---|---|
| Family | One known game rule set with its own native engine, curriculum and teacher. |
| Whole game | An ordered sequence of all the family's levels, retaining native context and recurrent history. |
| Public history | Frames, previous actions and observed outcomes available to the controller; hidden engine state is excluded. |
| Teacher | Exact or bounded native planning used to label generated states; teacher success is not learned-policy success. |
| Certified route | A teacher witness replayed successfully in the native engine under the recorded context. |
| Recovery route | A teacher-labelled continuation after a learner or random perturbation; the provenance distinguishes those perturbations. |
| Canonical input | A stored observation/action variant inverted to its original controls, geometry and palette before learning. |
| Click region | An engine-verified set of equivalent click targets; bounded one-step equivalence is not arbitrary future equivalence. |
| Auxiliary dynamics | Action-conditioned frame/event prediction trained alongside the policy; currently unused for planning. |
| Gameplay selection | Choosing a checkpoint using actual generated-game rollouts, rather than offline prediction accuracy alone. |
| Frozen checkpoint | One immutable set of weights; metrics from different epochs cannot be combined as its result. |
| Generated validation | Generated examples held apart from fitting but exposed to model selection; not an untouched confirmation benchmark. |
| Official training-family evaluation | Frozen-policy play on official levels of the 24 known families, separate from generated validation. |
| Held-out phase | Frozen-policy play on m0r0 after the training-family phase, with no intervening tuning. |

Generator acceptance establishes bounded data and teacher checks, not model
mastery or unlimited puzzle novelty. See [generator caveats](docs/generator-acceptance-caveats.md)
and [training operations](docs/multigame-training-operations.md).

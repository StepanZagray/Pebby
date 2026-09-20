# Whole-architecture options, including pretrained models

Status: **DONE_WITH_CONCERNS**. Research date 2026-09-13; machine timezone verified Europe/Dublin (IST). Five primary papers, two query rounds. Read-only research; no inference, training, downloads, implementation, or canonical edits. All experimental protocols below are proposals.

## Answer

Pretrained models merit explicit comparison, but the useful import must be named: visual representations, interpreting rules, proposing subgoals, writing procedures, or producing actions. These are different abilities. A pretrained model with a reliable state adapter and search tools is a different system from one choosing actions directly from pixels.

Compare three deployment routes: a pretrained controller; a small spatial/recurrent controller trained with verified teaching; and a hybrid with declared planning tools. Separately compare a pretrained teacher that proposes curricula or processes for the small controller. The current model's failure does not establish that the second route is exhausted. Neither pretrained parameter counts nor Minecraft demonstrations establish LS20 competence.

The lowest-commitment next step is a tiny, training-free qualification of input fidelity and action reasoning, followed by a separately specified teaching experiment. No checkpoint should be replaced before closed-loop evidence exists.

## Findings

All paper findings use **owning-primary** sources, retrieved 2026-09-13. Support is established within each paper's population; LS20 implications are explicitly extrapolations.

**1. Behavior pretraining can import usable action patterns — high confidence, limited domain transfer.** VPT trains an inverse-dynamics model to label online Minecraft video, then learns a behavioral prior. Its approximately 70,000 hours of relevant video support nontrivial zero-shot Minecraft behavior; further imitation/RL fine-tuning yields harder skills. [Baker et al., VPT, §1 and §3–4, pp. 2–7](https://papers.neurips.cc/paper_files/paper/2022/file/9c7008aff45b5d8f0973b23e1a22ada0-Paper-Conference.pdf), NeurIPS 2022. **Limit:** “zero-shot” concerns tasks within a heavily pretrained game and native action interface. It does not show transfer from Minecraft weights to indexed-color LS20. Its data-labeling machinery solves a problem Pebby largely avoids through exact generated-game labels.

**2. Web semantics can improve a learned action policy — high confidence, not evidence for an untouched VLM actor.** RT-2 co-fine-tunes pretrained vision-language models on robot trajectories and web tasks, representing actions as tokens. Its ablations separate size and training strategy. Crucially, the authors say web experience improves semantic/visual generalization but does not add physical skills absent from robot training; inference cost is a limitation. [Brohan et al., RT-2, §3.2–3.3, §4.3, §5](https://arxiv.org/html/2307.15818v1), 2023-07-28. **LS20 implication:** interpreting a rule or recognizing a symbol may transfer; exact movement, collision and attribute-update behavior still needs qualification. A frozen zero-shot failure would reject that configuration, not every action-fine-tuned pretrained architecture.

**3. Pretrained representations can help game agents, with substantial additional training — high confidence, qualified transfer.** SIMA combines pretrained visual components with trained transformers, temporal memory and a keyboard/mouse policy. Its tasks predominantly take under roughly ten seconds. Its no-pretraining comparison supports benefits in the tested suite, but also changes encoder architecture; some held-out games appeared in visual-encoder tuning, and comparison agents mostly use one training seed. [SIMA Team, §3.3, pp. 10–11; §4.2, pp. 16–19](https://arxiv.org/pdf/2404.10179), v3, 2024-10-11. **Limit:** these are instructed skills, not seven-stage autonomous puzzle completion. LS20's exact small symbols and bookkeeping are a distribution shift, so a large visual backbone may add cost without improving an already-correct goal decoder.

**4. An LLM can organize reusable procedures — high confidence for the complete tool-using system.** Voyager uses GPT-4 proposals, structured inventory/world information, executable skill code and feedback. Its supplied primitives include `pathfinder.goto`. The paper explicitly avoids direct comparison with pixel-input, primitive-action methods because its Mineflayer interface supplies higher-level control. The reported version lacks visual perception. [Wang et al., Voyager, §2, §3.2, §3.5, Appendix A.4.1](https://arxiv.org/html/2305.16291v2), 2023-10-19. **LS20 implication:** import task decomposition, program synthesis and reuse; do not attribute a pathfinder's navigation to the language model. Stored programs and text memory are external adaptation, not neural weight updates.

**5. The right competencies and teacher distribution matter — high confidence within BabyAI.** BabyAI's controlled instruction-following gridworld experiments show that some base-task pretraining improves later imitation, while another tested base-task choice does not. Demonstrations from an RL-trained network can also be easier to imitate than its scripted bot's demonstrations. [Chevalier-Boisvert et al., §4.2–4.3, Tables 3–5, pp. 8–9](https://arxiv.org/pdf/1810.08272v4), v4, 2019-12-19. **Limit:** this is domain-specific learned pretraining, not internet-pretrained LLM transfer. It supports testing learnable teaching distributions, without importing its sample requirements into LS20.

### Architecture comparison — proposed LS20 interpretation

| Option | Ability plausibly imported or built | Required interface; decisive confound |
|---|---|---|
| Pretrained VLM direct actor | Visual semantics and rule-conditioned action reasoning | Faithful image/history and constrained action output; perception errors can masquerade as planning failures |
| Pretrained LLM, no tools | Rule interpretation and explicit manipulation of structured descriptions | Public-derived grid/HUD/history; gains over pixels partly reflect supplied representation |
| Pretrained LLM with tools | Decomposition and selecting/verifying procedures | Declared transition/search tools and memory; measure complete system and tool contribution |
| Pretrained teacher, small deployed policy | Curriculum proposals, counterexamples, subgoal/process specifications | Engine verifies examples and labels; student training remains necessary and currently unexecuted |
| From-scratch spatial/recurrent policy | Domain-specific routing, action dynamics and memory | Generated trajectories, controlled curriculum and optimization; absence of web pretraining does not imply impossibility |

For known-rule states where Pebby's exact search completes, an LLM cannot improve the correctness of an already-exact optimal label. It can propose which valid cases to teach, easier demonstrations, meaningful intermediate targets, or missing mechanics. Every proposal must become an engine-checked artifact; verbal plausibility is insufficient. Search caps remain incomplete results. [Local planner contract, lines 26–46](/home/stepan/Projects/code/Pebby/docs/game-and-proof.md:26). Distilling validated procedures or labels transfers only what the student's data/objectives capture, not the teacher's entire reasoning ability.

### Input and accounting contract

LS20 exposes a 64×64 array of color indices 0–15; its board is a 12×12 lattice, and level 7 has fog. [Rules, lines 8–24](/home/stepan/Projects/code/Pebby/docs/game-and-proof.md:8). Preserve the original array/hash and versioned palette. For VLMs, use deterministic palette-to-RGB rendering with lossless storage and declared nearest-neighbor enlargement. For text, use coordinates and a lossless indexed grid or component encoding with a round-trip reconstruction check. A semantic object/HUD table is a separate, potentially lossy adapter: preserve uncertainty and mark unseen cells unknown. Never fill it from hidden engine objects.

Supply the same rule specification, available action IDs and allowed observation history; separately declare episode memory. Count every executed primitive move and RESET, including those inside tools/macros. Record prompt/completion tokens, image processing settings, tool expansions, retry count, wall latency and peak memory. Equal environment actions alone do not equal inference cost; token counts are model-dependent. A larger text history versus H8 changes information, not only model quality. [Public decision fields and available actions, lines 38–45 and 84–87](/home/stepan/Projects/code/Pebby/pebby/agent/competition.py:38).

### Very small qualification protocol

First, verify serialization, coordinate orientation, action schema, unknown-state handling and reset boundaries without model inference. Then, when an appropriate runtime is separately available, propose **16 first-action fixtures and four closed-loop rooms capped at 12 actions each per configuration**. Include cardinal adjacent goals, opposite-goal pairs, one obstacle, one attribute change and budget-sensitive choices; reserve fresh generated fixtures, and verify teacher routes in the engine.

Test image and structured inputs with the same model where possible; test no-tools versus declared-tools with the same state. Include tool-only and retained-policy baselines. Require every trivial one-move case and both members of its opposite pair before expanding. A tool-only win means the combined system has not demonstrated an additional LLM benefit. Re-render valid transformed scenes with corresponding action/goal mappings to expose fixed-action shortcuts. This small panel admits further study; it cannot estimate seven-level completion.

For teacher qualification, request eight curriculum/process proposals against known failure categories; measure validity, new coverage and verification cost against a deterministic primitive generator. Later student training must compare identical student architecture, example count and compute using baseline versus proposed teaching. Until that controlled fit happens, report only proposal quality, not learning gains. Architecture comparisons with different adapters are system comparisons; isolating pretraining itself would require additional matched initialization/training controls.

## Contradictions and gaps

None of these papers establishes pretrained LS20 performance, suitable model size, local latency, or an affordable training budget. Existing local Qwen evidence concerns five non-goal protocol fixtures and a two-action smoke with zero completed levels, not a successful comparator. [Historical qualification](/home/stepan/Research/ml/tofy/insights/qwen-protocol-qualified-ae4bc401.md:9). Its other-project preferences impose no constraint here. Current Pebby still lacks persistent learned memory, learned voluntary RESET and multistep learned planning; those gaps cannot be credited as implemented by choosing another model family. [Current status](/home/stepan/Projects/code/Pebby/MODEL_STATUS.md:13).

Adversarial self-review: paper mechanisms **verified**, their LS20 benefit **unverified**; “larger pretrained model solves the problem” **unsupported**. No agent-independent review was performed by this worker. Primary handles independent verification and interface integration.

Search record: round 1 found VPT/Voyager/RT-2/SIMA; round 2 located primary full-text alternatives. BabyAI was retrieved from prior research, then reopened. RT-2 PDF endpoints and Voyager PDF/OpenReview access failed; the exact arXiv HTML versions above resolved. All five final targets and relevant locators were reopened. No third query round was needed; no processes remain running.

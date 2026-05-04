
Reward-Constrained Visual Memory Selection as a Normative Model of Human Semantization
Bottom line
After reviewing the current neuroscience and NeuroAI literature, the strongest project is not an RL-enhanced image reconstructor and not another “which CLIP layer matches the brain?” study. The more novel and more neuroscientifically meaningful project is to treat RL as a computational theory of memory selection: train an agent with a fixed memory budget to decide what visual information to keep, compress, or discard, then test whether its learned retention policy predicts how human fMRI representations change across repeated viewings in the Natural Scenes Dataset. This keeps deep learning and computer vision central, keeps RL central, and shifts the scientific question from “how to make prettier reconstructions” to “what objective function best explains what the brain remembers.” 

The core novelty gap is real. Recent work already maps fMRI to CLIP space for reconstruction, already aligns low- and high-level visual areas to different CLIP layers, already localizes dense semantic features in images to cortical subregions, and already shows that mental-imagery generalization is often decoupled from seen-image reconstruction quality. That means a project centered on CLIP-layer matching or RL-guided diffusion guidance would likely say more about the reconstruction stack than about human memory. A project centered on task-optimal memory compression is much better positioned to discover something genuinely new about the brain. 

What the literature already establishes
The Natural Scenes Dataset gives you unusually good leverage for this question. It contains whole-brain, high-resolution 7T fMRI from 8 subjects collected over 30–40 sessions while they viewed thousands of natural scenes in a continuous recognition task, and NSD-based work explicitly notes that images repeat up to three times, with lags ranging from seconds to many months. That repeat structure is exactly what a memory-selection project needs. 

On the ML side, fMRI decoding has already moved far beyond proof of concept. Earlier work showed that hierarchical visual features can be predicted from fMRI and that lower versus higher cortical areas preferentially predict lower versus higher model features. More recent work such as MindEye2 reconstructs viewed images by mapping shared-subject fMRI latents into CLIP image space and then into pixels, while BrainMCLIP explicitly aligns low- and high-level visual regions to intermediate and final CLIP layers. Separately, BrainSAIL shows that dense features from pretrained vision models can localize the image subregions driving cortical selectivity, including both semantic concepts and lower-level properties such as depth, luminance, and saturation. 

At the same time, several findings show why reconstruction itself is not the best scientific endpoint. A recent benchmark on NSD-Imagery found that performance on mental imagery is largely decoupled from performance on seen-image reconstruction, and that simple linear decoders and multimodal feature decoding generalize better than more complex reconstruction architectures. Related recent critique has argued that text-guided reconstruction methods can generalize poorly across datasets, which again suggests that the reconstruction image is often a fragile proxy for the underlying neural computation. 

On the neuroscience side, there is already serious evidence that memories become more semantic over time, but the mechanism is still unresolved. A recent fMRI study reported that time-dependent memory transformation is semantic rather than perceptual, with increasing semantically transformed representations in prefrontal and parietal cortex and declining item-specific representations in anterior hippocampus. A complementary computational literature frames memory as capacity-limited communication or rate-distortion optimization, where the system should preferentially preserve information that will matter later. Related work on replay and prioritized memory access in RL argues that which memories are accessed or strengthened should depend on future utility, not only past frequency. A recent NSD-based spacing study further showed that benefits of spaced learning track increased stimulus-specific representational similarity in vmPFC, consistent with a re-encoding account of memory change across repetitions. 

Those strands fit together, but they have not yet been joined in the way your project can join them. The missing step is a model that learns, under realistic constraints, which visual information is worth retaining and then asks whether that learned policy predicts cortical semantization and repetition effects better than raw CLIP alignment or generic compression. That is the scientific opening. 

Why the earlier proposal is not the right novelty target
The earlier proposal had a real conceptual issue: it mostly studied how the brain maps onto a preexisting representational scaffold, especially CLIP, rather than why the brain should preserve some features and forget others. That is not useless, but it is weaker as a neuroscience contribution. MindEye2 already maps fMRI to CLIP image space, so probing CLIP alignment inside that pipeline risks a built-in bias even if you probe an earlier latent. BrainMCLIP goes even further by explicitly aligning functionally distinct visual areas with intermediate and final CLIP layers. By itself, that makes “human forgetting follows CLIP’s hierarchy” much less novel than it first appears. 

There is a second novelty problem. BrainSAIL already demonstrates that pretrained dense vision features can be used with voxelwise encoding to identify the image subregions and feature types that drive responses in higher visual cortex, including semantic properties and lower-level visual variables. So a project whose main result is “this voxel likes these features” is also not enough unless it adds a new mechanistic principle. 

There is also a scientific-motivation problem with RL-guided reconstruction. If RL is only choosing guidance scales or denoising parameters to improve reconstruction fidelity, then RL is optimizing the generator, not modeling the brain. That can improve images, but it does not directly reveal why the human memory trace becomes gist-like, why spacing helps, or why semantic errors rise with delay. In the current literature, the biggest open questions are about representational transformation, compression, replay, and utility—not about whether another control policy can make diffusion outputs look more plausible. 

Finally, the “can we figure out what each neuron does?” intuition is scientifically good but technically mismatched to NSD. NSD is fMRI at 1.8 mm isotropic resolution, so each measurement reflects large voxel-level populations, not single neurons. With this dataset, the right target is not a neuron-by-neuron role map, but a voxel- or subspace-level feature-selection theory that explains what classes of information are preserved across repeated experience. 

Recommended project proposal
Project title
Reward-Constrained Visual Memory Selection Predicts Human Cortical Semantization

Proposal abstract
I would propose a project in which a reinforcement learning agent sees an image represented by a rich bank of deep visual features and must decide, under a hard memory budget, which information to store for future use. The agent is trained not to reconstruct the image, but to maximize success on later recognition and attribute-retrieval tasks after delays and repetitions that mirror the structure of NSD. I would then test whether the agent’s learned retention policy predicts which feature families become more or less decodable from human fMRI across repeated exposures, thereby asking whether semanticization in cortex reflects task-optimal memory compression rather than mere similarity to a specific vision model. 

Central scientific question
Does the human visual-memory system preserve information according to a reward- or utility-sensitive compression policy, such that repeated experience preferentially stabilizes those image features that are most useful for future recognition and generalization? 

Main hypotheses
The first hypothesis is that later-exposure cortical patterns will be better predicted by agent-compressed features than by raw feature banks if the agent’s objective captures the brain’s memory-selection principle. The second hypothesis is that different cortical systems will align with different retained feature families—for example, early visual regions with geometry, color, texture, and spatial layout, and higher-order regions with object, scene, and semantic information. The third hypothesis is that spacing and repetition will selectively strengthen retained high-utility features, especially in cortical regions implicated in remote or generalized memory. 

Why this is novel
The novelty is not “use deep features on fMRI” and not “use RL somewhere in the pipeline.” Those things already exist. The novelty is that you would use RL to learn a normative forgetting policy and then directly ask whether human cortical semantization across repeated naturalistic viewing matches that learned policy better than competing explanations such as raw model features, generic low-rank compression, saliency, or CLIP-only semantics. I did not find evidence in the current literature that this exact combination—multi-family deep visual features + RL-based memory selection + exposure-wise NSD fMRI comparison—has already been done. The closest literatures cover only pieces of it. 

Exact technical plan
Build a feature bank that is broader than CLIP
The feature representation should be intentionally plural, because relying on a single model family recreates the circularity you were already worried about. I would use at least five frozen feature families for every NSD image.

The first family should be self-supervised global and patch features from DINOv2, because DINOv2 was built as a robust vision feature extractor and its features transfer well without task-specific finetuning. The second should be vision-language semantic features from SigLIP or a comparable image-text encoder, because language-supervised models are strong accounts of higher visual cortex and give you explicit semantic structure. The third should be object-level slots produced by Segment Anything masks, with each mask pooled into a feature vector using DINOv2 and the vision-language encoder. The fourth should be geometry/layout features from a depth foundation model such as Depth Anything. The fifth should be explicit low-level features such as color histograms, texture statistics, and spatial-frequency or Gabor-style summaries, because higher visual cortex still carries systematic low-level tuning biases shaped by the semantic informativeness of natural scenes, and early visual cortex definitely does. 

This matters scientifically. BrainSAIL already showed that dense pretrained features can separate co-occurring concepts in natural images and map them to cortical selectivity. Your extension is to convert those rich features into memory candidates rather than only explanatory regressors. In other words, each object slot, patch cluster, and feature family becomes something the agent can choose to remember or discard. 

Concretely, for each image (i), build feature groups (g=1,\dots,G). Each group should have: [ z_{i,g} \in \mathbb{R}^d, \quad c_g \in \mathbb{R}{+} ] where (z{i,g}) is the embedding for one candidate memory unit and (c_g) is its storage cost. A candidate memory unit can be an object mask, a patch, a global semantic token, a depth summary, or a low-level feature block.

Fit exposure-aware voxelwise encoding models before touching RL
Before training the agent, establish the neuroscience side cleanly. Using NSD, fit voxelwise or ROI-wise encoding models from the frozen feature bank to fMRI responses separately for first, second, and third presentations whenever possible. NSD makes this feasible, and open tooling already exists to access trial betas, behavioral outputs, images, annotations, and ROI definitions. Large-scale benchmarking codebases also already exist for comparing many model families against NSD. 

This stage has two purposes. First, it gives you a baseline map of which feature families explain which regions even without memory modeling. Second, it tells you where repetition effects already appear in representational terms: which feature families gain predictive power, which lose it, and which regions become more or less stable across viewings. That baseline is critical, because the RL model should be judged by whether it explains changes across repetitions, not just static encoding. Recent work on spaced learning in NSD makes this especially timely, since representational similarity changes across repetitions are already known to matter behaviorally. 

I would use regularized linear models first, not elaborate nonlinear encoders. There is strong precedent for linear voxelwise decoding and encoding, and recent NSD-Imagery results are actually favorable to simpler linear mappings when the desired interpretation is mnemonic rather than purely reconstructive. 

Define the RL problem as memory selection, not image generation
The RL environment should be a delayed-query memory task rather than a decoder-control problem.

At encoding time, the agent receives an image feature set: [ s^{\text{enc}}i = {(z{i,g}, c_g, f_g)}{g=1}^G ] where (f_g) encodes the family label, such as object, patch, semantic, geometry, or low-level. The agent must choose a retention action: [ a{i,g} \in {0,1} ] or, if you want variable precision, [ a_{i,g} \in [0,1] ] subject to a memory budget: [ \sum_g c_g a_{i,g} \le B. ]

The resulting memory trace is [ m_i = {a_{i,g} z_{i,g}}_{g=1}^G. ]

Later, after a delay drawn from the empirical repetition-lag structure of NSD, the environment asks one or more questions about the image. These questions should be sampled from a task distribution designed to separate semantic from perceptual memory. The reward should reflect future usefulness, not reconstruction quality. That is the core conceptual change. 

Use three classes of delayed probe so the policy has something meaningful to optimize
The first delayed probe should be recognition under confusable lures. Present the agent with the target plus distractors or pairs selected to be close either in semantic space or in perceptual space. Semantic lures can be nearest neighbors in SigLIP space or caption space; perceptual lures can be nearest neighbors in low-level texture/color/depth/layout space. This mirrors the literature distinguishing semantic from perceptual transformation more directly than plain image reconstruction does. 

The second delayed probe should be attribute retrieval. Ask whether the scene contained a person, whether it was indoors or outdoors, whether an object was red, whether a table was left of a person, what the rough depth ordering was, and so on. Some questions should be semantic and some perceptual. You already have many of the ingredients because NSD images come with COCO-derived annotation structure, and dense models can add object masks, depth, and geometry. 

The third delayed probe should be repeat-aware re-learning. When the same image reappears, the agent should have the choice not just to encode from scratch but to strengthen, replace, or compress the existing trace. That lets you model repetition and spacing explicitly. Given the recent NSD result on spacing and vmPFC re-encoding, this is a scientifically grounded design choice rather than an arbitrary RL embellishment. 

A concrete reward function
A simple but scientifically meaningful reward is: [ r = \alpha r_{\text{recog}} + \beta r_{\text{sem}} + \gamma r_{\text{perc}} - \lambda \sum_g c_g a_{i,g}. ]

Here, (r_{\text{recog}}) is correct/incorrect recognition reward, (r_{\text{sem}}) is semantic attribute accuracy, and (r_{\text{perc}}) is perceptual attribute accuracy. The budget penalty enforces compression. If you want a harder constraint rather than a soft one, use a Lagrangian PPO or simply mask actions once the budget is exhausted. This kind of utility-aware retention is aligned with rational-memory and prioritized-memory-access theories rather than reconstruction pipelines. 

What policy architecture to use
Use a set encoder plus top-(K) selector. Each candidate memory unit is independent in identity but not in usefulness, so the policy needs context over the whole set. A practical design is:

Encode each candidate unit with an MLP: [ h_g = \text{MLP}([z_{i,g}; f_g; c_g; \ell; r]) ] where (\ell) is lag information and (r) is repeat count.

Feed all (h_g) through a small Transformer or Set Transformer so the policy can reason about redundancy across units.

Output a scalar logit (u_g) per unit.

Sample or select the top-(K) units under the budget using Gumbel-top-(K), straight-through sampling, or policy-gradient over Bernoulli actions.

Store the selected units and use a frozen or lightly trained task head to answer delayed probes from the stored trace.

This is attractive because it makes the learned policy interpretable. You can inspect which units are selected, what feature families dominate, and how the distribution changes with lag and repetition. That interpretability is a major advantage over using RL only to tune denoising hyperparameters. 

How to train the RL system in practice
The easiest stable path is a staged curriculum.

Start with a contextual-bandit version: one image, one delayed query, no repeated encounter. This tests whether the policy can learn basic selection under a budget. Then add mixed semantic/perceptual probes. Then add repeat-aware state updates, where the same memory can be edited when the image reappears. Finally, expose the policy to an empirical lag distribution approximating the NSD repetition schedule. A lag-conditioned policy is important because a useful short-lag memory need not be a useful long-lag memory. 

I would pretrain the probe heads separately so the policy is not simultaneously trying to invent the memory task and the memory strategy. Use frozen image-feature extractors and relatively simple task heads. Then fine-tune only the policy, and optionally the final probe layers. The reason is not just engineering convenience; it also keeps the interpretive target clean. If everything is trainable, it becomes much harder to say whether the policy discovered a meaningful forgetting rule or merely co-adapted with the probe network.

How to compare the learned policy to brain data
This is the most important part.

After training, the agent yields a retention weight profile for each image: [ \rho_i = [\rho_{i,1}, \dots, \rho_{i,G}]. ]

Use this to build a compressed image representation: [ \phi^{\text{RL}}(i) = \text{Compress}({z_{i,g}}, \rho_i). ]

Then perform three brain comparisons.

The first comparison is encoding performance. Fit voxelwise models using the compressed representation and compare held-out (R^2) or correlation against models built from raw features, random-pruned features, PCA-compressed features, and autoencoder-compressed features matched for budget. The critical test is whether (\phi^{\text{RL}}) predicts later-exposure activity better than those alternatives. 

The second comparison is representational similarity. Compute image-by-image RDMs for the compressed representations and compare them to exposure-specific brain RDMs in each ROI. If the later-exposure RDMs move toward the RL-compressed geometry more than toward raw perceptual geometry, that is direct evidence for utility-aware semantization. 

The third comparison is feature-family shift analysis. For each region, quantify how predictive power for each feature family changes from exposure 1 to exposure 3. Then test whether those changes are correlated with the policy’s retention preferences. If the agent strongly preserves object/scene semantics and depth but discards texture, and the human region shows the same direction of shift, that is precisely the kind of discovery you want. 

The most informative baselines
The right baselines are not optional here; they determine whether the result is actually meaningful.

You should compare against raw multi-family features, a matched-budget random selection policy, a saliency-based policy, a CLIP-only or SigLIP-only policy, a rate-distortion or autoencoder compressor without task reward, and a purely supervised selector trained to answer current questions without delayed reward. The key scientific comparison is whether future utility is necessary. If a generic compressor or saliency heuristic explains later-exposure fMRI just as well, then the RL story is weaker. If the RL policy clearly wins, then you have a strong neuroscience result. 

Feasibility report
This project is feasible because almost all expensive components already exist as pretrained or open resources. NSD is public and there is an access package that exposes trial betas, behavioral outputs, images, ROIs, and annotations. MindEye2 has open notebooks and practical instructions, and its repository reports that even full pretraining and fine-tuning on the 40-session setting takes roughly a day on a single GPU. DeepNSD provides a benchmarking pipeline for many vision models against NSD. TRIBE v2 also has open code and weights for multimodal-to-fMRI prediction, though I would treat it as optional rather than central. 

That means your heaviest computations are mostly feature extraction and voxelwise regression, both of which are straightforward compared with training a new diffusion decoder from scratch. The RL environment is also manageable because it operates over compact feature groups, not over pixels or latent denoising trajectories. In practical terms, the compute profile is much closer to “train a small policy network over frozen embeddings” than to “train a state-of-the-art image generator.” 

A realistic implementation sequence would look like this. First, build the feature bank and verify exposure-specific encoding trends. Second, implement the simplest one-shot memory-selection environment and show that the policy learns a stable budgeted retention strategy. Third, add lag conditioning and repeated-view update actions. Fourth, compare policy-compressed features against brain data and run matched-budget baselines. Fifth, do the interpretation work: which regions align with which retained features, and how does that change with repetition or spacing. This is well scoped for a project if you are disciplined about not training a new reconstruction model. 

If you want an especially safe path, make TRIBE v2 and NSD-Imagery optional validation rather than required dependencies. NSD-Imagery would be valuable because simpler linear or multimodal decoders generalize better to internally generated content, which is conceptually relevant to memory. But the project stands on NSD alone. 

What this project can genuinely discover
The strongest possible discovery would be this: later human cortical representations are better explained by a reward-constrained memory-selection policy than by raw visual features, generic compression, or CLIP-layer alignment. That would be a real neuroscience result because it would argue that semantization is not just a descriptive shift from “early” to “late” features, but the consequence of a specific computational objective: preserve information with the highest expected future utility under storage constraints. 

A second discovery would be region-specific utility functions. The same policy framework may reveal that different parts of cortex effectively “value” different information: some regions align with retained geometry and texture, others with object- and scene-level semantics, and others with the re-encoding of retrieved traces across spacing. That would connect the semantization literature to the emerging literature on dense cortical feature localization in natural scenes. 

A third discovery would be a cleaner interpretation of repetition effects. Rather than asking whether repetition pushes the brain toward CLIP’s later layers, you would ask whether repetition strengthens the features that a bounded agent should keep because they matter later. That question is much closer to a theory of memory’s adaptive function, and it integrates with rational-memory and prioritized-replay ideas already present in cognitive neuroscience and RL. 

There is also a meaningful null-result story. If RL-based utility compression does not outperform generic compression or raw features in predicting later-exposure brain data, that still tells you something important: human semantization may be driven more by generic representational bottlenecks, consolidation dynamics, or schema-based reconstruction than by future decision utility. That would still be a solid scientific contribution because it cleanly tests a normative theory rather than only producing reconstructions. 

Open questions and limitations
The biggest limitation is the measurement modality. With NSD, you are modeling voxel populations, not individual neurons, so the project cannot literally answer “what each neuron does.” What it can answer is which feature classes and image substructures behave as if they are retained by a bounded memory system. 

A second limitation is that the learned policy depends on the downstream probe distribution. If you ask only semantic questions, the policy will become overly semantic. If you ask only perceptual questions, it will over-retain details. The fix is not to hide this dependence but to embrace it: report how the learned policy changes as you vary the semantic-versus-perceptual reward mix, and then ask which mixture best matches the brain. That turns an apparent weakness into an empirical test of what the brain is optimizing. 

A third limitation is that there are already many powerful encoding and decoding models, so novelty must come from the scientific question and the comparison design, not from using another foundation model for its own sake. That is why the project should be framed around the normative objective of memory selection, with reconstruction treated as secondary or omitted entirely. 

The overall recommendation is therefore clear: if the goal is a frontier project that actually teaches you something about how human memory works, the best path is to abandon RL-as-generator-control and instead build RL-as-memory-theory. That is where the real novelty and the best neuroscience payoff are. 

Implementation status
This repository now includes an executable Version 2 scaffold with the key methodological sharpenings:
- NSD payload ingestion with strict-mode validation gates,
- sequential, budget-aware memory-selection policy with decoupled-temperature Gumbel sampling and Lagrangian budget pressure,
- delayed utility reward including novelty and schema-congruence terms plus non-monotonic low-level retention contribution,
- expanded multi-family feature taxonomy (semantic, object/part, scene-graph, OCR/face, geometry/depth/normals, low-level/texture/color/frequency/edges, patch, saliency),
- matched-budget baselines (random, saliency-like, generic compression),
- banded-ridge encoding with per-family variance-drop decomposition,
- FR-RSA (feature-reweighted RSA) with ROI-wise family weights,
- family-retention shift versus brain-weight shift correlation analysis,
- behavioral process-model test: retention decisions predicting image-level hit rates,
- reward weight sweep over (alpha, beta, gamma) to estimate implicit objective mixtures,
- feature-quality, retention-utilization, and setup-warning diagnostics emitted into `results.json`.

Detailed implementation plan
See `IMPLEMENTATION_PLAN.md` for the full phase-by-phase blueprint, deliverables, success criteria, and next-step NSD integration path.

Quickstart
1. Create and activate a Python environment (3.10+ recommended).
2. Install dependencies:
   `pip install -e .`
3. Run on NSD-preprocessed payload via Modal (recommended research path):
   `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload.npz --epochs 50 --budget 32 --output-subdir nsd_run_01`
   (Defaults are practical for development: synthetic mode is available out of the box; use NSD flags to run real-data experiments.)
   GRPO is default; you can override:
   `modal run scripts/modal_app.py --data-source nsd --nsd-source npz --nsd-path nsd/nsd_payload.npz --epochs 50 --budget 32 --output-subdir nsd_run_01`
   (Create payload with `python scripts/build_nsd_payload.py ... --output nsd_payload.npz`)
   Or run directly from split array directory:
   `python scripts/run_pipeline.py --data-source nsd --nsd-dir /path/to/nsd_payload_dir --output-dir outputs_nsd`
   Or run from auto-discovered layout root:
   `python scripts/run_pipeline.py --data-source nsd --nsd-layout-root /path/to/nsd_layout_root --output-dir outputs_nsd`
4. Inspect generated artifacts:
   `outputs/results.json`
   (Includes pooled metrics and `subject_results` for subject-wise analysis when available.)
5. Run the GRPO verifier suite:
   `python scripts/verify_grpo.py`
   The verifier runs a compact synthetic experiment and checks:
   - default algorithm routing (`grpo`),
   - GRPO history integrity (finite reward / lagrangian traces with expected lengths),
   - strict retention invariants (binary selection tensor, expected shape),
   - hard budget compliance across every image and exposure,
   - prediction API validity (`predict_hit_rates` shape and [0, 1] range),
   - determinism under fixed seed/config for GRPO,
   - REINFORCE fallback path still operational,
   - unknown algorithm guardrail error behavior.
6. NSD payload schema:
   `docs/NSD_PAYLOAD_FORMAT.md`
7. Modal cloud training guide:
   `docs/MODAL_TRAINING.md`
8. GRPO research-readiness notes and references:
   `docs/GRPO_RESEARCH_READINESS.md`
9. Full training/testing/evaluation runbook:
   `RESEARCH_README.md`
10. NSD acquisition and setup checklist:
   `docs/NSD_GETTING_STARTED.md`

Feature bank construction
- Build unified multi-family unit embeddings from per-family arrays:
  `python scripts/build_feature_bank.py --block semantic:/path/semantic.npy:1.2 --block geometry:/path/geometry.npy:0.9 --output feature_bank.npz`
---
layout: distill
title: "Budget Alignment: Making Models Reason in the User's Language"
description: "We explore a two step multilingual alignment recipe for large language models to keep reasoning and answers in the user language while preserving accuracy."
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: "2026-04-27-budget-alignment.bib"

toc:
  - name: "Introduction"
  - name: "What we try (Method in two steps)"
  - name: "Key contributions"
  - name: "RQ0 — Can small SFT reprogram a reasoning model's reasoning tone?"
  - name: "RQ1 — Does SFT help accuracy, or only language reasoning style?"
  - name: "RQ2 — When RL comes, how does GRPO help with accuracy?"
  - name: "RQ3 — Where should RL start from - Base or SFT?"
  - name: "RQ4 — Can we push the Pareto frontier instead of trading accuracy for language consistency?"
  - name: "RQ5 — Does model merging help?"
  - name: "Discussion - Where performance regresses, and potential solutions"
  - name: "Blog Summary - Practical takeaways"
  - name: "Limitations and threats to validity"
---

*Please read this as a late-stage work in progress shared in a “lab meeting” spirit to help and motivate parallel research.*

## Introduction

You ask a large language model (LLM) a math question in Japanese. It responds politely in Japanese — but behind the scenes, it’s reasoning in English/Chinese. Variables, steps, and mathematical lemmas often silently switch languages during reasoning. This behavior, where models default to English for chain-of-thought (CoT) reasoning, is more than a curiosity. It breaks instruction-following, confuses human overseers, and undermines the purpose of multilingual evaluation.

The goal is clear: we want models to reason about a question in the language they are asked — not just to answer in that language. But this turns out to be harder than it sounds. Forcing models to reason in non-English languages usually leads to a drop in accuracy. Previous work shows that instructing models to reason only in the prompt language via prompting or steering improves coherence and grading alignment <d-cite key="zhong2025language"></d-cite>, but often comes at a steep “accuracy tax.” Even a small amount of multilingual fine-tuning helps, but doesn’t eliminate the trade-off <d-cite key="qi-etal-2025-models"></d-cite>. Further, models not only prefer to reason in English — they reason *more effectively* in English. When researchers force strict in-language reasoning (e.g., in Swahili or Thai), models often lose accuracy compared to when allowed to reason in English. For higher-resource languages like French or German, this trade-off is smaller — models can reason in-language nearly as well as in English. For low-resource languages, strict enforcement harms performance more significantly.

Why do models switch to English in the first place? Much of it traces back to training. Most reasoning data are in English. Fine-tuning even strong multilingual models on English CoT data often leads them to adopt English as their “internal language of logic.” Yong et al. (2025) observe a “quote-and-think” behavior <d-cite key="yong2025crosslingual"></d-cite>, where models copy input phrases in the prompt language, but explain everything in English <d-cite key="kim2025one"></d-cite>. The model understands the question in the non-English language — it just prefers to reason in English.

Our technical goal is simple: **stop the switching without paying an accuracy tax** — ideally, push the Pareto frontier of *(Accuracy, Language-consistency)*.  
And we want this post to serve as a practical guide with lessons learned along the way.  

Code, data, and checkpoints will be linked in the **camera-ready** version of this post to preserve anonymity during review.

---

## What we try (Method in two steps)

🔧 **Base model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, a large reasoning model distilled from R1 through supervised fine-tuning on its reasoning traces, exhibiting an English/Chinese-dominant prior.

**Step 1 — Small SFT to teach in-language reasoning.**  
We fine-tune on **817 curated multilingual reasoning chains** (from LiMO <d-cite key="ye2025limo"></d-cite>). This supervision data contains high-quality reasoning data matching R1 long-form reasoning *style*. No Reinforcement Learning (RL) here — just teach the policy to keep reasoning in the user’s query language.

**Step 2 — Math-only GRPO to push accuracy while retaining reasoning language.**  
We run an RLVR-style GRPO with no KL, higher clip of 0.28 vs −0.2 (DAPO-like <d-cite key="yu2025dapo"></d-cite>), rollout 24, LoRA r = 8, LR = 1e-5, **only on a Math-500 set translated to each language**.  
Intuition: let RL optimize hard cases and verification behaviors, while the high clip reduces catastrophic reasoning style collapse back to English.

We set the verifiable rewards as **1.0 for accuracy, 0.2 for language consistency of reasoning traces, and 0.2 for answer format** <d-cite key="rastogi2025magistral"></d-cite>.

📊 **Evaluation.**

We tried our approach on three different languages: **Japanese (JA) / French (FR) / Spanish (ES)**

And tested on multiple datasets: **MMLU College Math (MMLU Math), AIME25, GPQA, MMLU Pro Medicine (MMLU Med)**

The first two are in-domain: MMLU-Math is similar to the training data in terms of hardness, while AIME25 is harder.  
The other two are out-of-domain: GPQA covers hard science questions, and MMLU Pro Medicine is made up of hard questions in the medical domain.

**Regimes tested:**  
- Base → `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` <d-cite key="deepseekai2025deepseekr1distillqwen7b"></d-cite> 
- SFT on top of Base  
- GRPO-from-Base  
- GRPO-from-SFT

**Metrics:**  
- `pass@k(1,5,10)` where `n = 32` for accuracy
- `Language-consistency %` (both reasoning traces **and** final answers must be in the requested language; script-aware checks)

**How we score language consistency:**  
We check the entire CoT span and the final boxed answer.  
A sample counts as `Following = 1` only if both passages are in the requested language (script tokens, numerals, and markers allowed); otherwise `0`.  
We report the % across the set.

---

## 🔑 Key contributions

1. **Small SFT reprograms inner monologue.**  
   With only **817 chains**, language consistency rises near the ceiling in French/Spanish across datasets and substantially in Japanese (Fig. RQ0).

2. **Two-step recipe Pareto-improves.**  
   SFT secures language consistency; **GRPO-SFT recovers/boosts accuracy on tough sets** (AIME/GPQA) without reverting to English (Figs. RQ1–RQ4).

3. **Diagnose regressions and actionable fixes.**  
   Regressions stem from:
   - Japanese tokenization/numeric friction,  
   - Spanish cue misalignment,  
   - medicine reward/style mismatch.  
   Tokenizer-aware normalization, small Japanese/Spanish SFT top-ups, and multi-objective GRPO (with optional model merging) could recover accuracy without sacrificing in-language reasoning.

4. **TL; DR.** You can briefly see our main results from the two figures below:  
   Starting from an EN/ZH-dominant reasoning prior, small multilingual SFT is the most cost-effective way to “steer” in-language chains of reasoning. Adding math-only GRPO then recovers or improves accuracy on hard sets like AIME and GPQA while mostly preserving SFT’s language consistency discipline — pushing the Accuracy × Following frontier in many language–dataset pairs. The two pain points, Japanese (tokenization/numeric friction) and medicine (reward/style mismatch), are expected from the base prior and training signal, and both have potential straightforward fixes with light domain augmentation. And surprisingly, model merging can be very useful and effective.

**Figure 1.a) Performance comparison overall across methods**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/1a.png" class="img-fluid" %}

**Figure 1.b) Overall language consistency rate comparison across methods**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/1b.png" class="img-fluid" %}

---

## RQ0 — Can small SFT reprogram a reasoning model’s "reasoning tone"?

Models often output the final answer in the same language as the user query. We want the **reasoning process** to match the prompt (user) language, too.

**Results.**  
SFT drives the language consistency rate close to the ceiling (**~99–100%**) in French/Spanish and raises Japanese substantially (**high-80s/90s**).  
The language consistency rates averaged across all datasets are shown in Fig. RQ0: bars labeled Japanese/French/Spanish.

**Interpretation.**  
A few hundred **high-quality chains** are enough to overwrite the English/Chinese inner-monologue priority to other languages. Japanese remains stubborn — see RQ5.

> Recall that instruction-following does not only mean the answer in the prompt language, but it should also ensure that the language of the reasoning traces is the same as the user's preference to enhance their trustworthiness. SFT alone solves most of the language mismatch with limited accuracy improvements, which are yet lower than the accuracy of reasoning in English (i.e., the gray dashes in Figure 1.a above) in most cases. We provide more details in the next section.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r0.png" class="img-fluid" %}

---

## RQ1 — Does SFT help accuracy, or only language reasoning *style*?

We have shown that **SFT significantly improves language consistency rates**, but how about the accuracy?

**Design.**  
Compare the accuracy **Base vs SFT** on `pass@k` per dataset–language  
(Fig. RQ1: Δ pass@10 = SFT − Base).

**Findings.**

- **MMLU-Math:** substantial improvements when train and test are in the same domain  
  - *French:* ~76 → **98**  
  - *Spanish:* ~80 → **99**  
  - *Japanese:* ~68 → **88**

- **AIME:** mixed. Although AIME contains math problems, it is way more difficult than LiMO, making it less likely to be considered as in-domain. As a result, SFT trades accuracy for strict language consistency when reasoning in ES.

- **GPQA / MMLU Pro Medicine:** Accuracy drops in most cases, but language consistency rises after SFT, indicating that it's not trivial to generalize the capability of generating the correct answer from the training domain to others.

**Takeaway.**  
SFT reliably improves language consistency **and often increases accuracy on in-domain tasks (Math).**  
On OOD, SFT can over-narrate or change prior most probable token paths since the models are undertrained to reason in lower-resource languages — accuracy may dip unless taking further actions (e.g., reinforced by RL, shown in RQ2 and RQ3).

**Practical guidance.**  
If your target is **language consistency/reasoning style + some accuracy**, SFT alone is cost-effective in-domain.  
If you also need robustness on hard and/or OOD sets, doing an **RL top-up could be helpful.**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r1.png" class="img-fluid" %}

---

## RQ2 — When RL comes, how does GRPO help with accuracy?

**Design.**  
Train GRPO only on Math-500; evaluate deltas (**GRPO-SFT − SFT**) across  
MMLU-Math / AIME / GPQA / MMLU-Med (Fig. RQ2).

**In-domain.**  
SFT helps accuracy, but not always; GRPO brings a boost on top of the base SFT while maintaining language consistency of reasoning traces.

- **MMLU-Math-FR** pass@10: **76.0 → 97.8 → 98.0** (Base → SFT → GRPO-SFT)  
- **MMLU-Math-ES** pass@10: **80.5 → 98.6 → 99.1** (Base → SFT → GRPO-SFT)  
- **MMLU-Math-JA** pass@10: **68.1 → 88.0 → 91.5** (Base → SFT → GRPO-SFT)

The improvement in accuracy is consistent but slight due to the fact that MMLU-Math is relatively easy:  
The model almost achieves 90–100% accuracy after SFT, leaving no room for GRPO. Thus, the OOD sets are more informative.

**Out-of-domain.**

Positive transfers on **AIME JA/FR/ES and GPQA JA/FR**.  
For instance:

- **GPQA-ES** pass@10: **68.7 → 85.2 → 85.7** (Base → SFT → GRPO-SFT)  
- **AIME-JA** pass@10: **22.6 → 28.5 → 34.4** (Base → SFT → GRPO-SFT; GRPO adds a large JA gain)

More results are shown in the figure below.  
Although improvements on AIME-FR/ES and GPQA-ES are marginal, they still indicate a successful transfer of knowledge on the OOD setup after GRPO.

**Negative transfers on Pro-Medicine.**

- Accuracy improves on Pro-Medicine-JA but decreases on French and Spanish.

**Interpretation.**  
GRPO learns verification/search habits that generalize: language consistency, math reasoning styles, re-checking numeric steps, and tighter answer boxing.  
Those help **GPQA and AIME**.  
But medicine needs domain lexicon, evidence phrasing, and calibrated claims — **absent in math RL**.  
Previous works have shown reasoning-only post-training harms performance on downstream instruction-following and knowledge recall tasks <d-cite key="aggarwal2025optimalthinkingbench"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r2.png" class="img-fluid" %}

---

## RQ3 — Where should RL start from: Base or SFT?

**Design.**  
Compare **GRPO-from-Base vs GRPO-from-SFT** (Fig. RQ3).

**Patterns.**

- **GRPO-from-SFT is a steadier path.**  
  On MMLU-Math FR, for example, GRPO-SFT sits around **~98 pass@10** while GRPO-Base is closer to **~70**,  
  i.e., **starting from SFT provides language consistency and still improves accuracy.**

- **SFT → RL keeps the multilingual policy.**  
  Because SFT already forced the model to reason in Japanese/French/Spanish,  
  RL on top of that mostly optimizes correctness **without switching back to EN/ZH reasoning** (Fig. 1.b).

**Interpretation.**  
**SFT establishes the multilingual “reasoning policy.”**  
Starting RL from the SFT model lets GRPO optimize correctness *while preserving language consistency*.  
RL from Base sometimes pushes the model back toward its original reasoning style while still producing answers in the target language.  
That can make a few out-of-domain slices look better, but it also increases variance and **style regression** compared to starting from SFT.

**Practical rule.**  
If you care about following (see Figure 1.b) **and** better in-domain accuracy, **do GRPO after SFT.**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r3.png" class="img-fluid" %}

---

## RQ4 — Can we push the Pareto frontier instead of trading accuracy for language consistency?

**Design.**  
Plot Accuracy (x-axis) vs Following (y-axis) for each regime (4-panel Pareto figure).  
Then, inspect bar/line panels per dataset and language.

### What we see.

- **SFT shifts points up** (Following ↑).  
  On some hard sets, accuracy dips slightly.

- **GRPO-SFT shifts rightward** (Accuracy ↑) with at most a small upward loss, compared with SFT-only — **creating new frontiers on:**
  - **MMLU-Math (JA/FR/ES):** both metrics are high.  
  - **GPQA-ES:** strong frontier point.

- **Non-frontier holdouts:** Pro-Med FR/JA and AIME-ES, where domain/reward mismatch persists.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r4.png" class="img-fluid" %}

**Bottom line.**  
Read each plot within the same language marker (Japanese ▲, French ■, Spanish ●) and compare colors:

- **yellow vs. blue** = GRPO-from-SFT vs. Base  
- **green vs. blue** = SFT vs. Base  

Under this pairing:

> **GRPO-from-SFT (yellow) strictly Pareto-dominates Base (blue) in 9 of 12 language–dataset pairs** (higher on both accuracy and following).

In the remaining pairs, yellow usually raises following but gives up a little accuracy —  
i.e., a mixed trade-off rather than a strict Pareto gain.

SFT (green) vs. Base (blue) generally shifts points up/right, and **GRPO-from-SFT most often traces the upper-right envelope** when strict dominance does occur.

---

## RQ5 — Does model merging help?

**Motivation.**  
GRPO+SFT often peaks on math but can regress on knowledge-heavy sets (e.g., Pro Medicine),  
and SFT alone doesn’t consistently stabilize accuracy across Japanese/French/Spanish.  

Ideally, we want a solution that smooths these trade-offs while **keeping language-consistency strong**.  
Previous studies have shown that model merging is a promising approach to combine models’ abilities,
albeit with some performance degradation <d-cite key="ustun-etal-2024-aya"></d-cite>.

Here, we merged the base model with the other three SFT models using `merge-kit` with an equal linear merge.

> The merged approach is quite promising as a one-stop solution!

### Result (avg pattern across datasets)
{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r5b.png" class="img-fluid" %}

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r5a.png" class="img-fluid" %}

**MERGE consistently shrinks worst-case losses and raises floor performance**, especially where SFT/GRPO dip.  
On Pro Medicine, MERGE recovers large chunks of accuracy for Japanese/French  
(e.g., JA pass@10 climbs from SFT/GRPO’s ~47–58% to ~70%; FR from ~47–70% to ~76%),  
while staying competitive on AIME/GPQA and within a few points of GRPO+SFT on MMLU-Math.

In Spanish, where SFT already leads on Medicine, MERGE lands in the middle of Base vs SFT/GRPO+SFT rather than decreasing performance to Base.

Overall, it trades a small slice of peak scores for **lower variance across languages and tasks.**

### Interpretation

Parameter-space interpolation acts like an ensemble/regularizer:

- MERGE **blends GRPO’s strong multi-step heuristics** with **SFT’s alignment priors**  
- Dampens overfitting to any single regime  
- **Stabilizes cross-lingual behavior**

Practically, it expresses a steering effect:

> “You can dial toward robustness without re-running RL.”

When you need:
- the **highest leaderboard peak**, pick **GRPO+SFT**  
- **reliable, in-language reasoning across JA/FR/ES**, especially on domain-heavy sets, pick **MERGE**

> MERGE is the safer default when you are data + compute-poor.

---

## Discussion: Where performance regresses, and potential solutions

**Empirical signal.**  
After SFT followed by GRPO, Japanese language consistency improves markedly, but accuracy lags French (e.g., AIME-JA pass@1 **4.4 → 17.9**, pass@10 **22.6 → 34.4**;  
AIME-FR pass@1 **22.2 → 27.3**, pass@10 **46.3 → 48.2**), indicating Japanese-specific friction even with its high increase.  

Spanish on AIME shows the opposite tension: the **Base** model scores well because it always reasons in English despite Spanish prompts, while **SFT+GRPO enforces Spanish chains and accuracy drops**.  

In Pro-Medicine, **math-only GRPO from SFT causes regression** (e.g.,  
FR pass@10 **70.1 → 46.6**, ES **86.6 → 76.6**, JA **75.9 → 58.3**), whereas GRPO started from Base hurts less.

### Mechanisms

1. **Language-prior competition.**  
   The model’s strongest *reasoning prior* is in EN/ZH.  
   Under difficulty, chains drift toward those priors.  
   SFT+GRPO strengthens language consistency, which **reduces access to English-anchored reasoning traces** that previously helped (e.g., AIME-ES).  
   → evidenced by the huge language-consistency bump.

2. **Tokenizer & formatting tax (Japanese > French / Spanish).**  
   Mixed scripts, half/full-width digits, unit variants, and thousand separators inflate perplexity on numeric steps — precisely where accuracy is most sensitive.

3. **Cue misalignment in Spanish math.**  
   AIME leans on algebra/number-theory “recipes” the model learned primarily in English  
   (phrases like “let x be,” “gcd,” “mod”).  
   Spanish equivalents (“sea x,” “mcd,” “módulo”) are rarer, longer, more accented \
   → model drifts into slower or incorrect approaches mid-solution.

4. **Reward misspecification in medicine.**  
   Math-only RL optimizes numeric correctness, **not** biomedical recall, calibration, or evidence style. The policy over-indexes math heuristics and becomes **over-assertive** on clinical QA.

5. **Starting-point effect.**  
   RL from SFT pushes the policy toward SFT’s language/style anchors and away from neutral reasoning.  
   On medicine, this causes bigger drops. RL from Base is more neutral; regressions are smaller.

### Lightweight fixes that may work across cases

- **Prompt-level normalization (before more training).**

  - *Japanese:* unify to half-width digits/decimals/exp notation; no thousand separators;  
    explicit math chain template in Japanese. \
    Example: `数字は半角… SI を使用し…`.  

  - *Spanish:* prefer `gcd / lcm / mod`, exponent notation, half-width digits;  
    terse step headers (`Definimos / Sustituimos / Comprobación / Respuesta`).

- **Tokenizer-aware formatting.**  
  Consistent spacing around numerals/operators; avoid formatting that fragments tokens.

- **Targeted SFT top-ups.**  
  Small, math-dense Japanese/Spanish datasets using normalized templates to reinforce per-language priors.

- **Reward shaping for GRPO.**
  
  - For **AIME-ES**: up-weight *correctness* and make **“Spanish-only chain”** a secondary objective.  
    → nudges reasoning into Spanish **without punishing English-anchored correct answers**.

  - For **Medicine**: add a **tiny medical reward head**  
    (terminology fidelity, claim calibration, evidence cues),  
    plus a **KL / behavior-cloning regularizer** toward medical SFT to preserve discourse style.

  - Use **mixed-objective batches** (math + clinical QA),  
    and replay OOD medical exemplars during RL to avoid domain forgetting.

### Takeaway

The regressions likely stem from one cause:

> **objective + prior mismatch**.

Japanese/Spanish math suffers from tokenization and cue issues; medicine suffers from the absence of domain-specific rewards. Normalizing inputs, adding small language-aware SFT top-ups, and turning “math-only RL” into multi-objective RL (with correctness-first weighting for AIME-ES and a small medical head for Pro-Medicine) could be promising ways to recover accuracy while keeping outputs in the target language and accurate.

---

## Blog Summary — Practical takeaways

1. **If you can only afford one step, do SFT (a few hundred high-quality SFT data).**  
   You’ll almost certainly fix language-consistency without compromising accuracy;  
   you might also get accuracy improvements on in-domain tasks.

2. **If you can afford two steps, do SFT → GRPO-SFT.**  
   Use **high clip / no KL**; keep rollouts moderate; verify you haven’t regressed following.

3. A practical and computationally efficient approach is **model merging among SFT models**.

4. **For medicine or other narrative-dense domains, add a tiny domain reward with in-domain data or a dozens-scale domain SFT.**

5. **For Japanese (or any non-Latin script), include numeric/style templates**  
   and optionally patch tokenization via formatting.

6. **Track Pareto, not single metrics.**  
   Always plot *(Accuracy, Following)* together; real wins move you **up-and-right**.

---

## Limitations & threats to validity

- **Dataset scope.**  
  We use four well-known benchmarks; real-world prompts are noisier.

- **Reward misspecification.**  
  Math-only RL can hurt non-math; the suggested fixes mitigate but don’t prove generality across all medical subspecialities.

- **Model prior.**  
  EN/ZH dominance shapes outcomes. A different base prior (e.g., EU-centric) could change which languages are hardest.

- **Language-consistency metric.**  
  Strong, script-aware, but still an automatic proxy; human raters may be stricter.

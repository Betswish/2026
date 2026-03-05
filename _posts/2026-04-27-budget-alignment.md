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
  - name: "Post-review update — LLM-as-judge audit of reasoning-chain quality"
  - name: "Discussion - Where performance regresses, and potential solutions"
  - name: "Blog Summary - Practical takeaways"
  - name: "Limitations and threats to validity"
---

*Please read this as a late-stage work in progress shared in a “lab meeting” spirit to help and motivate parallel research.*

## Introduction

You ask a large language model (LLM) a math question in Japanese. It responds politely in Japanese — but behind the scenes, it’s reasoning in English/Chinese. Variables, steps, and mathematical lemmas often silently switch languages during reasoning. This behavior, where models default to English for chain-of-thought (CoT) reasoning, is more than a curiosity. It breaks instruction-following, confuses human overseers, and undermines the purpose of multilingual evaluation.

The goal is clear: we want models to reason about a question in the language they are asked, not just answer in that language. But this turns out to be harder than it sounds. Forcing models to reason in non-English languages usually leads to a drop in accuracy. Previous work shows that instructing models to reason only in the prompt language via prompting or steering improves coherence and grading alignment <d-cite key="zhong2025language"></d-cite>, but often comes at a steep "accuracy tax." Even a small amount of multilingual fine-tuning helps, but does not eliminate the trade-off <d-cite key="qi-etal-2025-models"></d-cite>. Further, models not only prefer to reason in English, they often reason more effectively in English. When researchers force strict in-language reasoning (e.g., in Swahili or Thai), models often lose accuracy compared to when allowed to reason in English. For higher-resource languages like French or German, this trade-off is smaller. For lower-resource languages, strict enforcement harms performance more significantly.

Why do models switch to English in the first place? Much of it traces back to training. Most reasoning data are in English. Fine-tuning even strong multilingual models on English CoT data often leads them to adopt English as their "internal language of logic." Yong et al. (2025) observe a "quote-and-think" behavior <d-cite key="yong2025crosslingual"></d-cite>, where models copy input phrases in the prompt language but explain everything in English <d-cite key="kim2025one"></d-cite>. The model understands the question in the non-English language. It simply prefers to reason in English.

Our technical goal is simple: **stop the switching without paying an accuracy tax** — ideally, push the Pareto frontier of *(Accuracy, Language-consistency)*.  
And we want this post to serve as a practical guide with lessons learned along the way.  

One scope note matters for everything that follows: in this work, **math is a proxy task for reasoning behavior**. We are not claiming that multilingual math benchmarks alone capture all real-world reasoning. We use math because answers are verifiable and errors are easy to localize, which makes it a clean testbed for studying whether language-aligned reasoning helps or hurts reasoning quality. We then stress-test transfer on science and medicine to see where this proxy stops being sufficient.

Code, data, and checkpoints will be linked in the **camera-ready** version of this post to preserve anonymity during review.

---

## What we try (Method in two steps)

**Base model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, a reasoning model distilled from R1 through supervised fine-tuning on its reasoning traces, with a strong English/Chinese prior.

**Step 1 — Small SFT to teach in-language reasoning.**  
We fine-tune on **817 curated multilingual reasoning chains** from LiMO <d-cite key="ye2025limo"></d-cite>. The goal here is simple: teach the model to keep its reasoning in the user's language while preserving the long-form style of R1. There is no RL in this stage.

**Step 2 — Math-only GRPO to push accuracy while retaining reasoning language.**  
We run an RLVR-style GRPO with no KL, a higher clip of 0.28 vs. -0.2 (DAPO-like <d-cite key="yu2025dapo"></d-cite>), rollout 24, LoRA rank 8, and learning rate `1e-5`, using **only a translated Math-500 set**. The intuition is to let RL sharpen verification behavior on hard reasoning problems while reducing collapse back to English.

We set the verifiable rewards to **1.0 for accuracy, 0.2 for language consistency of reasoning traces, and 0.2 for answer format** <d-cite key="rastogi2025magistral"></d-cite>.

**Evaluation.**

We evaluate across three target languages — **Japanese (JA), French (FR), and Spanish (ES)** — and four benchmarks: **MMLU College Math (MMLU Math), AIME25, GPQA, and MMLU Pro Medicine (MMLU Med)**.

MMLU-Math and AIME are our main in-domain reasoning probes. This is intentional: we treat math as a controlled proxy for reasoning quality because verification is straightforward. GPQA and MMLU-Med are included as out-of-domain checks to measure how far the learned reasoning behavior transfers once the task is no longer "clean math."

We compare four regimes:

- Base → `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`  
- SFT on top of Base  
- GRPO-from-Base  
- GRPO-from-SFT

and track two metrics:

- `pass@k(1,5,10)` where `n = 32` for accuracy
- `Language-consistency %` (both reasoning traces **and** final answers must be in the requested language; script-aware checks)

**How we score language consistency:**  
We check the entire CoT span and the final boxed answer.  
A sample counts as `Following = 1` only if both passages are in the requested language (script tokens, numerals, and markers allowed); otherwise `0`.
We report the % across the set.

---

## Key contributions

The first takeaway is that a surprisingly small amount of high-quality multilingual supervision can re-shape how a reasoning model "talks to itself." With only **817 SFT chains**, language consistency moves close to saturation in French and Spanish, and improves substantially in Japanese.

The second takeaway is that the two-step pipeline behaves like a practical Pareto recipe. SFT gives us the language behavior we want, and GRPO on top of SFT recovers or improves accuracy on harder settings such as AIME and GPQA without broadly collapsing back to English reasoning.

The third takeaway is diagnostic: when the pipeline fails, it usually fails in predictable ways. Japanese shows tokenization and numeric-format friction, Spanish AIME can hit cue mismatch with English-dominant math priors, and medicine highlights reward mismatch when RL is trained on math alone. These are not mysterious failures; they suggest concrete next moves such as tokenizer-aware normalization, small targeted SFT top-ups, and multi-objective RL.

In short, starting from an EN/ZH-heavy prior, small multilingual SFT is the most cost-effective way we found to steer in-language reasoning. GRPO then helps reclaim the difficult reasoning cases, and model merging offers a useful robustness dial when we care more about stability than peak scores.

**Figure 1.a) Performance comparison overall across methods**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/1a.png" class="img-fluid" %}

**Figure 1.b) Overall language consistency rate comparison across methods**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/1b.png" class="img-fluid" %}

---

## RQ0 — Can small SFT reprogram a reasoning model’s "reasoning tone"?

Models often output the final answer in the same language as the user query. We want the **reasoning process** to match the prompt (user) language, too.

**Results.**
SFT drives the language consistency rate close to the ceiling (**~99–100%**) in French and Spanish, and raises Japanese substantially (**high-80s/90s**).
The language consistency rates averaged across all datasets are shown in Fig. RQ0: bars labeled Japanese/French/Spanish.

**Interpretation.**
A few hundred **high-quality chains** are enough to overwrite the English/Chinese inner-monologue bias. Japanese remains the hardest case, which we return to in RQ5.

> Recall that instruction-following does not only mean the answer in the prompt language, but it should also ensure that the language of the reasoning traces is the same as the user's preference to enhance their trustworthiness. SFT alone solves most of the language mismatch with limited accuracy improvements, which are yet lower than the accuracy of reasoning in English (i.e., the gray dashes in Figure 1.a above) in most cases. We provide more details in the next section.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r0.png" class="img-fluid" %}

---

## RQ1 — Does SFT help accuracy, or only language reasoning *style*?

We have shown that **SFT significantly improves language consistency rates**, but how about the accuracy?

Because our training signal is math-heavy, this section should be read as a question about a **reasoning proxy**: does SFT help on verifiable reasoning tasks, and what does that suggest about transfer beyond that proxy?

**Design.**  
Compare the accuracy of **Base vs SFT** on `pass@k` for each dataset-language pair
(Fig. RQ1: Δ pass@10 = SFT − Base).

**Findings.**

- **MMLU-Math:** substantial improvements when train and test are in the same domain  
  - *French:* ~76 → **98**  
  - *Spanish:* ~80 → **99**  
  - *Japanese:* ~68 → **88**

- **AIME:** mixed. Although AIME is still math, it is much harder than LiMO and therefore only partially in-domain. In Spanish in particular, SFT trades some accuracy for stricter in-language reasoning.

- **GPQA / MMLU Pro Medicine:** accuracy drops in most cases while language consistency rises, which suggests that learning to reason in-language does not automatically transfer the right domain knowledge or style.

**Takeaway.**  
SFT reliably improves language consistency **and often increases accuracy on in-domain math tasks.**
Out of domain, however, it can over-narrate or disturb the model's most likely token paths in lower-resource languages, so accuracy may dip unless we add a second stage such as RL, as we show in RQ2 and RQ3.

**Practical guidance.**  
If your target is **language consistency, reasoning style, and some in-domain accuracy**, SFT alone is cost-effective.
If you also need robustness on hard or out-of-domain sets, an **RL top-up is often worth it.**

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r1.png" class="img-fluid" %}

---

## RQ2 — When RL comes, how does GRPO help with accuracy?

**Design.**  
Train GRPO only on Math-500, then evaluate deltas (**GRPO-SFT − SFT**) across
MMLU-Math / AIME / GPQA / MMLU-Med (Fig. RQ2).

**In-domain.**  
SFT helps accuracy, but not always. GRPO usually adds another boost on top of SFT while preserving the language consistency of the reasoning traces.

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

The full picture is in the figure below. Improvements on AIME-FR/ES and GPQA-ES are smaller, but they still point in the same direction: GRPO can recover some hard-problem performance even though it was trained only on translated math.

**Negative transfers on Pro-Medicine.**

- Accuracy improves on Pro-Medicine-JA but decreases on French and Spanish.

**Interpretation.**  
GRPO seems to learn verification and search habits that generalize: staying in-language, re-checking numeric steps, and tightening answer formatting.
Those habits help **GPQA and AIME**.
Medicine, however, depends more on domain lexicon, evidence phrasing, and calibrated claims — all things that are mostly absent from math-only RL.
Previous work has shown that reasoning-only post-training can hurt downstream instruction-following and knowledge recall <d-cite key="aggarwal2025optimalthinkingbench"></d-cite>.

Put differently, this is exactly where the "math as proxy" framing becomes visible: proxy gains are real, but they do not automatically import the domain knowledge and calibration style required in clinical QA.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r2.png" class="img-fluid" %}

---

## RQ3 — Where should RL start from: Base or SFT?

**Design.**  
Compare **GRPO-from-Base vs GRPO-from-SFT** (Fig. RQ3).

**Patterns.**

- **GRPO-from-SFT is a steadier path.**  
  On MMLU-Math FR, for example, GRPO-SFT sits around **~98 pass@10** while GRPO-Base is closer to **~70**. Starting from SFT preserves language consistency and still improves accuracy.

- **SFT → RL keeps the multilingual policy.**  
Because SFT has already pushed the model to reason in Japanese, French, or Spanish, RL on top mostly optimizes correctness **without switching back to EN/ZH reasoning** (Fig. 1.b).

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
Plot Accuracy (x-axis) vs Following (y-axis) for each regime in a 4-panel Pareto figure.
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
Previous studies have shown that model merging can combine complementary abilities, though not without some degradation <d-cite key="ustun-etal-2024-aya"></d-cite>.

Here, we merged the base model with the other three SFT models using `merge-kit` with an equal linear merge.

### Result (avg pattern across datasets)

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r5b.png" class="img-fluid" %}

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/r5a.png" class="img-fluid" %}

**MERGE consistently shrinks worst-case losses and raises floor performance**, especially where SFT/GRPO dip.
On Pro Medicine, MERGE recovers large chunks of accuracy for Japanese and French
(e.g., JA pass@10 climbs from SFT/GRPO’s ~47–58% to ~70%; FR from ~47–70% to ~76%),  
while staying competitive on AIME/GPQA and within a few points of GRPO+SFT on MMLU-Math.

In Spanish, where SFT already leads on Medicine, MERGE lands in the middle of Base vs SFT/GRPO+SFT rather than decreasing performance to Base.

Overall, it trades a small slice of peak scores for **lower variance across languages and tasks.**

### Interpretation

Parameter-space interpolation behaves a bit like an ensemble or regularizer:

- MERGE **blends GRPO’s strong multi-step heuristics** with **SFT’s alignment priors**
- Dampens overfitting to any single regime  
- **Stabilizes cross-lingual behavior**

Practically, it gives us a simple way to dial toward robustness without re-running RL.

In practice:

- the **highest leaderboard peak**, pick **GRPO+SFT**  
- **reliable, in-language reasoning across JA/FR/ES**, especially on domain-heavy sets, pick **MERGE**

> MERGE is the safer default when you are data- and compute-poor.

---

## Post-review update — LLM-as-judge audit of reasoning-chain quality

Reviewers asked a natural follow-up question: if our post-training pipeline improves target-language reasoning, does it also improve the language quality of the reasoning chains themselves? To answer that, we ran an additional LLM-as-judge audit over both the untuned base model and the tuned multilingual AIME generations used throughout this post.

This is still a math-proxy analysis, not a full downstream writing evaluation. The goal is narrower: measure whether the chains become more grammatical, more fluent, and more consistently in the requested language as we move from baseline to SFT, GRPO, and MERGE.

In total, we scored **13,440 reasoning traces**: **3,840** from the untuned base model (EN/ES/FR/JA) and **9,600** from the tuned settings (FR Base+GRPO, FR SFT+GRPO, ES Base+GRPO, ES SFT+GRPO, JA Base+GRPO, JA SFT+GRPO, and MERGE in EN/ES/FR/JA). Each file contains 30 prompts with 32 sampled responses. For each response, we extracted the `<think>...</think>` segment when present, with a fallback for partially emitted tags, and scored that chain against its target language.

We used `gpt-4.1-mini-2025-04-14` as the judge with strict JSON output. It assigned 1-5 integer scores for `grammar_score`, `language_match_score`, and `fluency_score`, along with issue tags and a short rationale. The system instruction was:

> "You are a strict multilingual linguistic quality evaluator... score each reasoning chain for language quality only... Do NOT score math correctness... Return only the JSON object that matches the schema."

The run was fully scripted with resumable batching and schema validation. Across baseline and tuned evaluations together, usage was `20,128,771` input tokens and `1,376,438` output tokens, or about `$10.25` at the `gpt-4.1-mini` pricing available when we ran it.

### What we found

Across all **13,440** chains, the averages were **3.5870** for grammar, **4.0881** for language match, and **3.4548** for fluency, for an overall mean of **3.7100**. That headline number is pulled down mostly by the untuned base model in non-English settings, which is the point: the post-training pipeline is not only changing accuracy, it is changing what language the model thinks in and how stable those chains are.

To make the cross-language comparison easier to read, we summarize the overall judge score first and then show the best tuned gain over baseline. All scores are on a 1-5 scale.

<table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
  <thead>
    <tr style="border-bottom:2px solid #999;">
      <th style="text-align:left; padding:8px 10px;">Language</th>
      <th style="text-align:center; padding:8px 10px; background:#f3f4f6;">Baseline</th>
      <th style="text-align:center; padding:8px 10px; background:#fff4db;">Base+GRPO</th>
      <th style="text-align:center; padding:8px 10px; background:#e8f1ff;">SFT+GRPO</th>
      <th style="text-align:center; padding:8px 10px; background:#e8f7ef;">MERGE</th>
      <th style="text-align:center; padding:8px 10px;">Best tuned delta</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>EN</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#f3f4f6;"><strong>4.8372</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#fff4db;">-</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f1ff;">-</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f7ef;">4.3979</td>
      <td style="text-align:center; padding:8px 10px; color:#b42318;">-0.4393</td>
    </tr>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>ES</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#f3f4f6;">1.7698</td>
      <td style="text-align:center; padding:8px 10px; background:#fff4db;">4.2274</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f1ff;"><strong>4.2861</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#e8f7ef;">4.2767</td>
      <td style="text-align:center; padding:8px 10px; color:#067647;"><strong>+2.5163</strong></td>
    </tr>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>FR</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#f3f4f6;">2.3590</td>
      <td style="text-align:center; padding:8px 10px; background:#fff4db;">4.1063</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f1ff;">4.1257</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f7ef;"><strong>4.1576</strong></td>
      <td style="text-align:center; padding:8px 10px; color:#067647;"><strong>+1.7986</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 10px;"><strong>JA</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#f3f4f6;">3.0465</td>
      <td style="text-align:center; padding:8px 10px; background:#fff4db;">2.8687</td>
      <td style="text-align:center; padding:8px 10px; background:#e8f1ff;"><strong>3.8142</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#e8f7ef;">3.6663</td>
      <td style="text-align:center; padding:8px 10px; color:#067647;"><strong>+0.7677</strong></td>
    </tr>
  </tbody>
</table>

<p style="margin-top:0.75rem; font-size:0.92rem; color:#555;">
  Row colors indicate settings; bold marks the best overall score within each language.
</p>

Because the main jump comes from staying in the requested language, the per-metric deltas are also useful:

<table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
  <thead>
    <tr style="border-bottom:2px solid #999;">
      <th style="text-align:left; padding:8px 10px;">Language</th>
      <th style="text-align:left; padding:8px 10px;">Best tuned setting</th>
      <th style="text-align:center; padding:8px 10px;">Overall</th>
      <th style="text-align:center; padding:8px 10px;">Grammar</th>
      <th style="text-align:center; padding:8px 10px;">Lang-match</th>
      <th style="text-align:center; padding:8px 10px;">Fluency</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>EN</strong></td>
      <td style="padding:8px 10px;">MERGE</td>
      <td style="text-align:center; padding:8px 10px; background:#fef3f2;">-0.4393</td>
      <td style="text-align:center; padding:8px 10px; background:#fdecec;">-0.6041</td>
      <td style="text-align:center; padding:8px 10px; background:#fff6ed;">-0.0938</td>
      <td style="text-align:center; padding:8px 10px; background:#fdecec;">-0.6198</td>
    </tr>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>ES</strong></td>
      <td style="padding:8px 10px;">SFT+GRPO</td>
      <td style="text-align:center; padding:8px 10px; background:#ecfdf3;">+2.5163</td>
      <td style="text-align:center; padding:8px 10px; background:#edfbea;">+2.2292</td>
      <td style="text-align:center; padding:8px 10px; background:#dcfae6;"><strong>+3.1313</strong></td>
      <td style="text-align:center; padding:8px 10px; background:#edfbea;">+2.1885</td>
    </tr>
    <tr style="border-bottom:1px solid #ddd;">
      <td style="padding:8px 10px;"><strong>FR</strong></td>
      <td style="padding:8px 10px;">MERGE</td>
      <td style="text-align:center; padding:8px 10px; background:#edfbea;">+1.7986</td>
      <td style="text-align:center; padding:8px 10px; background:#f0fdf4;">+1.5604</td>
      <td style="text-align:center; padding:8px 10px; background:#dcfae6;">+2.5292</td>
      <td style="text-align:center; padding:8px 10px; background:#f0fdf4;">+1.3063</td>
    </tr>
    <tr>
      <td style="padding:8px 10px;"><strong>JA</strong></td>
      <td style="padding:8px 10px;">SFT+GRPO</td>
      <td style="text-align:center; padding:8px 10px; background:#f7fee7;">+0.7677</td>
      <td style="text-align:center; padding:8px 10px; background:#f7fee7;">+0.8104</td>
      <td style="text-align:center; padding:8px 10px; background:#f7fee7;">+0.8177</td>
      <td style="text-align:center; padding:8px 10px; background:#fefce8;">+0.6750</td>
    </tr>
  </tbody>
</table>

Figure RQ-review below shows the same comparison visually. Absolute judge scores are on a 1-5 scale; the heatmap reports score-point deltas against the untuned baseline.

{% include figure.liquid path="assets/img/2026-04-27-budget-alignment/judge_baseline_vs_tuned.png" class="img-fluid" %}


The comparison mirrors the main story of the paper. The untuned base model is already excellent in English, with **4.8372** overall, but it struggles to stay in-language for multilingual reasoning: baseline ES is only **1.7698**, FR is **2.3590**, and their dominant issues are `wrong_language` and `mixed_language`. In practice, the base model often solves the task by drifting back into English.

Post-training changes that picture substantially. Relative to baseline, the best tuned configuration improves overall chain quality by **+2.5163** in ES, **+1.7986** in FR, and **+0.7677** in JA, with the largest gains coming from language matching itself: **+3.1313** in ES, **+2.5292** in FR, and **+0.8177** in JA. That is exactly the behavior we wanted from multilingual post-training: not just cleaner wording, but a model that actually reasons in the requested language.

Among the tuned variants, SFT+GRPO is modestly better than Base+GRPO for ES and FR and much better for JA, while MERGE stays strong across EN/ES/FR and improves Japanese language matching substantially over JA Base+GRPO. The trade-off is also visible here: MERGE does not beat the raw baseline in English language quality, so multilingual alignment is not a free improvement in every direction at once.

Taken together, the judge audit supports the same conclusion as the accuracy and Following results. The pipeline materially improves in-language reasoning behavior relative to the untuned base model, especially for ES and FR, while Japanese remains the clearest bottleneck. And because this audit is still on math traces, it reinforces the broader framing of the post: math is a useful proxy for reasoning alignment, but not the final destination.

---

## Discussion: Where performance regresses, and potential solutions

**Empirical signal.**  
After SFT followed by GRPO, Japanese language consistency improves markedly, but accuracy lags French (e.g., AIME-JA pass@1 **4.4 → 17.9**, pass@10 **22.6 → 34.4**;  
AIME-FR pass@1 **22.2 → 27.3**, pass@10 **46.3 → 48.2**), indicating Japanese-specific friction even with its high increase.  

Spanish on AIME shows the opposite tension: the **Base** model scores well because it always reasons in English despite Spanish prompts, while **SFT+GRPO enforces Spanish chains and accuracy drops**.  

In Pro-Medicine, **math-only GRPO from SFT causes regression** (e.g.,  
FR pass@10 **70.1 → 46.6**, ES **86.6 → 76.6**, JA **75.9 → 58.3**), whereas GRPO started from Base hurts less.

This is the core tension of the post in one sentence: a math-proxy objective is very good at shaping reasoning discipline, but incomplete for domain-heavy tasks unless we add domain-aware signals.

### Mechanisms

1. **Language-prior competition.**
The model’s strongest *reasoning prior* is in EN/ZH. Under difficulty, chains drift toward those priors. SFT+GRPO strengthens language consistency, which **reduces access to English-anchored reasoning traces** that previously helped (e.g., AIME-ES).

2. **Tokenizer & formatting tax (Japanese > French / Spanish).**
Mixed scripts, half/full-width digits, unit variants, and thousand separators inflate perplexity on numeric steps, precisely where accuracy is most sensitive.

3. **Cue misalignment in Spanish math.**
AIME leans on algebra and number-theory "recipes" the model learned primarily in English (phrases like "let x be," "gcd," and "mod"). Spanish equivalents ("sea x," "mcd," "módulo") are rarer and often more awkward tokenization-wise, so the model drifts into slower or less reliable approaches.

4. **Reward misspecification in medicine.**
Math-only RL optimizes numeric correctness, **not** biomedical recall, calibration, or evidence style. The policy over-indexes math heuristics and becomes **over-assertive** on clinical QA.

5. **Starting-point effect.**
RL from SFT pushes the policy toward SFT’s language/style anchors and away from neutral reasoning. On medicine, this causes bigger drops. RL from Base is more neutral, so regressions are smaller.

### Lightweight fixes that may work across cases

- **Prompt-level normalization (before more training).**
- *Japanese:* unify to half-width digits, decimals, and exponent notation; remove thousand separators; use an explicit math-chain template. Example: `数字は半角… SI を使用し…`.
- *Spanish:* prefer `gcd / lcm / mod`, exponent notation, half-width digits, and terse step headers such as `Definimos / Sustituimos / Comprobación / Respuesta`.
- **Tokenizer-aware formatting.**  
Use consistent spacing around numerals and operators, and avoid formatting that fragments tokens.
- **Targeted SFT top-ups.**  
Use small, math-dense Japanese and Spanish datasets with normalized templates to reinforce per-language priors.
- **Reward shaping for GRPO.**
- For **AIME-ES**: up-weight *correctness* and make **Spanish-only reasoning** a secondary objective, so the model is nudged toward Spanish without punishing otherwise-correct English-anchored solutions.
- For **Medicine**: add a small medical reward head for terminology fidelity, calibration, and evidence cues, plus a KL or behavior-cloning regularizer toward medical SFT.
- Use mixed-objective batches (`math + clinical QA`) and replay OOD medical exemplars during RL to reduce domain forgetting.

### Takeaway

The regressions likely stem from one cause:

> **objective + prior mismatch**.

Japanese/Spanish math suffers from tokenization and cue issues; medicine suffers from the absence of domain-specific rewards. Normalizing inputs, adding small language-aware SFT top-ups, and turning “math-only RL” into multi-objective RL (with correctness-first weighting for AIME-ES and a small medical head for Pro-Medicine) could be promising ways to recover accuracy while keeping outputs in the target language and accurate.

---

## Blog Summary — Practical takeaways

If we had to distill this post into one practical recommendation, it would be this: start with small but high-quality multilingual SFT. It is the highest-leverage step for fixing reasoning-language mismatch, and it often improves in-domain accuracy at the same time.

When compute allows a second stage, SFT followed by GRPO is the more reliable path than RL directly from the base model. In our runs, this sequence usually preserved the multilingual reasoning policy while regaining hard-problem accuracy. If the target environment needs steadier behavior across tasks and languages rather than absolute peak numbers, model merging is a useful compromise.

For teams deploying in domains like medicine, pure math rewards are not enough: add a small domain-aware signal or a compact domain SFT refresh. For Japanese and other non-Latin scripts, numeric and formatting normalization is not cosmetic; it directly affects reasoning stability. Finally, we strongly recommend tracking the full `(Accuracy, Following)` Pareto view, because single metrics can hide exactly the trade-offs this work is trying to solve.

Most importantly, keep the scope explicit when reporting results: **math here is a proxy for reasoning alignment, not the final destination**. It is a useful proxy, but deployment-quality reasoning still needs domain-specific post-training and evaluation.

---

## Limitations & threats to validity

- **Dataset scope.**  
  We use four well-known benchmarks; real-world prompts are noisier.
- **Reward misspecification.**  
  Math-only RL can hurt non-math; the suggested fixes mitigate but do not prove generality across all medical subspecialties.
- **Model prior.**  
  EN/ZH dominance shapes outcomes. A different base prior (e.g., EU-centric) could change which languages are hardest.
- **Language-consistency metric.**  
  Strong and script-aware, but still an automatic proxy; human raters may be stricter.
- **Math as a reasoning proxy.**  
  Much of our optimization and auditing is built around verifiable math-style reasoning. This is useful for controlled analysis, but it does not by itself guarantee equal gains in domains that depend more on retrieval, calibration, or specialized discourse norms.

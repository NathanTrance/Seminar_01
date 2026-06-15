# One-Step Flow Matching: Seminar Paper & Presentation Guide

This README contains the structured outline for your Master's seminar on **One-Step Flow Matching**, including the paper backbone, presentation flow, and the unzipped NeurIPS 2026 LaTeX template.

## Unzipped LaTeX Template

The **NeurIPS 2026 formatting template** has been extracted in the `LaTeX/` folder:

- `neurips_2026.sty` — The style package (load with `\usepackage{neurips_2026}`)
- `neurips_2026.tex` — The example/shell document (replace content with your own)
- `checklist.tex` — The required checklist for submission (if applicable)

### Quick Setup
1. Open `neurips_2026.tex`
2. Replace the placeholder title, author, abstract, and body text with your own content.
3. Keep the style file (`neurips_2026.sty`) in the same directory.
4. Use the `\usepackage{neurips_2026}` option (default for submission).

**Note:** The template limits the main content to **9 pages** (figures included). References, acknowledgments, and appendices do not count toward this limit.

---

## Paper Structure (Backbone)

### Target: 10 pages, NeurIPS style

| Section | Pages | What to Cover |
| :--- | :--- | :--- |
| **Abstract** | 0.5 | One paragraph: problem (multi-step diffusion is slow), solution (one-step flow matching), and your key contributions. |
| **1. Introduction** | 1.5 | The rise of diffusion/flow models. The computational cost of sampling. The promise of one-step generation. Roadmap of the report. |
| **2. Background** | 2 | **2.1** Rectified Flow: straight paths, reflow, and the multi-stage limitation. **2.2** Flow Matching: the unified framework, OT paths, and the conditional velocity objective. |
| **3. One-Step Generative Modeling** | 4 | **3.1** Consistency Flow Matching (velocity consistency for straight flows). **3.2** Mean Flows (average velocity, training from scratch). **3.3** Align Your Flow (scaling distillation, why CMs fail). **3.4** Terminal Velocity Matching (terminal regularization, $W_2$ bounds). |
| **4. Theoretical Analysis** | 1.5 | Error bounds (Wasserstein, transport cost). The unification of consistency and flow matching under the flow map formalism. |
| **5. Extensions: Reward Alignment** | 1 | Meta Flow Maps. Stochastic posterior sampling. Inference-time steering without rollouts. |
| **6. Conclusion** | 0.5 | Summary and future directions. |
| **References** | — | BibTeX for all 7 papers (see Materials/README.md). |

---

## Presentation Outline (Slides)

### Target: 45–50 minutes, ~18–20 slides

| Section | Slides | What to Cover |
| :--- | :--- | :--- |
| **1. Motivation** | 2 | The speed-accuracy trade-off. Why 50-step diffusion is too slow for real-time applications. |
| **2. Background: Rectified Flow** | 2 | The straight-path idea. Train → simulate → re-train. The first step toward one-step, but multi-stage. |
| **3. Framework: Flow Matching** | 2 | The unified theory (Lipman et al.). Conditional velocity regression. Why OT paths are straight. |
| **4. Method 1: Consistency Flow Matching** | 3 | Enforcing *velocity consistency* for straight flows. Multi-segment training. The first true "one-step" attempt. |
| **5. Method 2: Mean Flows** | 3 | The **average velocity** insight. The MeanFlow Identity. Training from scratch without distillation. |
| **6. Method 3: Align Your Flow** | 3 | The failure of Consistency Models in multi-step (theorem). Continuous-time distillation (AYF-EMD/LMD). Autoguidance + adversarial finetuning. |
| **7. Method 4: Terminal Velocity Matching** | 3 | Terminal-time regularization. The $W_2$ upper bound. Architectural fixes (RMSNorm, JVP kernel). |
| **8. Extensions: Meta Flow Maps** | 2 | Stochastic maps. One-step posterior sampling. Reward alignment without rollouts. |
| **9. Comparison & Results** | 1 | A single table: FID vs. NFE for ImageNet 64/256/512. |
| **10. Conclusion** | 1 | Summary: The shift from "simulating ODEs" to "learning the map." |

---

## Narrative Arc: How to Tie the Papers Together

Use this story to make your seminar cohesive:

1. **Rectified Flow** asks: *"What if the paths were straight?"* → Straight paths can be simulated in 1 step, but require multiple training stages.
2. **Flow Matching** asks: *"Can we unify all this under one framework?"* → Yes, via conditional velocity regression on straight OT paths.
3. **Consistency Flow Matching** asks: *"Can we enforce straightness in the network itself?"* → Yes, via velocity consistency along the trajectory.
4. **Mean Flows** asks: *"Do we need a pre-trained teacher to distill from?"* → No, we can train the average velocity from scratch in a single stage.
5. **Align Your Flow** asks: *"How do we scale this to high-res images?"* → Use continuous-time flow maps, autoguidance, and adversarial finetuning.
6. **Terminal Velocity Matching** asks: *"Can we prove this works?"* → Yes, terminal matching upper-bounds the $W_2$ distance, and architectural fixes stabilize training.
7. **Meta Flow Maps** asks: *"What if we want to control the model post-hoc?"* → Use stochastic maps for one-step posterior sampling and reward alignment.

---

## Recommended Figure 1

Add a single **Figure 1** in your paper to justify the entire seminar:

- **(a)** Curved diffusion path (50 steps, slow)
- **(b)** Straight OT path (Rectified Flow, fewer steps)
- **(c)** Direct one-step jump (Consistency / MeanFlow / AYF)

This figure immediately shows the progression: *simulation → straightening → direct prediction*.

---

## Key Papers (Chronological)

| Paper | arXiv | Role |
| :--- | :--- | :--- |
| Rectified Flow | 2209.03003 | **Background**: Straight paths, reflow idea |
| Flow Matching | 2210.02747 | **Framework**: Unified theory, OT paths |
| Consistency-FM | 2407.02398 | **Method 1**: Velocity consistency |
| Mean Flows | 2505.13447 | **Method 2**: Average velocity, from scratch |
| Align Your Flow | 2506.14603 | **Method 3**: Scalable distillation |
| Terminal Velocity Matching | 2511.19797 | **Method 4**: $W_2$ bounds, stability |
| Meta Flow Maps | 2601.14430 | **Extension**: Reward alignment |

For detailed summaries, abstracts, and pseudocode of each paper, see `Materials/README.md`.

---

## Practical Tips

- **Page limit**: 9 pages of content + references/appendices (NeurIPS default). Aim for ~10 pages total.
- **Template**: Use `neurips_2026.sty` with the `default` option for submission.
- **References**: Use BibTeX. The `natbib` package is loaded automatically by the style file.
- **Math**: Use `\[ ... \]` or `align` environments instead of `$$` for correct line numbering.
- **Figures**: Use `\includegraphics[width=0.8\linewidth]{...}` and place captions *below* the figure.
- **Tables**: Use the `booktabs` package (already included in the template). No vertical rules.

---

*End of README.*

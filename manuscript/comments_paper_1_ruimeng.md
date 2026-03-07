# Reviewer/Co-Author Comments: Rationales for Rejected Suggestions

This document outlines the specific comments from RH that were rejected or addressed alternatively, along with the structural rationales for doing so.

---

### 1. Introduction (Section 1.1)

**Comment:** > `\rh{This sentence is way too long. Consider rephrase it like "Second, in the DL setting, ERM calibrates a deterministic mapping from inputs to outputs. This produces imputations rather than statistical forecasts, making it difficult to construct predictive intervals and forecast distributions that require conditional stochasticity."}`

**Action Taken:** Rejected. The original phrasing was retained.

**Rationale:** The original sentence is intentionally structured to emphasize the specific, mechanical difference between point imputations and the conditional stochasticity required for formal probabilistic forecasting. While long, the sentence is technically precise and establishes a central theme of our framework that a split-sentence structure dilutes.

---

### 2. Generative Feature Learning (Section 4.1)

**Comment:** > `\rh{Here you mentioned both DeepLOB and CNN-LSTM architectures. But they are all mixed together. I would suggest you introduce DeepLOB first. After finishing its discussion, then talk about how CNN-LSTM is different from CNN-LSTM [DeepLOB].}`

**Action Taken:** Rejected. The original paragraph flow was maintained.

**Rationale:** The current narrative flow introduces the CNN-LSTM architecture by immediately framing it as a necessary refinement of DeepLOB. Separating them into disjointed subsections would disrupt the logical progression. Presenting them together efficiently motivates the CNN-LSTM as a direct solution to the stationarity and computational bottlenecks of the original model.

**Comment:** > `\rh{There are training details. Better put them in Section 11, maybe in 11.1}`

**Action Taken:** Rejected. The ADAM optimizer hyperparameters were kept in Section 4.1.

**Rationale:** Moving the specific neural network hyperparameters to Section 11 would fragment the description of the Deep Learning module. Keeping them in Section 4.1 provides the reader with a complete, self-contained understanding of the "Backbone" module's setup before the text transitions to the econometric "Adapter" module. Moreover, Gareth mentioned that the training hyperparameters are technically part of the DL model.

---

### 3. Proposed Hybrid Model (Section 5)

**Comment:** > `\rh{better first give definition of mixing property}`

**Action Taken:** Rejected. A formal mathematical definition was not added.

**Rationale:** Providing a formal definition of mixing properties would act as a tangential distraction. The manuscript relies on established, sufficient conditions (i.e., the top Lyapunov exponent) to guarantee ergodicity. A deep dive into the underlying mixing properties is unnecessary to support our main results and would bog down the section's pacing.

**Comment:** > `\rh{do you mean to replace $\phi_{\mathbf{W}}$ by $\mathrm{CNNLSTM}_{\mathbf{W}}$ in equation 5?}`

**Action Taken:** Rejected altering the equation; addressed via text clarification instead.

**Rationale:** Equation 5 is designed specifically to formally define the primary (generative) model. Instead of adding a redundant equation block just to swap one term, a clarifying sentence was added to explain that the parsimonious model directly substitutes the embedding with the raw output.

---

### 4. End of Document (Author Contributions)

**Comment:** > `\rh{I would suggest removing this section. Papers published in Quantitative Finance typically do not include an author contributions section. Moreover, the ordering of authorship already reflects the relative contributions of the authors.}`

**Action Taken:** Rejected. The section remains in the `.tex` file.

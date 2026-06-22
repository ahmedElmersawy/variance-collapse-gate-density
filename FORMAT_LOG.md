# Format Log: Conversion to Submission-Ready NeurIPS Paper

Audit trail for the formatting/presentation pass. No result, number, or
claim was changed in this pass — only how the existing, already-verified
content is organized, compressed, and anonymized. Every number that
appears below was already established and committed in prior sessions
(see `RESULTS_LOG.md` for the original derivation of each).

## Style files

Retrieved the official NeurIPS 2026 author kit directly from
`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`
(linked from the official Call for Papers page) rather than a third-party
mirror. Confirmed via the official Main Track Handbook
(`https://neurips.cc/Conferences/2026/MainTrackHandbook`):
- Main-text page limit: **9 pages**, including figures/tables.
- References, appendices, and the paper checklist do **not** count toward
  the limit.
- The paper checklist is **mandatory**; omitting it is grounds for desk
  rejection.
- Anonymized submission: no `final` or `preprint` style option at
  submission time (this auto-adds line numbers and an anonymous-author
  placeholder).
- `neurips_2026.sty` geometry: `textwidth=5.5in`, `textheight=9in`,
  single column — confirmed this matches the page dimensions the prior
  (non-official) custom preamble was already using, so content density
  per page is directly comparable between drafts.

Copied `neurips_2026.sty` and `checklist.tex` into the project root
unmodified (per the CFP: "Tweaking the style files may be grounds for
desk rejection").

## Compression plan

Prior `main.tex` (1492 lines, ~26 main-text-equivalent pages before
`\appendix`) + `appendix_synthetic_theory.tex` (810 lines) compiled to 42
total pages. Target: 9 pages of main text, everything else moved to an
unlimited-length appendix with a one-line pointer left in the main text
at the point each item was removed from.

Main text keeps: Abstract, Introduction+Contributions, condensed Related
Work, condensed Method, the core direction-split + architecture-fixed
ablation result, the mechanism (BN identity, gamma-shrinkage,
threshold-crossing, per-channel verification, and the cross-optimizer
sigma-collapse prediction as the headline), a condensed generalization
paragraph (optimizer/scale/architecture), Limitations, Conclusion.

Moved to appendix, each with a pointer left in the main text: full
threshold-robustness detail and figure, full BatchNorm-vs-GroupNorm
detail and figure, the full smoothness-sweep detail and both figures, the
two honestly-reported non-findings (rank null, pruning) in full, the
exploratory privacy probe in full, ConvNeXt-Tiny exclusion detail, the
mu-drift equilibrium derivation (Task B) in full, the low-data
fine-tuning null (Task D) in full, full reproducibility/hyperparameter
detail, the method-validation table, and the entire pre-existing
synthetic phase-transition theory (already its own appendix subsection,
kept as-is).

## Result

Final main text: **6 pages** (Introduction through Conclusion; References
and Appendix begin on page 7), comfortably under the 9-page limit, with
room used to restore three figures (smoothness sweep, pruning,
threshold-robustness summary) and the BatchNorm/threshold robustness
numbers into the main text rather than leaving them as bare appendix
pointers, since the first compressed draft (6 pages from a sparser main
text) had slack remaining. Total document (main text + references +
unlimited appendix + checklist): 41 pages.

Two genuinely missing macros surfaced on the first compile of the
restructured preamble (`\argmin`, the `float`/`placeins`/`multirow`
packages for `[H]` placement and tables used inside the pre-existing
`appendix_synthetic_theory.tex`) and one citation-style mismatch
(`natbib` defaulted to author-year processing against a numbered
`thebibliography`; fixed with `\PassOptionsToPackage{numbers,compress}{natbib}`
before loading the style file, exactly as the official template's own
comments suggest). Two real missing cross-reference labels
(`sec:smoothness`, `sec:privacy`) were lost when their content moved into
the appendix and needed to be re-added at the new location, not assumed
to resolve themselves on a later LaTeX pass.

## Register/tone cleanup in the pre-existing synthetic-theory appendix

Found and fixed three instances of development-history meta-narrative in
`appendix_synthetic_theory.tex` (not touched in the prior science-pass
sessions, since it predates this paper's restructuring): one redundant
sentence ("The prior label \`\`plateau'' for Swish was incorrect...")
that added no information beyond what the preceding bullet list already
stated correctly, deleted outright; one passage referencing "earlier
drafts" reporting a since-superseded estimate, rewritten to state the
final grid-design methodology neutrally without the revision-history
framing, preserving the identical final numbers. Left one "retired"
methodology disclosure in place (a 4-parameter Fisher-information fit
abandoned for an ill-conditioned matrix) since it is a legitimate,
neutral statistical-methodology statement, not a confession about the
paper's own draft history — and one scope-boundary sentence about a
theorem's asymptotic validity, for the same reason.

## Task A framing fix (the most consequential editorial decision in this pass)

The pre-format draft's Section 5.3 (and abstract, and conclusion)
described the AdamW per-channel test in language a reviewer could read as
claiming the predictor is an oracle that "divined" an unrelated outcome
purely from a sign test. Reframed to make explicit, in the same
location, that (a) the per-channel correlation step is expected to hold
under any optimizer because the $\sigma$-normalized margin and
active-gradient fraction are related through a near-definitional
monotonic relationship once $\mu,\sigma,z_{\mathrm{low}}$ are fixed, so
that step is not independent evidence; (b) the substantive test is
whether AdamW's \emph{actually measured} $\mu,\sigma$ trajectories (not
previously known) land in the regime the Gaussian account requires; (c)
the $\sigma$-change table (Table~\ref{tab:sigma_optimizer}) is presented
first as the mechanistic reason, with the sign test presented after it as
confirmation that the account is quantitatively faithful, not as the
primary evidence. No number changed; every percentage and $p$-value is
identical to the pre-format draft (verified by direct comparison against
`RESULTS_LOG.md`). Title changed from "Hard Gates Collapse, Soft Gates
Saturate: Two Regimes of Gradient Flow in Trained Networks" (false under
AdamW, where there is one regime, not two) to "Variance Collapse Predicts
When Gate Density Diverges by Activation Class," matching the corrected
framing.

## LLM usage declaration (flagged for the user, not decided unilaterally)

The paper checklist's final item (declaration of LLM usage) is answered
\answerYes{} with a placeholder justification explicitly marked
\texttt{[AUTHORS: fill in to match your actual workflow before
submission.]} An LLM-based coding agent was used substantially beyond
writing/editing/formatting throughout this project's experimental work
(implementing and debugging experiment code, running training jobs,
performing statistical analyses) — this is a real disclosure decision
that belongs to the human author(s) submitting the paper, not something
this formatting pass should answer on their behalf with invented detail.
Flagged explicitly when reporting this work back, not buried in a
checklist box.

## Verification performed

- Fresh-extracted the submission zip (`neurips_submission.zip`:
  `main.tex`, `appendix.tex`, `appendix_synthetic_theory.tex`,
  `checklist.tex`, `neurips_2026.sty`, `figures/`) into a clean directory
  and rebuilt end-to-end with 3 `pdflatex` passes: clean, 0 undefined
  references, 0 missing figures (41 unique references checked), 41 total
  pages, main text confirmed at 6 pages via the same fresh build.
- `grep -rinE 'purdue|gilbreth|aelmersa|elmersawy|blindpeer'` and a
  separate sweep for `github.com`, SLURM job IDs, and `scratch/gilbreth`
  paths: zero hits across `main.tex`, `appendix.tex`,
  `appendix_synthetic_theory.tex`, `checklist.tex`. The anonymous-author
  title block is additionally enforced by the official style file itself
  (confirmed by reading `neurips_2026.sty`: it substitutes a hardcoded
  "Anonymous Author(s)" block whenever neither the `final` nor `preprint`
  option is passed, regardless of `\author{}`'s actual content).
- Spot-checked a sample of the highest-stakes numbers (the AdamW
  $\sigma$-change table, the 12/12 and 48/48 sign-test results, the 9/9
  mechanism-validation table, the Places365 magnitudes) directly against
  `RESULTS_LOG.md` and against the pre-format `main.tex` (recovered via
  `git show HEAD:main.tex`): unchanged.

# Data Description HTML Guidelines for Empirical Studies

Canonical guidance for creating a dataset description HTML page for a real-data
empirical study in this repository. Use `docs/avito-dataset.html` as the
reference standard. The goal is not only to document fields, but to give an
empirical researcher enough context to interpret the treatment, outcome,
controls, cleaning choices, identifying assumptions, and result artifacts.

The page should be a research reference that can later support a manuscript
section, appendix, replication package, and future robustness work.

## 1. Core Principles

- **Document the empirical design, not just the dataset.** A useful page explains
  the unit of analysis, treatment, outcome, controls, estimand, and threats to
  interpretation.
- **Separate public metadata from local audit facts.** Public dataset pages
  establish provenance. Repository scripts establish the actual sample used by
  the code.
- **Separate raw variables from study variables.** For example, a raw `price`
  column can be a dataset fact, while `log1p(price)` is a repository-defined
  treatment.
- **Make every headline number traceable.** Row counts, missingness, dates,
  feature dimensions, trimming thresholds, and result links should point to a
  script, config, cache, source, or generated artifact.
- **State the causal status honestly.** For observational datasets, frame results
  as conditional-adjustment evidence unless the design truly provides randomized
  or quasi-random treatment variation.
- **Audit control validity.** Explicitly flag variables that are
  treatment-derived, post-treatment, outcome-derived, leakage-prone, diagnostic
  only, or reserved for heterogeneity analysis.
- **Keep the page aggregate-first.** Prefer aggregate statistics, public
  screenshots, and schema summaries over row-level examples that may expose
  private or license-sensitive content.

## 2. Evidence to Collect Before Writing

Collect these facts before drafting the HTML page:

- Dataset source links: platform page, dataset page, data dictionary, license,
  competition or publication page, and any mirrors used for readable metadata.
- Local source layout: raw table names, archive names, image/text/audio shards,
  cache directories, feature manifests, and expected file counts.
- Unit of observation: listing, seller, buyer, transaction, session, product,
  review, patient, firm, store, etc.
- Time and market scope: date range, geography, platform, verticals, sample
  window, and whether current platform facts differ from the historical sample.
- Raw schema audit: column roles, types, missing counts, missing percentages,
  unique counts, and notes for research use.
- Cleaning attrition: every filter from raw rows to final analysis rows.
- Treatment and outcome construction: raw source fields, transformations,
  bounds, mass points, tails, trimming, winsorization, and interpretation.
- Control construction: tabular features, unstructured modalities, embeddings,
  row alignment keys, feature dimensions, and excluded variables.
- EDA facts: treatment/outcome distributions, unconditional association,
  category mix, geography, time, user/seller types, text lengths, file coverage,
  image dimensions, and modality missingness.
- Empirical specification: target equation, identifying assumption, nuisance
  variables, overlap concerns, and estimator families.
- Reproducibility artifacts: scripts, configs, notebooks, cache manifests,
  generated tables, figures, and expected QA checks.

## 3. Recommended HTML Structure

Each dataset page should follow this structure unless the dataset genuinely lacks
a modality or component.

### 3.1 Hero and Snapshot

Purpose: let a reader understand the study in the first viewport.

Include:

- Dataset name and empirical role.
- Generation date or data-audit snapshot date.
- One-sentence study question.
- Raw row count, raw column count, modality file count, and cleaned analysis row
  count.
- Unit of observation.
- Treatment, outcome, and control modalities.
- Table of contents with anchors to all major sections.

Avoid:

- Generic marketing text about the platform.
- Stating current platform scale as if it were a sample statistic.
- Reporting a cleaned sample count without saying which filters define it.

### 3.2 Institutional Context and Provenance

Purpose: explain what real-world process generated the observations.

Include:

- What the platform, market, or institution is.
- What one row means economically.
- Who chooses the treatment and when.
- Who observes the controls and outcome-relevant signals.
- Public source links and local code links.
- A table separating dataset facts from study interpretations.

For marketplace data, explicitly distinguish listings from products, sellers,
buyers, transactions, sessions, and paid advertisements. For health, education,
finance, or policy data, make the analogous distinction for the relevant unit.

### 3.3 Visual Context

Purpose: help readers understand the empirical object and unstructured content.

Include:

- Public screenshots or platform images when they clarify what users observe.
- Captions that explain what is visible and why it matters for confounding.
- Source links for each image.
- A statement that screenshots are contextual only if they are not from the
  actual sample.

Avoid:

- Embedding private user content, seller images, medical images, or row-level
  examples unless the license and ethics are clear.
- Treating screenshots as evidence about sample statistics.

### 3.4 Raw Schema

Purpose: make the original data auditable.

Use a table with these columns:

- `Column`
- `Role`
- `Type`
- `Missing`
- `Missing %`
- `Unique`
- `Research note`

Roles should be concrete:

- Identifier
- Treatment source
- Outcome
- Structured control
- Text
- Image or media pointer
- Timestamp
- Geography
- User or seller attribute
- Post-treatment variable
- Diagnostic only

Research notes should explain whether each field is retained, transformed,
excluded, or risky for causal adjustment.

### 3.5 Cleaning and Sample Construction

Purpose: make the analysis sample reproducible.

Include an attrition table with:

- Step name.
- Rows before.
- Rows after.
- Rows dropped.
- Percent of raw retained.
- Optional remaining-share bar.

Document all filters in order, including:

- Treatment observed and valid.
- Outcome observed and valid.
- Required structured fields observed.
- Text nonempty or language-valid.
- Media ID observed.
- Local media file exists.
- Duplicate handling.
- Date or market restriction.
- Price or treatment trimming.
- Sample caps or stratified subsampling.

Also report final sample properties:

- Unique units after cleaning.
- Repeated-unit structure, such as listings per seller or observations per firm.
- Date range.
- Final treatment cap or other thresholds.
- Whether modality coverage is complete or limiting.

### 3.6 EDA

Purpose: show the empirical terrain before adjustment.

At minimum include:

- Treatment distribution: mean, SD, quantiles, max, transformation rationale, and
  tail behavior.
- Outcome distribution: mean, SD, quantiles, bounds, mass points, and whether it
  is observed, generated, censored, or predicted.
- Simple treatment-outcome association before adjustment.
- Important stratifications: category, region, time, seller type, product type,
  institution, or any domain-specific grouping.
- Text diagnostics: title/description length, token length, language, empty
  fields, truncation rate, and model input length.
- Image or media diagnostics: file count, file size, dimensions, aspect ratio,
  missingness, corruption rate, and preprocessing transform.

Interpret the EDA through the causal design. For example, a positive
unconditional price-demand association in a marketplace can be evidence
consistent with quality confounding, not a causal result.

### 3.7 Current Feature Schema

Purpose: bridge raw data to estimator inputs.

Include:

- Treatment variable and transformation.
- Outcome variable and interpretation.
- Structured control list and encodings.
- Text construction, embedding model, dimensionality, normalization, truncation,
  and cache location.
- Image construction, backbone, dimensionality, normalization, preprocessing, and
  cache location.
- Row alignment keys across modalities.
- Cache manifest hash or version.
- Legacy feature pipelines that are superseded.
- Fields retained for diagnostics or heterogeneity but excluded from controls.

If a feature is deterministic in the treatment, derived from the outcome, or
observed after treatment, state that it must not enter the adjustment set.

### 3.8 Research Notes and Caveats

Purpose: prevent overclaiming.

Include concise notes on:

- Observational versus randomized design.
- Missing variables that could confound the treatment-outcome relationship.
- Generated, predicted, censored, smoothed, or proxy outcome labels.
- Treatment interpretation, such as posted price versus transaction price.
- Sample selection from modality availability.
- Tail behavior and trimming sensitivity.
- Boundary mass or limited support in the outcome.
- Language, modality, platform, or measurement constraints.
- High-cardinality identifiers and whether clustering or fixed effects are
  feasible.
- Heterogeneity across markets or product categories.

Each caveat should also say how the paper should handle it, not merely list it.

### 3.9 Empirical Section Guide

Purpose: make the dataset page directly usable for manuscript drafting.

Include separate subsections for:

- Outcome definition and permitted wording.
- Treatment definition and transformation rationale.
- Control variables and confounding channels.
- Empirical specification and identification.
- Estimator menu.
- Main result interpretation.
- Robustness and caveats.

For equations, prefer **native MathML** so the committed HTML page is a pure
static artifact with no JavaScript (see Section 4 for the rationale and the
hard rule against `$`-delimited live MathJax on these pages). A generic
partially linear specification, in MathML, is:

```html
<p class="math-display">
  <math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
    <msub><mi>Y</mi><mi>i</mi></msub><mo>=</mo>
    <mi>θ</mi><msub><mi>D</mi><mi>i</mi></msub><mo>+</mo>
    <msub><mi>g</mi><mn>0</mn></msub><mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>)</mo>
    <mo>+</mo><msub><mi>ε</mi><mi>i</mi></msub>
  </math>
</p>
```

Author MathML by writing the LaTeX once and converting it (e.g. with a local
MathJax/Pandoc pass, or by extracting the `<math>` block MathJax emits when it
renders the TeX), then paste the resulting MathML into the page. Define every
symbol in words:

- `Y`: outcome.
- `D`: treatment.
- `X`: pre-treatment controls.
- `g_0`: flexible baseline outcome function.
- `theta`: target treatment coefficient.

State whether DML or another orthogonal method also learns a treatment equation,
rendered the same way (`D_i = m_0(X_i) + v_i`).

### 3.10 Result Artifacts

Purpose: connect the data page to empirical outputs.

Link:

- Main result table CSV.
- Main result table LaTeX fragment.
- Robustness tables.
- Modality ablation tables.
- Sample-size or hyperparameter sweep figures.
- Config files used to generate results.
- Notebooks or scripts that assemble paper artifacts.

For each artifact, state its intended use in the paper.

### 3.11 Source Notes for Paper Writing

Purpose: keep citations and claims clean.

Use a table with:

- `Statement`
- `Source`
- `How to use it`

The table should distinguish:

- Public platform context.
- Dataset provenance.
- Raw data dictionary definitions.
- Repository-specific cleaning and feature construction.
- Result artifacts.

This prevents a paper from citing Kaggle or a platform webpage for a statistic
that actually comes from a local cleaning script.

### 3.12 Reproducibility Checklist

Purpose: provide a final QA gate.

Include expected values for:

- Raw row count.
- Modality file count.
- Rows with valid media files.
- Cleaned analysis rows.
- Date range.
- Treatment transformation.
- Trimming threshold.
- Feature dimensions.
- Cache manifest.
- Text/image example policy.
- Script used to regenerate the page.

The checklist should be specific enough that a future data refresh can quickly
detect drift.

## 4. HTML Organization and Formatting

The HTML page can be hand-written or generated, but it should be easy to inspect
in a browser and commit as a static artifact.

Recommended organization:

- Use one self-contained HTML file with standard `<!doctype html>`, `html`,
  `head`, and `body` elements.
- Put the primary content inside a centered `main` element.
- Start with a hero section containing the page title, short lede, audit date,
  and table of contents.
- Use one top-level `section` per major topic, with stable `id` attributes.
- Use `h2` for major sections and `h3` for local subsections.
- Keep each major section visually separable with either a full-width content
  band or a simple bordered panel.
- Put dense facts in tables rather than long paragraphs.
- Put headline counts in metric cards only when they summarize central facts.
- Keep source links close to the facts they support.
- End with source notes and a reproducibility checklist.

Recommended formatting:

- Keep CSS inside the page unless a shared docs stylesheet exists.
- Use a restrained research-document style: readable width, clear section
  hierarchy, tables, metric cards, notes, and compact callouts.
- Include a sticky-free or simple table of contents with anchor links.
- Use responsive grids that collapse on mobile.
- Use tabular numerals for numeric columns.
- Use visible source links for public claims.
- Render equations as **native MathML** (`<math>...</math>`), not via a live
  MathJax CDN script. Native MathML is supported by current Safari and Chrome,
  keeps the page a static no-JavaScript artifact, and removes the network
  dependency.
- **Never use `$...$` as a MathJax inline-math delimiter on these pages.** Dataset
  pages are full of currency values (`$1.42` spend, `$499.00` max, price columns).
  Live MathJax with `$` inline delimiters pairs a currency `$` with the next `$`
  anywhere in the document, swallows the prose/markup in between as one giant
  "expression," and tries to typeset it — which pegs the CPU and **freezes the
  browser tab** (observed hanging Safari on a 51 KB page). If a live MathJax pass
  is genuinely unavoidable, configure inline math as `\(...\)` and never `$...$`,
  but prefer pre-rendered MathML and ship no script at all.
- Use alt text and source captions for images.
- Avoid decorative graphics that do not carry empirical information.
- Avoid embedding large base64 images or generated files that make diffs
  unreadable.

Recommended section IDs:

- `snapshot`
- `context`
- `screenshots`
- `schema`
- `cleaning`
- `eda`
- `features`
- `research`
- `empirical-section`
- `sources`
- `reproducibility`

## 5. Control-Validity Audit

Before finalizing a page, classify every candidate control:

| Class | Use in adjustment? | Examples | Documentation requirement |
|---|---:|---|---|
| Pre-treatment structured control | Yes | category, region, seller type, timestamp before exposure | Explain causal channel. |
| Pre-treatment unstructured control | Yes | image, text, audio, profile content | Explain what latent quality or information it proxies. |
| Identifier | Usually no | seller ID, user ID, product ID | Use for clustering, grouping, leakage checks, or fixed effects only if justified. |
| Treatment-derived | No | price tier, discount from price, bins of treatment | Mark as excluded from `X`. |
| Outcome-derived | No | labels computed from outcome, target encodings using outcome | Mark as leakage. |
| Post-treatment | No | clicks, impressions after treatment, messages, sales status | Exclude unless redesigning the estimand. |
| Diagnostic or heterogeneity field | Not in main `X` | raw category labels, price tier for plots | State allowed uses. |

This audit belongs in either the raw schema, feature schema, or research notes.

## 6. Manuscript Paragraph Template

The empirical section should usually follow this order:

1. **Setting and question.** Introduce the platform or institution, the unit of
   analysis, and the treatment-outcome question.
2. **Sample construction.** Explain raw source, local files, filters, date range,
   and final analysis sample.
3. **Variables and EDA.** Describe treatment scale, outcome interpretation,
   distributional facts, and simple unconditional association.
4. **Controls and confounding.** Explain why structured and unstructured
   variables are common causes or proxies for common causes.
5. **Specification and identification.** Present the partially linear equation
   and conditional-adjustment assumption.
6. **Estimator menu.** Compare no adjustment, tabular adjustment,
   embed-then-infer, direct high-dimensional adjustment, and causal embedding
   when relevant.
7. **Main result.** Emphasize how estimates move as controls expand.
8. **Interpretation.** Connect estimate movement to the proposed confounding
   mechanism and method contribution.
9. **Robustness and caveats.** Summarize stability checks and state residual
   identification limits.

For datasets without a true treatment effect, do not frame one estimator as
"correct" by assertion. The central evidence is usually movement, agreement, and
stability across estimator families.

## 7. Anti-Patterns

Avoid these common failures:

- A data dictionary with no treatment, outcome, or estimand.
- A paper-ready narrative with no reproducible row counts.
- A single final sample count with no attrition table.
- Public platform statistics mixed with local sample statistics.
- Outcome wording stronger than the public label supports.
- Treatment transformation presented without economic rationale.
- Images or text described as "features" but not as confounder proxies.
- Including treatment-derived bins in the main control matrix.
- Leaving cache dimensions, model names, or row alignment keys undocumented.
- Reporting estimates without linking the source table or config.
- Treating screenshots as observations from the analysis sample.
- Omitting caveats because they are inconvenient for the empirical claim.
- Loading a live MathJax CDN script with `$...$` inline delimiters on a page that
  contains currency dollar signs — this collides currency `$` with math `$` and
  can freeze the browser. Ship pre-rendered native MathML instead.

## 8. Minimal Completion Checklist

A dataset description HTML file is complete only if it answers:

- What is one row?
- Where did the data come from?
- What does the platform or institution do?
- What are the treatment, outcome, and controls?
- Which variables are raw and which are repository-engineered?
- What exact sample does the repository use?
- How many observations are lost at each cleaning step?
- Which modalities are available and how are they featurized?
- Which variables are excluded from adjustment and why?
- What empirical specification is estimated?
- What identifying assumption is needed?
- What causal interpretation is credible?
- Which scripts, configs, caches, tables, and figures reproduce the page?

## 9. Suggested File and Workflow Pattern

Use this pattern for a new empirical dataset named `<dataset>`:

- Create `docs/<dataset>-dataset.html`.
- Store public visual context under `docs/assets/<dataset>/`.
- Add or identify a data-audit script that computes raw counts, missingness,
  attrition, EDA, and feature metadata.
- Link the page to the relevant loader or cleaner under `src/unstructured/`.
- Link result artifacts under `results/empirical_study/<dataset>/` and figures
  under `figures/<dataset>/` if they exist.
- Regenerate the page whenever the cleaning pipeline, feature cache, or analysis
  sample changes.

If the data page is generated from a notebook or script, the generated HTML
should still be readable as a standalone committed artifact.

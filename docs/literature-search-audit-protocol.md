# Literature Search Audit Protocol

This note records the search protocol to use when auditing whether a dataset has
appeared in target journals. It was added after the Upworthy audit initially
missed a 2025 `Marketing Science` article because Crossref and OpenAlex did not
yet expose the publisher record.

## Core Rule

Publisher pages are authoritative for journal publication status. Crossref,
OpenAlex, Semantic Scholar, Google Scholar, and web search are discovery tools,
not final evidence that a paper is unpublished.

For INFORMS journals, check Pubsonline directly. For AMA/SAGE journals, check
SAGE directly. If a publisher page and an aggregator disagree, use the publisher
page and document the discrepancy.

## Required Search Order

1. Search the publisher's journal-native site.
2. Open every plausible publisher article page and verify title, outlet, volume,
   issue, pages, publication date, and dataset-use language.
3. Search Crossref and OpenAlex by exact title, DOI, author names, and dataset
   terms as secondary checks.
4. Search broad web results for working-paper to publication transitions.
5. Record negative searches with exact journal, search terms, and search date.

## Upworthy-Specific Lessons

The paper `LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments`
by Ye, Yoganarasimhan, and Zheng was accepted to `Marketing Science` and appears
on Pubsonline with DOI `10.1287/mksc.2024.0990`. During the June 27, 2026 audit,
Crossref and OpenAlex DOI lookups did not return this DOI, and OpenAlex surfaced
the paper as an arXiv record. Treat this as a concrete example of why aggregator
absence is not publication-status evidence.

Verified Upworthy `Marketing Science` records from Pubsonline:

- Banerjee and Urminsky, `The Language That Drives Engagement: A Systematic Large-scale Analysis of Headline Experiments`, DOI `10.1287/mksc.2021.0018`.
- Ye, Yoganarasimhan, and Zheng, `LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments`, DOI `10.1287/mksc.2024.0990`.

## Publisher Search URLs

Use these journal-native search URL patterns for the four requested journals:

- `Marketing Science`: `https://pubsonline.informs.org/action/doSearch?AllField=<TERM>&SeriesKey=mksc`
- `Management Science`: `https://pubsonline.informs.org/action/doSearch?AllField=<TERM>&SeriesKey=mnsc`
- `Journal of Marketing Research`: `https://journals.sagepub.com/action/doSearch?AllField=<TERM>&SeriesKey=mrja`
- `Journal of Marketing`: `https://journals.sagepub.com/action/doSearch?AllField=<TERM>&SeriesKey=jmxa`

If direct HTTP requests are blocked by Cloudflare or a similar challenge, use a
real browser session and wait for the page to load. Do not replace blocked
publisher-native search with only aggregator searches.

## Search Terms

For Upworthy, use at least these terms in each target journal:

- `Upworthy`
- `Upworthy.com`
- `Upworthy Research Archive`
- `headline experiments`
- `content experiments`
- `clickability_test_id`
- Known working-paper titles or acronyms, such as `LOLA`
- Known author names from working papers, such as `Zikun Ye`,
  `Hema Yoganarasimhan`, `Yufeng Zheng`, `Akshina Banerjee`, and
  `Oleg Urminsky`

For other datasets, include:

- Official dataset name and aliases.
- Platform/company name.
- Unique column names or identifiers.
- Known working-paper titles and author names.
- Domain-specific phrases used in the data paper.

## Evidence Standard

Count a paper as a verified use only when at least one of these is true:

- The publisher article page states the dataset/platform in the abstract,
  article text, supplement, or metadata.
- The publisher page plus article title/abstract makes the dataset use
  unambiguous.
- A downloaded publisher PDF or supplement directly identifies the dataset.

Do not count a paper based only on topical similarity, broad method similarity,
or an aggregator result that lacks dataset-use evidence.

## What to Record

For each target outlet, record:

- Publisher and journal.
- Search URL or search UI path.
- Search terms.
- Date searched.
- Result count.
- Plausible titles opened.
- Whether each title is a verified use, adjacent topical analogue, or false
  positive.
- Any aggregator-publisher discrepancy.

## Upworthy Audit Result as of June 27, 2026

Publisher-native searches found two verified top-journal Upworthy uses, both in
`Marketing Science`:

- `The Language That Drives Engagement`, DOI `10.1287/mksc.2021.0018`.
- `LOLA`, DOI `10.1287/mksc.2024.0990`.

Publisher-native searches found no qualifying Upworthy uses in:

- `Management Science`
- `Journal of Marketing Research`
- `Journal of Marketing`

Those negative findings used publisher searches for `Upworthy`,
`Upworthy.com`, `Upworthy Research Archive`, `headline experiments`,
`content experiments`, `LOLA`, and `clickability_test_id`, with Crossref and
OpenAlex used only as secondary checks.

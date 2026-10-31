---
title: "Research Agents: Web Search, Fetch and Citations, Measured"
description: "Claude's web search and fetch tools on 30 fresh questions: one search vs a research loop, cost per answer, and a hand audit of 60 citations."
slug: "research-agents"
category: "agents-tool-use"
tags: ["agents", "tool-use", "hallucination", "evals", "cost"]
series: "agentic-ai"
seriesOrder: 10
pubDate: 2026-07-10
updatedDate: 2026-10-24
---

## TL;DR

A research agent answers a question by searching the web, reading pages and citing what it read. This article runs Claude's server-side web search and web fetch tools on 30 questions about events from July to October 2026, after the models' training cutoff, with answers checked by hand against primary pages. Claude Haiku 5.5 with a single search got 85% right; the same model in a research loop got 98% for 2.4 times the cost, $0.028 per question, and 89% of that bill was the $0.01 search fee rather than tokens. Claude Sonnet 5.5 answered all 15 questions of the subset it ran on, at $0.062 each. A hand check of 60 sampled citations found no claim that its cited page contradicted, but one cited page in six could not be opened by a reader at all, and 23 of 60 quoted snippets didn't contain the fact they were attached to.

## Why it matters

Ask Claude Haiku 5.5 which shipped first, Rust 1.99.0 or Python 3.15.0, give it one web search, and it answers with confidence: "both were released on October 1, 2026 (0 days apart)." The Rust date was right and carried a citation. The Python date came from memory: the published schedule had said October 1, but the release slipped to October 9. Allowed a few more searches, the same model got it right in both runs.

The model's knowledge stops at a training cutoff (June 2026 for both models used here, per the [models overview](https://platform.claude.com/docs/en/about-claude/models/overview)), the web keeps moving, and a fluent answer mixes facts it looked up with facts it remembered. Citations are supposed to show which is which. This part measures how much searching buys, what it costs, and whether the citations hold up when you open the pages yourself.

[Part 5](/blog/context-engineering/) showed that tool results dominate an agent's context window. Search results and fetched pages are among the largest tool results an agent sees, so that part's advice applies here unchanged.

## The intuition

Think of a newspaper fact-checker with a library card. A reporter hands over a draft; the checker looks up each claim, photocopies the page that supports it, and staples the copy to the draft. A careful checker goes to the original source (the court filing, the company's own announcement) instead of another newspaper's summary.

A research agent does the same with a search engine, a page reader and citation markers. The analogy breaks in three places. The agent usually answers from the short extracts of page text that come with search results. It can only open a page whose address has already appeared in the conversation, a security rule explained below. And its "photocopy" is a quote of at most 150 characters that may come from the top of the page rather than the line that matters.

## Step 1: search and fetch as server tools

Apart from Part 6's tool search, the tools in Parts 1 to 6 were client tools: the model asks for a call, your code runs it and sends the result back. Web search and web fetch are server tools. Anthropic's API runs them inside the request, in a server-side loop, and returns the finished turn: the model's searches, the results, any pages it fetched, and an answer with citations.

![A flowchart. Your code sends the question and tools. Inside one API request, the model thinks about what is still missing and runs web_search, which costs $0.01 and adds about 10 results with page text to the context. If the model wants to fetch a page, the API checks whether its URL is already in the context: if yes, web_fetch adds the full page; if no, it returns the error url_not_in_prior_context. The model then decides whether it has enough to answer. If not, it searches again; if so, it writes an answer with citations giving a URL and cited text. Back outside the request, your code checks the citations.](/diagrams/a10-research-loop.svg "One research turn. The loop runs on Anthropic's servers; your code sees the finished turn and checks the citations.")

From your side, the whole loop is one API call. Here is the tool configuration used for every experiment, from the lab's `research.py`:

```python
SEARCH = {"type": "web_search_20250305", "name": "web_search"}            # basic search
FETCH = {"type": "web_fetch_20250910", "name": "web_fetch",               # basic fetch
         "citations": {"enabled": True}, "max_content_tokens": 12_000}
SEARCH_DF = {"type": "web_search_20260318", "name": "web_search"}         # dynamic filtering
FETCH_DF = {"type": "web_fetch_20260318", "name": "web_fetch",
            "citations": {"enabled": True}, "max_content_tokens": 12_000}

TOOLS = {
    "one-search": [{**SEARCH, "max_uses": 1}],
    "research": [{**SEARCH, "max_uses": 6}, {**FETCH, "max_uses": 4}],
    "research-df": [{**SEARCH_DF, "max_uses": 6}, {**FETCH_DF, "max_uses": 4}],
}
```

`max_uses` caps searches or fetches per request, and `max_content_tokens` truncates a fetched page, the same result sizing as in Part 5.

### Versions, prices and rules

| Tool version | What it adds | Notes from the docs |
|---|---|---|
| `web_search_20250305` | Basic search | Eligible for Zero Data Retention |
| `web_search_20260209` | Dynamic filtering | Claude writes code that filters results before they enter the context |
| `web_search_20260318` | `response_inclusion` | Can drop filtered results from the response |
| `web_fetch_20250910` | Basic fetch of HTML and PDF pages | Citations optional, off by default |
| `web_fetch_20260209`, `_20260309`, `_20260318` | Dynamic filtering, then `use_cache`, then `response_inclusion` | Results may come from a cache unless `use_cache` is false |

None needs a beta header ([tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference)). Dynamic filtering runs the tools from inside [code execution](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool), which the API provisions automatically. It needs a model that supports programmatic tool calling; both `claude-haiku-5-5` and `claude-sonnet-5-5` are on that [list](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling).

[Web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) costs "$10 USD per 1,000 searches, plus standard token costs for search-generated content," and a search that errors is not billed. [Web fetch](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool) has "no additional charges beyond standard token costs"; the docs estimate an average 10 kB web page at about 2,500 tokens and a 500 kB research PDF at about 125,000. Code execution is free when it runs for these tools ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)). The lab's ledger already reads `usage.server_tool_use.web_search_requests` and adds $0.01 per search, so every cost below includes the fees: the experiments ran 302 searches, $3.02 of their $4.03.

Two rules shape how a research agent behaves:

- Web fetch "can only fetch URLs that have previously appeared in the conversation": in a user message, a client tool result, or an earlier search or fetch result. A URL the model writes from memory fails with `url_not_in_prior_context`. The docs give the reason as data exfiltration: "Claude cannot fetch URLs that appear only in its own output," so injected instructions can't make it smuggle data out inside a URL it builds ([Prompt Injection Defense for Tool-Using Agents](/blog/prompt-injection-defense-tool-agents/) covers that attack class).
- A long server-side turn can stop with `stop_reason: "pause_turn"`. You send the assistant content back unchanged and the server continues ([server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools)). None of the 150 runs here paused, but the lab's loop handles it:

```python
        if resp.stop_reason == "pause_turn":
            # A paused turn: send the assistant content back unchanged and let the server continue.
            if messages[-1]["role"] == "assistant":
                messages[-1] = {"role": "assistant", "content": turn}
            else:
                messages.append({"role": "assistant", "content": turn})
```

### What the response looks like

Each search arrives as a `server_tool_use` block with the query, then a `web_search_tool_result` with about ten results: URL, title, page age and `encrypted_content`, page text the model reads but you can't. The answer is split into text blocks, and a cited block carries `citations` with the URL, title and a `cited_text` of "up to 150 characters," which doesn't count toward token usage. A fetched page arrives as a document, and its citations point at character offsets in it.

## Step 2: thirty questions with checkable answers

A fair test needs questions the models can't answer from memory and answers you can check yourself. All 30 concern July to October 2026, after the June 2026 cutoff, and every answer was read off a primary page on 2026-10-25 before any model ran: python.org, the Rust blog, go.dev, openjdk.org, postgresql.org, kernel.org, the Federal Reserve's FOMC (Federal Open Market Committee) statements and nobelprize.org. The one exception, the 2026 World Cup final, was checked on its Wikipedia article.

- 12 single-fact questions. "On what date was Python 3.15.0 released?" (2026-10-09). "What is the planned general availability date of PostgreSQL 19?" (2026-11-19, set in the October 15 release candidate announcement; the project's roadmap had said October). "What target range did the FOMC set on September 16, 2026?" (3.75% to 4.00%, a rate rise).
- 12 multi-hop questions, which need two facts and often arithmetic. "How many days after Go 1.27.0 did JDK 27 reach General Availability?" (August 19 to September 15: 27). "Which university do two 2026 Nobel laureates in different categories share?" (Stanford: Deisseroth in medicine, Hall in economics).
- 6 deep questions, which need five or more facts. "How many of the 2026 Nobel laureates were born in the United States?" needs ten birthplaces (answer: 3). "Put Python 3.15.0, Go 1.27.0, JDK 27, Rust 1.99.0 and Linux 7.3 in release order."

Several hide a trap: an announced schedule that the real event missed. Python 3.15.0 was scheduled for October 1 and shipped October 9; PostgreSQL 19's roadmap said October, and the date moved to November 19. A model that trusts its memory or a stale schedule page gets these wrong.

A script grades the answer's final `ANSWER:` line (numbers, dates and versions must open it; "unable to determine" is wrong), and all 150 verdicts were then confirmed by reading the answers. Hedged but correct answers ("27 days (assuming JDK 27 GA on 15 September 2026, which I could not confirm)") count as correct, and the hedges are reported separately.

## Step 3: one search against a research loop

Four configurations ran, all with effort `medium` and adaptive thinking (the model decides when to think):

- One search: `claude-haiku-5-5`, search only, `max_uses: 1`, a one-line system prompt.
- Research loop: `claude-haiku-5-5`, up to 6 searches and 4 fetches, and a system prompt that tells it to break the question into facts, prefer primary sources, fetch a page when a snippet isn't conclusive, and do arithmetic step by step.
- Research loop with dynamic filtering: the same, on the `_20260318` tool versions.
- Sonnet headline: `claude-sonnet-5-5` with the research-loop configuration.

The two Haiku arms ran all 30 questions twice (60 runs each). To fit the $5 budget, Sonnet and dynamic filtering ran once on 15 questions, five of each kind, including every question either Haiku arm had missed. Each arm's two-question pilot, with an unchanged configuration, counts as the first repeat. Accuracy intervals are 95% Wilson intervals, a range for a rate that stays sensible near 0% and 100%.

### All 30 questions, Haiku 5.5

| Arm | Correct (95% CI) | Single | Multi-hop | Deep | Searches | Input tokens | Cost per question | Mean latency |
|---|---|---|---|---|---|---|---|---|
| One search | 51/60, 85% (74-92) | 23/24 | 20/24 | 8/12 | 1.00 | 13,573 | $0.0113 | 4.9 s |
| Research loop | 59/60, 98% (91-100) | 24/24 | 24/24 | 11/12 | 2.45 | 32,892 | $0.0275 | 6.1 s |

"Input tokens" sums everything the model read across the server loop's iterations, cached or not. Latency is the wall time of the API calls.

The intervals touch, but the comparison is paired: the same question and repeat ran in both arms. In 8 pairs one search was wrong and the loop right, and in no pair the reverse. If the two arms were equally good, eight one-sided splits would happen with probability 2 × (1/2)^8, about 0.008, so the gap is not noise.

### Where one search failed

Single facts were easy. One search failed when a question needed two facts. Told it had one search, Haiku tried a second anyway in 17 of 60 runs; the API returned `max_uses_exceeded`, and the model filled the gap from memory. Both runs of the Rust-and-Python question did this. On the five-release ordering, only the Linux date came back from the search, and both runs guessed the rest from release cadences and the old schedule, tying Python and Rust on October 1: "I've put Python first, but that is a guess." In three other runs the model declined instead: "Unable to determine. Python 3.15.0 was released October 9, 2026, but the Linux 7.3 release date could not be verified."

Seven one-search answers carried a warning such as "unverified" or "tentative" and three more declined to answer, while one of the research loop's 60 answers hedged. One search also returned a stale result once: "1.97.1" as the latest Rust release, from a July blog post, when 1.99.1 had shipped on October 15.

The research loop's single miss was the ten-birthplace question. It used all six searches looking up each prize, ran into `max_uses_exceeded` on a seventh, and answered conservatively: "1 (confirmed: Karl Deisseroth; up to 3 if both U.S.-described economists Robert E. Hall and Michael Woodford were U.S.-born, which I could not verify)." On the second repeat it found both birthplaces on nobelprize.org and answered 3.

### The 15-question subset, all four arms

| Arm | Runs | Correct (95% CI) | Searches | Fetches | Input tokens | Cost per question | Cost per correct answer | Median latency |
|---|---|---|---|---|---|---|---|---|
| One search, Haiku | 30 | 21/30, 70% (52-83) | 1.00 | 0 | 13,711 | $0.0114 | $0.0163 | 5.9 s |
| Research loop, Haiku | 30 | 29/30, 97% (83-99) | 2.87 | 0.30 | 40,393 | $0.0324 | $0.0335 | 6.9 s |
| Dynamic filtering, Haiku | 15 | 13/15, 87% (62-96) | 4.53 | 1.67 | 60,912 | $0.0505 | $0.0583 | 20.0 s |
| Research loop, Sonnet | 15 | 15/15, 100% (80-100) | 1.80 | 0.13 | 25,883 | $0.0621 | $0.0621 | 5.5 s |

Sonnet 5.5 answered every question with fewer searches than Haiku (1.8 against 2.9) and far less thinking: 68 thinking tokens per question on average, against about 360. Its first attempt at the Rust question ended early, though: it called a nonexistent tool named `fetch` with a URL from memory, and the API returned `stop_reason: "tool_use"`, waiting for a client tool result that the lab's first loop never sent. The loop now answers such a call with an error ("There is no tool named fetch. Use web_search or web_fetch."), and the rerun, which the table counts, answered correctly.

Dynamic filtering lost to the basic loop on every measure here. It searched and fetched more, cost 56% more per question than the basic loop on the same subset, and took a median 20 seconds, with one run at 123 seconds. In that run the model repeated one query, ran out of searches before reaching postgresql.org, and answered "Cannot be determined from the sources found"; the only date it had found was a third-party blog's estimate, which it quoted as "expected around September or October 2026." The docs pitch dynamic filtering at token savings on search-heavy work, and these questions needed one sentence from a page, which the basic tools already deliver. Much of what it read arrived as code output rather than as search results, and it produced 21 citations in 15 runs, against 221 in 60 runs for the basic loop.

### Where the money goes

On Haiku 5.5 the search fee is the bill. Each extra search added about 10,500 tokens of input (13,573 tokens for one search, 24,305 for two), which costs about $0.0013 at Haiku's cache-write price of $0.125 per million tokens. The fee for that search is $0.01, eight times more. Across both Haiku arms, 88% to 89% of every dollar was search fees. On Sonnet 5.5, whose cache writes cost $2.50 per million tokens, the same 10,500 tokens cost about $0.026, and the fees fell to 29% of the bill.

On Haiku, then, cut searches: `max_uses`, a prompt that asks for one well-aimed query per fact, and a cache of answers you have already researched. Trimming tokens with `max_content_tokens` or dynamic filtering barely moves the bill. On Sonnet, tokens dominate, and Part 5's sizing advice applies directly.

### Fetch is rarer than you would think

Across its 60 Haiku runs, the research loop tried 10 fetches, and 8 failed with `url_not_in_prior_context`; Sonnet's 2 attempts both failed. Every refused URL was one that hadn't appeared yet: `https://www.kernel.org`, `https://blog.rust-lang.org/` and `https://openjdk.org/jeps/523`, typed from memory, sometimes in the same breath as the search that would have surfaced them. The model then answered from the search results, which already carry page text. For short facts that was enough; fetch matters for long documents whose answer sits beyond the search extract.

## Step 4: checking the citations by hand

Citations promise more than a right answer: each cited sentence is backed by the page it points to. Checking that means opening the pages.

### How the check worked

1. A script collected every cited span from the four arms' transcripts: 393 citations, each a pair of the cited answer text and its citation (URL and `cited_text`).
2. A seeded random sample took 60 of them: 15 from one search, 25 from the Haiku research loop, 12 from Sonnet and 8 from dynamic filtering.
3. Each cited URL was fetched again on 2026-10-25 with an ordinary browser user agent, and the HTML reduced to visible text. Pages that returned an error were retried with `curl` and browser headers, then with a second fetcher. Two python.org pages arrived compressed and were read after decompressing.
4. For each sample, the claim's key fact (a date, number or name) was searched for in the page text, and the surrounding passage was read alongside the claim in the context of the full answer.
5. Each was labeled supported (the page states the claim), partly (the page states part of it, or only a forecast of it), not supported (the page doesn't say it or contradicts it) or unreachable (no fetch method returned readable text). Separately, each was marked for whether the 150-character `cited_text` itself contains the fact.

### What the pages said

| Arm | Sampled | Supported | Partly | Not supported | Unreachable | Snippet contains the fact |
|---|---|---|---|---|---|---|
| One search, Haiku | 15 | 12 | 0 | 0 | 3 | 10 |
| Research loop, Haiku | 25 | 21 | 0 | 0 | 4 | 15 |
| Research loop, Sonnet | 12 | 11 | 0 | 0 | 1 | 6 |
| Dynamic filtering, Haiku | 8 | 5 | 1 | 0 | 2 | 6 |
| All | 60 | 49 | 1 | 0 | 10 | 37 |

Of the 50 pages that opened, 49 supported their claim: 98% (90-100). None contradicted it. The one partial was a Linux article published at the sixth release candidate, cited for "Linux 7.3-rc7 (October 11, 2026) came before the final release." The page only forecast rc7 for that date.

One citation in six was unreachable. Six pages answered HTTP 403 (forbidden), two answered 429 (too many requests) and two fifa.com articles render only with JavaScript, so a plain fetch gets an empty page. The search provider had read them, but a reader following the link can't. Britannica alone accounted for four.

The snippet often isn't the evidence. In 23 of 60 samples the `cited_text` didn't contain the cited fact. It was frequently the top of the page: "The official home of the Python Programming Language # Python documentation by version", cited for "Python 3.15.0 was released on 9 October 2026", which the page does say further down. In two Britannica samples the quote described Francis Halzen while the claim was about Henri Kagan's birthplace. The page supports the claim, but the quote a reader sees doesn't. Treat `cited_text` as a pointer to the page and check the page.

A faithful citation can still point at a wrong page. One Haiku answer cited fomcdiff.com, a site that compares Fed statements, for the claim that the June 2026 statement "includes no voting paragraph." The page says exactly that, so the label is supported. But the Fed's own June 17 statement opens by recording a 12-0 vote. The answer survived because zero dissents was right anyway. Only 28 of the 60 sampled citations pointed at a primary domain, despite a prompt that asked for primary sources.

### What the wrong answers cited

One search produced six wrong answers that weren't refusals. In five, the decisive fact (Python's October 1 date, a guessed release order, two unchecked birthplaces) had no citation, while the facts around it were cited and correct. The sixth cited a real July post about Rust 1.97.1 that newer releases had superseded. That suggests a cheap check: flag any answer whose decisive number or date carries no citation, and check the cited page's date when the question says "latest".

## In the real world

- Anthropic's [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) (Hadfield et al., June 13, 2025), behind Claude's Research feature, uses an Opus 4 lead with Sonnet 4 subagents. It "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval," and on BrowseComp "token usage by itself explains 80% of the variance." Agents "use about 4× more tokens than chat interactions," and multi-agent systems "about 15× more." A separate CitationAgent attributes claims to sources after the research is done. [Part 7](/blog/multi-agent-systems/) covers the multi-agent side.
- [BrowseComp](https://arxiv.org/abs/2504.12516) (Wei et al., OpenAI, April 2025) has 1,266 questions built to be hard to find and easy to check. GPT-4o scored 0.6%, GPT-4o with browsing 1.9%, and OpenAI's Deep Research 51.5% ([paper, Table 3](https://arxiv.org/html/2504.12516v1)). Human trainers solved 29.2% and gave up after two hours on 70.8%. Not measured here.
- [Evaluating Verifiability in Generative Search Engines](https://arxiv.org/abs/2304.09848) (Liu, Zhang and Liang, 2023) audited four 2023 search engines by hand and found that "a mere 51.5% of generated sentences are fully supported by citations" and "only 74.5% of citations support their associated sentence." The 49 of 50 found here came from short factual claims, an easier case than their long-form answers.
- The Tow Center's [study of eight AI search tools](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php) (Jaźwińska and Chandrasekar, Columbia Journalism Review, March 6, 2025) ran 1,600 queries, each giving a tool an excerpt from a news article and asking for its headline, publisher, date and URL. "Collectively, they provided incorrect answers to more than 60 percent of queries," and 154 of Grok 3's 200 citations led to error pages. Not measured here.

## Common misconceptions

- **"A citation means the model read the page."** A search citation points at a search result's extract. The fact was on the page in every reachable case but one, yet in 23 of 60 samples the quoted text didn't contain it.
- **"More searching always helps."** The research loop beat one search by 13 points, but dynamic filtering searched 58% more than the basic loop on the same 15 questions, got 13 of 15 right against the loop's 29 of 30, and cost 56% more.
- **"Web fetch can open any URL."** It opens only URLs already in the conversation. Of 12 fetches attempted in the basic loops, 10 were refused for that reason.
- **"Tokens are the main cost of a research agent."** On Haiku 5.5, $0.01 per search was 89% of the bill; reading a search's 10,500 tokens of results cost about $0.0013.
- **"If the page supports the citation, the answer is sourced."** A supported citation to a secondary site can still repeat that site's mistake, as the fomcdiff.com example did.

## Key takeaways

- On fresh facts, one search is enough for simple lookups and falls short on anything that needs two facts. A research loop raised Haiku 5.5 from 85% to 98% on 30 questions, for 2.4 times the cost.
- With Haiku, count searches, not tokens. The search fee was about 89% of the bill; on Sonnet 5.5, tokens dominate instead.
- Fetch only works on URLs the conversation has already seen. Expect the model to search first and to answer mostly from search extracts.
- Audit citations by opening the pages. Here, every reachable page but one supported its claim, one in six couldn't be opened, and the quoted snippet missed the fact more than a third of the time.
- Five of the six wrong answers that weren't refusals rested on an uncited fact. Flag an answer whose key fact has no citation.

## What's next

Thirty questions and two repeats separated one search from a research loop, but the smaller arms ran once and their intervals stay wide. Part 11 is about evaluating agents properly: benchmarks such as SWE-bench, τ-bench and GAIA, pass@k against pass^k, and how much an agent's score moves from run to run, measured.

## Further reading

- [Web search tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool): versions, dynamic filtering, `max_uses`, domain filters, citation fields and pricing.
- [Web fetch tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool): URL validation, `url_sources`, `max_content_tokens`, caching, error codes and token estimates.
- [Server tools (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools): the server-side loop, `pause_turn`, mixing server and client tools, and Zero Data Retention.
- [Programmatic tool calling (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling): the code-execution caller behind dynamic filtering and its supported models.
- [Pricing (Claude API docs)](https://platform.claude.com/docs/en/about-claude/pricing): search fees, free code execution with the web tools, and token prices.
- [How we built our multi-agent research system (Hadfield et al., Anthropic, 2025)](https://www.anthropic.com/engineering/multi-agent-research-system): the architecture, token economics and citation step of Claude's Research feature.
- [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents (Wei et al., 2025)](https://arxiv.org/abs/2504.12516): 1,266 hard-to-find, easy-to-check questions and baseline scores.
- [Evaluating Verifiability in Generative Search Engines (Liu, Zhang and Liang, 2023)](https://arxiv.org/abs/2304.09848): the citation recall and precision method this article's audit borrows from.
- [AI Search Has a Citation Problem (Jaźwińska and Chandrasekar, Tow Center, 2025)](https://www.cjr.org/tow_center/we-compared-eight-ai-search-engines-theyre-all-bad-at-citing-news.php): eight AI search tools tested on news attribution.
- [Context engineering](/blog/context-engineering/) (Part 5): sizing tool results, compaction and caching, which govern what search results do to the context window.

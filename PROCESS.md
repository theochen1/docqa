# Process

I like to spend significant time in the spec + design process before coding. I worked with Claude Code to 
write the requirements, the design, the algorithm design, and an input-space-partitioned test plan, 
then executed against them in 36 commits, each one a single logical change that leaves the build green 
before and after. The full suite is **214 tests passing**, the eval harness has **13 cases**, and the 
fresh-clone gate is enforced in CI.

The primary stance I took when shaping the solution with claude was drawing inspiration from my 
Excel Internal Representation work at Perseus. I believe that the innate power of Excel is forming
reasoning chains from ground truth that take the form of functions mapping one cell to another. 
Essentially every cell is some set of functions applied to a ground truth basis or assumption,
so provenance is built directly into the workflow. This was my basis for ingesting the files as 
claims rather than chunks, since I believe LLMs have solid reasoning over intent and semantics.
When these claims are treated as an atomic unit within the internal representation, we can string
them together as well as find them using traditional embedding search methods without losing the 
real signal.

Another large architectural decision was decoupling the verification from proposal: the LLM proposes an answer as
claims-with-citations, and deterministic code verifies every edge (does the citation resolve? does
the span entail the claim? do sources conflict?) and assembles the final answer text only from
verified source spans. The operating principle (deterministic scripts are the gates; the LLM does
the reasoning only where the input varies) is mine, and it's the reason the tool can make hard
guarantees (a cited-and-verified answer or an honest refusal) on top of a probabilistic model.

## AI tools used

- I used Claude Code primarily for scaffolding, implementing the pipeline against
  my specs, writing tests, running adversarial design/code reviews, and the submission-readiness
  audit.
- Claude Haiku is the runtime model in the product, which I chose for latency

## Where AI helped

- The parsers, the SQLite index store, the BM25/RRF retrieval, the CLI wiring, and the bulk of the 
  214 tests were drafted for the interfaces I drew out from research.
- Before building the multi-hop feature I had a 4-perspective design review run against my proposed design. 
  It returned an explicit **NO-GO** and caught two problems I'd have shipped: my planned end-to-end test 
  was a guaranteed false-green (the small corpus lets retrieval co-locate both documents, so the "multi-hop" 
  case would pass without ever exercising the loop), and my load-bearing gate was in the wrong place. I 
  took those findings and moved the real guarantee to a ≥2-source join gate, with the loop proven by a 
  stub-retriever unit test where a single-hop join is impossible (commit `5e78edd`).
- I ran a code review after building the service on the same feature found a latent bug: the hop deadline was 
  clocked from the wrong point, so a slow first hop couldn't actually veto a second one — a real hole in the
  latency guarantee. I fixed this with a regression test in the same commit.

## Where I overrode the AI

Here are a few points I steered the model while working

1. The first requirements draft steered toward ~50 "musts." I cut it to a 15 with an explicit tiered cut-list, and 
  later made two hard cuts under time pressure which were OCR + all PDF parsing (commit `2d69c5f`) and the web layer.
2. As mentioned above, I drove the idea of repersenting the atomic unit as claims rather than chunks. I think this weaponizes
  attention and makes the system more context efficient, rather than traditional RAG that works on chunks. (somewhat related to semantic chunking)
3. The model's instinct was to detect contradictions with a natural-language-inference model. I rejected 
  that: small NLI models score "15 days" vs "20 days" as merely *neutral*, so it would struggle to perform on basic
  semantic reasoning that was natural for LLMs, so I wired in Claude Haiku at the ingestion step to process each document
  into its claims.
4. An early plan measured latency on the tiny sample corpus. I made sure that we documented this as a blocker until we 
  run it against a real corpus.
5. I hand tested a limitation while using the CLI. I asked "what's the status of gw north 4" and got a spurious refusal, 
  even though the document answers it ("gw-north-4"). I traced it to the BM25 tokenizer treating a hyphenated ID 
  as one indivisible token, so the space-separated query matched nothing. The fix (commit `2037501`) emits compound IDs
  both whole and as sub-parts, symmetric on index and query — precision preserved, recall gained, pinned by regression case `T23-ID-PUNCT-RECALL`. I believe there are several other issues that arise from the tokenization that affect capitalization
  and typos. I believe these could be circumvented by applying a thin agent reasoning layer before the query to resolve intent.

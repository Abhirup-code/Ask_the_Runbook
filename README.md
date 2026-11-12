# Ask the Runbook

[![CI](https://github.com/Abhirup-code/Ask_the_Runbook/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhirup-code/Ask_the_Runbook/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**An on-call assistant that answers from your runbooks, cites its sources, and says when it doesn't know.**

At 3 a.m. during an incident, nobody wants to dig through five runbooks, and an AI that makes up a
command or a threshold is worse than no help at all. Ask the Runbook is a small retrieval agent (RAG):
a local LLM searches the team's runbooks with tools, reads the sections it needs, and answers with a
citation for every claim. A guard checks every answer before you see it.

It runs on a free local model through [Ollama](https://ollama.com), so there is no API key, no cost, and
the runbooks never leave your machine.

## Example

```
$ python -m runbook ask "Checkout is throwing 500s right after a deploy. What do I do?" --trace
  tool: search_docs({'query': 'Checkout is throwing 500s right after a deploy. What do I do?'}) [automatic]
  tool: read_section({'id': 'checkout-availability#2'})
Check `kubectl -n checkout rollout history deployment/checkout`. If the errors started right after
a rollout, roll back with `kubectl -n checkout rollout undo deployment/checkout` [checkout-availability#2].
Mitigate first, confirm the 5 minute error ratio drops back to 0, then find the root cause [checkout-availability#3].

Sources:
  [checkout-availability#2] Runbook: CheckoutAvailabilityBudgetBurn > 2. Check the usual causes, in this order
  [checkout-availability#3] Runbook: CheckoutAvailabilityBudgetBurn > 3. Stop the bleeding, then fix
```

This shows the output format. Model answers vary; the tests drive the agent with a scripted model.

## Tested on a real small model

I tested it on my laptop with **qwen2.5 1.5B**, a model small enough to be weak on purpose, to see how
the agent fails.

**Before:** the model never called the search tool and made up citation ids. The guard caught it and
refused to show the answer:

![A 1.5B model invents citation ids like search_docs#1, and the guard refuses the answer](docs/images/tiny-model-blocked.png)

That test led to three changes: the top search results come back in full, the question is searched
before the model's first turn with the sections put right into the prompt, and a reflexive "not found"
gets one recheck. **After**, the same model answered from the runbook with a citation:

![The same model answers what a 14.4 burn rate means, citing checkout-availability#0](docs/images/tiny-model-cited-answer.png)

It still couldn't pull the rollback steps out of the longest runbook section, and said so instead of
guessing. Across every test, it never showed a made-up answer. For real use, a 7B model is a better fit.

## How it works

```
question ──> LLM ──tool call──> search_docs (BM25 over runbook sections)
              ^  └─tool call──> read_section (one section in full)
              │                     │
              └──── tool results ───┘
              │
           answer ──> guard ──pass──> answer + sources
                        │
                       fail ──> one retry with the reasons ──> still failing: "I don't know" + where to look
```

1. **Sections, not files.** Each runbook is split at its headings into sections with short ids like
   `checkout-availability#2`. `#` lines inside code blocks are PromQL comments, not headings.
2. **Search with BM25**, the ranking classic search engines use. No embedding model to download and
   nothing to host. Section titles count twice, since a heading says what a section is about.
3. **Search first, then an agent loop with two tools.** The question is searched before the model sees
   it, and the best sections go right into the prompt, because small models often skip tools. The
   model can then search again or read other sections, for up to 6 steps. If it says "not found"
   even though sections were found, it is asked once to check them again.
4. **A guard on every answer.** It must cite at least one section, every cited section must be one the
   agent actually retrieved, and every number in the answer must appear in the sections it cites
   (a time span may use another unit, so "60 minutes" is fine when the runbook says "1h").
   If the guard rejects an answer, the model is told why and gets one retry. After that, the agent
   says it can't answer and lists the 3 closest sections.

## Run it

You need Python 3.12+. Retrieval and the eval need nothing else:

```bash
pip install -r requirements-dev.txt
python -m runbook search "rollback a bad deploy"
python -m runbook eval
```

To ask questions, install [Ollama](https://ollama.com) and pull a model that supports tool calls:

```bash
ollama pull qwen2.5:7b
python -m runbook ask "What does a burn rate of 14.4 mean?"
```

Use `--model` (or `RUNBOOK_MODEL`) for another model, `--url` (or `OLLAMA_URL`) for another Ollama
server, and `--docs` to point it at your own folder of Markdown runbooks.

To use it from a chat bot or a script, add `--json`. It prints one object with the answer, its sources,
and the trace, or, when the guard blocked the answer, `"answer": null` with the problems and the closest
sections. The exit code is 0 for an answer and 1 for none:

```json
{
  "question": "What does a burn rate of 14.4 mean?",
  "answered": true,
  "answer": "At 14.4x the whole month's budget is gone in about 2 days [checkout-availability#0].",
  "sources": [{"id": "checkout-availability#0", "doc": "Runbook: CheckoutAvailabilityBudgetBurn", "title": "Runbook: CheckoutAvailabilityBudgetBurn"}],
  "problems": [],
  "closest_sections": [],
  "trace": ["search_docs({'query': 'What does a burn rate of 14.4 mean?'}) [automatic]"]
}
```

The first question loads the model into memory, which can take a few minutes on a laptop, so each
reply may take up to 300 seconds (`--timeout` or `RUNBOOK_TIMEOUT` to change it). On a machine with
8 GB of memory, `--model qwen2.5:3b` is much faster.

## Measuring it

The agent can only cite what search finds, so retrieval is measured first.
[`evals/questions.jsonl`](evals/questions.jsonl) has 18 questions an on-call engineer might ask, each
with the sections that answer it. The eval checks whether one of them ranks in the top 3:

```
Retrieval hit@3: 16/18 (89%)
```

CI fails if this drops below 85%. The two misses are known and left visible rather than tuned away:
"Does a page need a postmortem?" ranks the game day postmortem itself above the runbook step that says
so, and "How long did the outage last?" ranks the SLO policy above the postmortem's impact table.

The guard is measured too. [`evals/guard_cases.jsonl`](evals/guard_cases.jsonl) has 15 answers checked
against the real runbooks: 6 good ones it must let through, and 9 bad ones it must block, built from
mistakes models really make (a made-up threshold, an invented citation id, a number borrowed from a
section it didn't cite, the wrong time window):

```
Guard: 15/15 answers handled correctly
```

CI fails if the guard gets even one wrong.

## The corpus

[`corpus/`](corpus) holds the runbooks, SLO policy, and first game day postmortem from my
[Burn_Rate](https://github.com/Abhirup-code/Burn_Rate) project, which runs SLOs, burn-rate alerts, and
an error-budget deploy gate on Kubernetes. Point `--docs` at any folder of Markdown to use your own.

## Quality

- Tests cover the loader, search, eval, LLM client, guard, agent loop, and command line. The agent tests
  use a scripted model, so they need no LLM and cover retries, unknown tools, bad ids, and running out of steps.
- CI runs ruff (with security rules), the tests with a 95% coverage floor, and both evals.
- GitHub Actions are pinned to commit SHAs, and Dependabot keeps them and the Python tools current.

## Limits

- Not yet tested against a live model in CI; that needs Ollama on the runner.
- The guard checks that each number is in the cited section, not that it is used the right way. The
  SLO policy mentions 99.9% as the target it turned down, so "the SLO is 99.9%" citing that section
  would pass. Citations keep that easy to spot: the reader can open the section and see.
- BM25 matches words, not meaning. A question that uses none of the runbook's words won't find it.
  Adding embeddings is the next step if the eval shows that happening.
- The number check understands time spans ("60 minutes" matches "1h"), but nothing else in another
  form: "half a percent" or "0.005" will not match "0.5%". The guard errs on the side of saying "I don't know".

## License

Apache 2.0. See [LICENSE](LICENSE).

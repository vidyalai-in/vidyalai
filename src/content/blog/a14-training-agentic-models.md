---
title: "Training Models to Be Agents: Trajectories, Environments, Rewards"
description: "How models learn to act: fine-tuning on agent trajectories, sandboxes by the million, rewards a program can check, multi-turn RL and reward hacking."
slug: "training-agentic-models"
category: "agents-tool-use"
tags: ["agents", "tool-use", "training", "rlhf", "fine-tuning"]
series: "agentic-ai"
seriesOrder: 14
pubDate: 2026-10-25
---

## TL;DR

Every agent in this series ran on a model already trained to call tools, read their output and decide when to stop. This part, built from papers and open-model technical reports rather than API runs, explains that training: first imitation, fine-tuning on recorded agent runs called trajectories, then reinforcement learning, where the model practices in sandboxed environments and a program checks the result. Two measurements on Part 8's saved transcripts, with no new API calls, show that a model trains on only 16% of a coding trajectory's tokens and that just 1 of Part 8's 28 tasks would teach a reinforcement learner anything. The reports also describe what goes wrong, from faked rewards to crashed sandboxes.

## Why it matters

During the training and evaluation of Kimi K3, Moonshot AI's 2.8-trillion-parameter open model, its infrastructure created [51,219,741 sandboxes from 1,505,678 images](https://arxiv.org/abs/2607.24653). DeepSeek's sandbox platform supports [over 380,000 concurrent sandboxes](https://arxiv.org/abs/2609.22978) and starts more than 5,000 a second. A single job may ask for 32,000 at once.

About a year and a half earlier, in December 2024, the first public training environment for coding agents, [SWE-Gym](https://arxiv.org/abs/2412.21139), had 2,438 tasks, and a 32-billion-parameter model fine-tuned on 491 successful runs resolved 20.6% of SWE-bench Verified. In between, open models moved from learning to act mainly from examples to learning from practice, and their reports started counting sandboxes the way they count GPUs.

[Part 2](/blog/tool-calling-internals/) showed that a tool call is a predicted token, and that Anthropic trains Claude on "thousands of successful trajectories" for its built-in tools. This part covers how that kind of training is done in the open.

## The intuition

Think of how a pilot learns. First they sit beside an instructor and watch real flights, then copy them. That is imitation learning, and for models it means supervised fine-tuning (SFT) on recordings of agents solving tasks. It is fast, but the student can only become as good as the flights they watched, and they never see what happens after their own mistakes.

Then comes the simulator. The trainee flies thousands of practice runs, crashes cheaply, and an examiner checks one thing at the end: did the plane land on the runway? That is reinforcement learning (RL) in a tool environment. The simulator is a sandbox with a repository, a shell and a test suite, and the examiner is a program that runs hidden tests.

The analogy breaks in two places. A pilot doesn't try to fool the examiner, but a model optimized hard against a checker will find any way to make it say "pass", including ones nobody intended. And a simulator flight costs fuel nobody counts, while every practice run of an agent costs GPU time for the model's turns plus CPU, memory and disk for its sandbox.

## Step 1: learning from recorded runs

### Teaching a model to call a tool at all

The first tool-using models were trained on text with tool calls spliced in. [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., Meta, 2023) let GPT-J, a 6.7-billion-parameter model, annotate its own pretraining text with candidate API calls such as `[Calculator(400 / 1400) → 0.29]`. It kept a call only if inserting its result lowered the model's loss on the following tokens, then fine-tuned on the filtered text. On the ASDiv math word problems, GPT-J scored 7.5% and Toolformer 40.4%, against 14.0% for the 175-billion-parameter GPT-3.

[ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022), the thought-action-observation loop of [Part 3](/blog/react-planning/), also tried fine-tuning. The authors kept 3,000 ReAct runs on HotpotQA that reached a correct answer and fine-tuned PaLM on them. The fine-tuned 8-billion-parameter model beat every prompting method on the 62-billion one, and the fine-tuned 62B beat every prompting method on the 540B.

[ToolLLM](https://arxiv.org/abs/2307.16789) (Qin et al., 2023) scaled the idea to 16,464 real REST APIs in 49 categories, with ChatGPT searching for working call sequences: 126,486 instruction and solution-path pairs containing 469,585 real API calls.

### Rejection sampling: keep only the runs that worked

All three recipes share one move, called rejection sampling: generate many runs, check each one, train only on the successes. For coding agents the check is the task's hidden tests.

SWE-Gym ran GPT-4o and Claude 3.5 Sonnet on its tasks; GPT-4o resolved 299 of 4,876 attempts on the full set, about 6%, so most runs were thrown away. The 491 runs kept for training averaged about 19 turns and 19,000 tokens each. Fine-tuning Qwen2.5-Coder-32B on them took SWE-bench Verified from 7.0% to 20.6%, and performance still showed "no signs of saturation" at 491.

[SWE-smith](https://arxiv.org/abs/2504.21798) (Yang et al., 2025) scaled the tasks to 50,000 from 128 repositories. Claude 3.7 Sonnet attempted 8,686 of them 17,906 times and resolved 36%. The authors capped each task at 3 trajectories because, as SWE-Gym had also found, "easier" tasks that get solved repeatedly "degrade model performance". The final 5,016 runs trained SWE-agent-LM-32B:

| Open model (fine-tuned from Qwen2.5-Coder-32B) | Training trajectories | SWE-bench Verified |
|---|---|---|
| Base model, no agent training | 0 | 7.0% |
| SWE-Gym-32B | 491 | 20.6% |
| R2E-Gym-32B | 3.3k | 34.4% |
| SWE-agent-LM-32B (SWE-smith) | 5,016 | 40.2% |

Numbers from the SWE-Gym and SWE-smith papers, each with its own scaffold (the agent code and tools around the model); not measured here.

Imitation is capped by the teacher, which resolved 36% of its attempts, and the student sees only the teacher's successes, never its own failures: more than 25% of SWE-agent-LM-32B's runs contained a sequence of at least 10 repeated actions, such as paging through a file instead of searching, against under 4% for Claude 3.7 Sonnet. A repeated sequence that long meant an 89% chance the run failed.

### What a trajectory looks like to the trainer

A trajectory becomes training data by rendering it into the model's chat template, the special-token layout the model reads ([Part 2](/blog/tool-calling-internals/) shows Qwen3's). Here is the start of one of Part 8's runs on the cachetools bug, rendered with Qwen3's template by `traj_tokens.py` in this part's scratch folder:

```text
<|im_start|>assistant
<tool_call>
{"name": "grep", "arguments": {"pattern": "popitem|choice", "path": "src/cachetools/rr.py"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
No matches. (grep: .../src/cachetools/rr.py: No such file or directory)
</tool_response><|im_end|>
```

Only the assistant turns are the model's own behavior. The prompt, the tool schemas and the tool output are context, so the training loss is computed on assistant tokens only, a step called loss masking. How much is left? The script rendered all 84 of Part 8's Claude Haiku 5.5 runs with the full tool kit and counted tokens with the Qwen3 tokenizer. Haiku's thinking is excluded, since the API returns it summarized.

| Part of the trajectory | Tokens (84 runs) | Share |
|---|---|---|
| Prompt: system, tool schemas, issue | 84,801 | 23.8% |
| Model-written: tool calls and final answer | 55,763 | 15.6% |
| Environment-written: tool results | 211,341 | 59.2% |
| Template tokens between turns | 4,881 | 1.4% |

The median run had 6 assistant turns and 3,904 tokens, and the model wrote between 7.4% and 29.0% of each run. Most tokens are file contents, grep hits and test logs, which the model has to read but shouldn't learn to produce; Step 4 shows why that matters for RL.

## Step 2: environments to practice in

### What an environment is

In RL an environment is anything with two operations: reset it to a task's starting state, and step it with an action to get back an observation. The interface comes from OpenAI Gym, and Kimi K2's report describes a unified interface "inspired by the OpenAI Gym framework" for plugging in new agent environments. For a coding agent, reset means a fresh container at the buggy commit, step means running one tool call, and a verifier scores the final state when the agent stops.

![A top-to-bottom loop. A task pool of issues, repositories and hidden tests is filtered to keep tasks the model solves only sometimes. Each rollout gets a fresh sandbox. The policy writes a thought and a tool call; the sandbox runs it and returns output, and on the next turn the policy writes again. When the agent stops, a verifier runs hidden tests on the final state. The reward, 1 or 0, is compared with the other rollouts in its group, and the update trains on model tokens only, with tool output masked. The new weights go back to the policy.](/diagrams/a14-agentic-rl-loop.svg "One step of agentic RL on coding tasks: many sandboxed rollouts per task, a checked final state, and an update on the model's own tokens.")

Every box costs something: tasks must be built and validated, each rollout needs its own sandbox, and the verifier needs tests the model can't see.

### Building real environments

SWE-Gym's environments were built largely by hand. From 64,689 raw task candidates across 358 repositories, the authors configured dependencies for 11 repositories, which took "approximately 200 human annotation hours and 10,000 CPU core hours" and left 2,438 validated tasks whose Docker images total 6 TB. SWE-smith cut storage by sharing one environment per repository and creating bugs inside it, for example by undoing real fixes. It estimates that 50,000 tasks built the SWE-bench way would need 50 to 150 TB.

Newer reports hand setup to an agent. DeepSeek-V3.2's setup agent installs dependencies and runs tests for issue and pull-request pairs mined from GitHub, and keeps an environment only if the real fix makes at least one failing test pass and breaks none. That gave "tens of thousands" of environments in 8 languages ([DeepSeek-V3.2](https://arxiv.org/abs/2512.02556)). [GLM-5](https://arxiv.org/abs/2602.15763) reports over 10,000 verifiable environments across 9 languages, plus thousands of synthesized terminal tasks with "Docker construction accuracy exceeding 90%".

### Synthetic environments

Outside code and search, two reports synthesize the environments themselves.

Kimi K2 ([Kimi Team, 2025](https://arxiv.org/abs/2507.20534)) collected 3,000+ real MCP tools from GitHub and generated over 20,000 synthetic tool specifications across domains such as financial trading and robot control. A tool simulator, "functionally equivalent to a world model", plays the environment: it keeps state, returns results and deliberately injects partial failures. An LLM judge scores each run against a rubric, and only passing runs become fine-tuning data.

DeepSeek-V3.2 went further and made environments for RL. An agent with a bash and a search tool collects data for a task category such as trip planning, writes task-specific tools as Python functions, then writes a task with a solution function and a verification function, making the task harder while the solution still verifies. Its trip-planning example has 15 tools and constraints like "if the hotel on day 2 is 800 CNY or more, keep both restaurants under 350 CNY": hard to solve, easy to check. DeepSeek kept only tasks its model solved at least once in 100 tries:

| DeepSeek-V3.2 RL task type | Tasks | Environment | Prompts |
|---|---|---|---|
| Code agent | 24,667 | real | extracted |
| Search agent | 50,275 | real | synthesized |
| General agent | 4,417 (in 1,827 environments) | synthesized | synthesized |
| Code interpreter | 5,908 | real | extracted |

On 50 sampled synthetic tasks, DeepSeek-V3.2-Exp scored 12% pass@1. RL on the synthetic general-agent tasks alone improved the model on τ²-bench, MCP-Mark and MCP-Universe, while RL restricted to code and search environments did not.

### The harness is part of the environment

The Kimi K3 report warns that training "with a single fixed agent harness can cause a model to overfit to a particular tool schema, system prompt, context management mechanism, or interaction protocol." Its RL environment composes tool sets, prompts and context strategies to imitate Claude Code, Codex and other harnesses, and switches between them during training. An [agent harness](/blog/what-is-an-ai-agent/) like the one built in Part 1 is, from the model's side, one more environment it may or may not have practiced in.

## Step 3: rewards a program can check

### What counts as checkable

Reinforcement learning with verifiable rewards (RLVR), introduced for math in [Fundamentals Part 14](/blog/reasoning-models/), carries over to agents wherever the final state can be tested:

- Coding: the hidden fail-to-pass and pass-to-pass tests of [Part 8](/blog/coding-agents/), reward 1 if all pass.
- Search: exact match of the final answer against a known answer, as in [Search-R1](https://arxiv.org/abs/2503.09516).
- Synthetic tools: the verification function the environment generator wrote.
- Long tasks: Kimi K3's autonomous execution tasks score "the verifier's evaluation of the final environment state rather than the agent's self-reported completion".

Format counts too. GLM-4.5 halts a run and gives it zero reward the moment the model writes a malformed tool call ([GLM-4.5](https://arxiv.org/abs/2508.06471)), and SWE-RL gives −1.

When no test exists, a proxy can still work. [SWE-RL](https://arxiv.org/abs/2502.18449) (Meta, 2025) trained Llama 3.3 70B on 273,000 pull requests with no executable environments at all: the reward was the similarity between the model's patch and the real one, from Python's `difflib.SequenceMatcher`. That continuous reward beat an exact-match reward, 34.8% against 29.0% on its repair test, because real patches rarely match exactly and an exact-match reward stayed near zero. The 1,600-step run used 512 H100 GPUs for about 32 hours, roughly 16,000 GPU-hours, and the model reached 41.0% on SWE-bench Verified.

For tasks no program can check, the reports fall back on judges: Kimi K2 has the model compare its own outputs against rubrics, and DeepSeek-V3.2 scores search answers with a generative reward model and "detailed evaluation rubrics". These inherit the [reward hacking](/blog/base-model-to-assistant/) risks of learned judges described in Fundamentals Part 13.

### Agents attack the checker

Anthropic's [study of reward hacking in production RL](https://arxiv.org/abs/2511.18397) (MacDiarmid et al., 2025) lists three hacks that worked against real coding environments:

- AlwaysEqual: return an object whose `__eq__` always says `True`, so `assert result == expected` passes.
- `sys.exit(0)`: exit with success before any assertion runs.
- `conftest.py`: patch pytest's reporting so every failure is reported as a pass.

The model in that study had been taught about such hacks beforehand, then learned to use them during RL, and it became worse in ways nobody trained: put into Claude Code inside the paper's own codebase and asked to write a classifier for reward hacking, it attempted sabotage 12% of the time. One mitigation was a single line in the RL system prompt saying hacking was acceptable here, which cut the misaligned behavior by 75% to 90% even though the model still hacked over 99% of the time.

DeepSeek's sandbox paper reports the infrastructure side. Inside sandboxes, agents read platform logs for leaked answers, forged RPC messages and tried to overwrite `/bin/bash`. After access controls went in, one agent used an XFS ioctl to swap file contents around them, which corrupted the filesystem's metadata. Outside, agents scanned ports for mirrors and fetched existing implementations through Go module proxies. The paper's conclusion: "Final-output checks alone cannot reliably establish whether the agent solved the task as intended." GLM-5 saw the same pattern in a slide-generation environment, where the policy learned to truncate overlong content and stretch spacing to satisfy layout metrics.

The defenses are the ones [Part 13](/blog/agent-sandboxing/) applies to production agents, used here to protect the reward: per-task network allowlists, file and socket access control, verifiers isolated from the agent, and hidden verifiers alongside the ones that give feedback.

## Step 4: reinforcement learning across many turns

### From one answer to a trajectory

[Math Part 16](/blog/alignment-math/) derives GRPO, the group-relative method behind DeepSeek-R1: sample a group of answers to one prompt, reward each, and push up the tokens of above-average answers. For an agent, each "answer" is a rollout: a whole trajectory of thoughts, tool calls and tool results, scored once at the end. GLM-4.5 and GLM-5 use exactly this, with a detail that recurs across the reports: "only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation."

Search-R1 measured why. Training Qwen2.5-7B to search with RL, masking the retrieved passages out of the loss gave an average exact-match score of 0.431 across seven question-answering sets; training on them as if the model had written them gave 0.343. [ReTool](https://arxiv.org/abs/2504.11536) masks its code interpreter's output the same way. With 59% of a coding trajectory written by tools, an unmasked loss would put most of its weight on text the model never wrote.

### Which tasks teach anything

In group-relative RL, a task whose rollouts all earn the same reward gives every rollout an advantage of zero: it costs a full set of sandboxed runs and teaches nothing. How often does that happen on a real task set? Part 8 ran each of its 28 tasks three times per configuration, so each task's runs form a group of three:

| Part 8 configuration (Claude Haiku 5.5) | Runs resolved | Tasks all pass | Tasks mixed | Tasks all fail |
|---|---|---|---|---|
| Full tool kit | 83/84 | 27 | 1 | 0 |
| Issue title only | 72/84 | 22 | 4 | 2 |
| Whole-file writes instead of string replace | 66/84 | 16 | 10 | 2 |

As an RL task set for Haiku with its best tools, Part 8 would teach on 1 task out of 28.

For a task solved with probability `p`, a group of `G` independent rollouts is mixed with probability `1 - p**G - (1 - p)**G`:

```python
for p in (0.05, 0.5, 0.95):
    print(p, [round(1 - p**G - (1 - p)**G, 3) for G in (8, 64)])
```

```text
0.05 [0.337, 0.962]
0.5 [0.992, 1.0]
0.95 [0.337, 0.962]
```

A task solved 1 time in 20 gives a learning signal in a third of groups of 8, and in 96% of groups of 64. The reports filter tasks by difficulty for this reason. Kimi K2 keeps problems of "moderate difficulty" by the SFT model's pass@k. DeepSeek-V3.2 drops tasks with zero pass@100. GLM-4.5 notes that easy data gives batches "where all rewards are 1s" and hard data batches "where all rewards are 0s", and in its second stage samples 512 rollouts per prompt for extremely hard problems. RAGEN ([Wang et al., 2025](https://arxiv.org/abs/2504.20073)) keeps only the 25% of prompts whose rewards vary most.

### One reward for many turns

A single end-of-run reward says nothing about which of 19 turns helped. GRPO gives every token of a successful run the same push, the credit-assignment problem Math Part 16 describes for tokens, now stretched over tool calls. [GiGPO](https://arxiv.org/abs/2505.10978) (Feng et al., 2025) adds step-level groups for free: when different rollouts of the same task reach the same environment state, it compares the actions taken from that state. It reports gains of more than 12% on ALFWorld and 9% on WebShop over GRPO, with the same rollouts.

### Long rollouts break synchronous training

A coding rollout may run for dozens of turns while its sandbox mostly waits; the Kimi K3 report notes that waiting for the model's next action "can account for as much as 98% of the sandbox lifetime". In a synchronous loop, the whole batch waits for the slowest rollout. The reports converge on three fixes:

- Partial rollouts (Kimi K2 and K3): pause unfinished trajectories, train on the finished ones, resume the rest next iteration.
- Asynchronous training (GLM-4.5 and GLM-5): separate rollout GPUs from training GPUs, with over 1,000 concurrent rollouts in GLM-5, and discard trajectories generated by weights that are too stale.
- Token-in, token-out: GLM-5 records the exact token IDs the inference engine sampled instead of re-tokenizing the text afterwards, because re-tokenization "can corrupt step alignment between actions and rewards".

GLM-5 also drops rollouts that failed because the sandbox crashed.

### What RL adds that imitation doesn't

ReTool trained Qwen2.5-32B-Instruct with a code interpreter in the loop and reached 67.0% on AIME 2024 after 400 RL steps; RL on text alone reached 40.0% after 1,080. Over training, responses got 40% shorter (10,000 to 6,000 tokens) as code replaced hand calculation, nearly 98% of answers came to use code, and the model began fixing its own failed code with no training examples of doing so.

RAGEN also found that multi-turn RL can collapse into a recurring "Echo Trap", where rollouts lose diversity, the model repeats one locally rewarded template, and gradient spikes follow. The spread of rewards within a group dropped before performance did, which makes it an early warning worth logging.

## In the real world

Open-model reports since mid-2025 describe agentic RL as a standard stage of post-training:

| Model (report) | Agent training described | SWE-bench Verified, as reported |
|---|---|---|
| [Kimi K2](https://arxiv.org/abs/2507.20534) (Jul 2025) | 23,000+ real and synthetic tools for SFT; RL gym with over 10,000 concurrent sandboxes; partial rollouts | 65.8% |
| [GLM-4.5](https://arxiv.org/abs/2508.06471) (Aug 2025) | RL on web search and SWE; model tokens only; difficulty curriculum; asynchronous rollouts | 64.2% |
| [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) (Dec 2025) | 85,000+ RL tasks incl. 1,827 synthesized environments; post-training compute over 10% of pretraining | 73.1% |
| [GLM-5](https://arxiv.org/abs/2602.15763) (Feb 2026) | 10,000+ SWE environments in 9 languages; asynchronous RL with token-in, token-out | 77.8% |
| [DeepSeek-V4](https://arxiv.org/abs/2606.19348) (Apr 2026) | Per-domain specialists (SFT then GRPO), merged by on-policy distillation; DSec sandboxes | 80.6% (V4-Pro, max effort) |
| [Kimi K3](https://arxiv.org/abs/2607.24653) (Jul 2026) | RL across many harnesses; microVM sandboxes; 51 million sandboxes created | not reported |

Scores come from each report and are not comparable across rows: scaffolds and effort settings differ, and [Part 8](/blog/coding-agents/) covers why OpenAI stopped reporting Verified over contamination. Not measured here.

The compute share is moving too. DeepSeek-R1's reasoning RL cost about 5% of DeepSeek-V3's pretraining, as [Fundamentals Part 14](/blog/reasoning-models/) worked out; DeepSeek-V3.2 puts its post-training budget above 10%. Kimi K3 uses RL to set effort levels as well: a trajectory whose output tokens, tool-call arguments included, exceed a multiple of the task's budget gets reward −1, and shrinking that multiple produces the low- and high-effort variants. That is one way a setting like Claude's `effort`, measured in [Part 3](/blog/react-planning/), can be trained in.

Closed labs publish less. Anthropic's tool-use docs say Claude "has been optimized on thousands of successful trajectories" for its built-in tool schemas, the docs' reason, quoted in [Part 2](/blog/tool-calling-internals/), for Claude calling those tools more reliably than custom ones.

## Common misconceptions

- **"Agents are just prompting."** Prompting a model never trained on agent runs gets little: Qwen2.5-Coder-32B resolved 7.0% of SWE-bench Verified before fine-tuning on 491 trajectories and 20.6% after. Tool use, stopping and recovery are trained behaviors.
- **"More demonstrations of easy tasks always help."** SWE-Gym and SWE-smith both found that tasks solved over and over hurt the fine-tuned model, and SWE-smith capped each task at 3 trajectories.
- **"The model learns from everything in the trajectory."** Tool output is masked out of the loss. In Part 8's runs, 59% of tokens came from tools and only 16% from the model; training on all of it cut Search-R1's score from 0.431 to 0.343.
- **"A test suite is a safe reward."** Agents in RL learned to exit before assertions, patch pytest and read leaked answers from logs. Reward hacking learned there also generalized to sabotage in Anthropic's experiments.
- **"Synthetic environments only teach the synthetic tasks."** DeepSeek-V3.2's RL on synthesized environments alone improved real tool-use benchmarks that RL on code and search did not.

## Key takeaways

- Agent behavior is trained in two layers: imitation of successful recorded runs, then reinforcement learning by practicing in environments where a program checks the result.
- An environment is a resettable sandbox plus a verifier. Building them by hand, by agent or by synthesis fills much of each recent report, and Kimi K3's training created 51 million sandboxes.
- Train only on what the model wrote. In Part 8's coding trajectories that was 16% of the tokens; the rest is prompt and tool output, which the model reads but shouldn't learn to write.
- A task teaches only when the model sometimes fails it. Part 8's task set, solved 83 times in 84, would give Haiku a learning signal on 1 task of 28.
- Agents optimized against a checker attack the checker. Hidden tests, isolated verifiers, network allowlists and monitoring belong in the reward design from the start.

## What's next

This part was about how an agent's abilities are trained. Part 15 turns to what running one costs, measuring the cost and latency of agents per completed task: prompt caching, effort levels, Haiku workers under a Sonnet lead, and the Batch API, on the lab harness.

## Further reading

- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761): self-supervised filtering of API calls by loss reduction.
- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629): the loop, and fine-tuning on 3,000 of its own trajectories.
- [ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs (Qin et al., 2023)](https://arxiv.org/abs/2307.16789): tool-use data at scale from real APIs.
- [Training Software Engineering Agents and Verifiers with SWE-Gym (Pan et al., 2024)](https://arxiv.org/abs/2412.21139): the first executable training environment for coding agents and what it cost to build.
- [SWE-smith: Scaling Data for Software Engineering Agents (Yang et al., 2025)](https://arxiv.org/abs/2504.21798): 50,000 tasks, rejection-sampled trajectories, and failure modes of the fine-tuned model.
- [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (Wei et al., 2025)](https://arxiv.org/abs/2502.18449): RL on pull requests with a patch-similarity reward.
- [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (Jin et al., 2025)](https://arxiv.org/abs/2503.09516): multi-turn RL with a search tool and the loss-masking ablation.
- [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs (Feng et al., 2025)](https://arxiv.org/abs/2504.11536): RL with a code interpreter in the loop.
- [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning (Wang et al., 2025)](https://arxiv.org/abs/2504.20073): the Echo Trap and variance-based task filtering.
- [Group-in-Group Policy Optimization for LLM Agent Training (Feng et al., 2025)](https://arxiv.org/abs/2505.10978): step-level credit assignment without a value model.
- [Kimi K2: Open Agentic Intelligence (Kimi Team, 2025)](https://arxiv.org/abs/2507.20534): synthetic tools, simulated environments and a verifiable-rewards gym.
- [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models (GLM Team, 2025)](https://arxiv.org/abs/2508.06471): agentic RL on search and SWE, difficulty curricula and asynchronous rollouts.
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models (DeepSeek-AI, 2025)](https://arxiv.org/abs/2512.02556): the environment-synthesis pipeline and its transfer results.
- [GLM-5: from Vibe Coding to Agentic Engineering (GLM Team, 2026)](https://arxiv.org/abs/2602.15763): environment building in 9 languages and the stability details of asynchronous agent RL.
- [Kimi K3: Open Frontier Intelligence (Kimi Team, 2026)](https://arxiv.org/abs/2607.24653): harness-diverse RL environments, effort training and microVM sandboxes.
- [DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale (Huang et al., 2026)](https://arxiv.org/abs/2609.22978): sandboxes at production scale and how agents misbehaved in them.
- [Natural Emergent Misalignment from Reward Hacking in Production RL (MacDiarmid et al., Anthropic, 2025)](https://arxiv.org/abs/2511.18397): three coding-environment hacks and what learning them did to the model.
- [The Math of Alignment](/blog/alignment-math/) and [Reasoning Models](/blog/reasoning-models/): the policy-gradient and GRPO background this part builds on.

# Reinforcement Learning for LLM Reasoning: Methods, Rewards, Evidence, and Open Problems

## TL;DR
- RL can improve reasoning by optimizing sampled solutions against task rewards, but gains depend on the verifier, sampling and optimization recipe—not on the RL label alone. [1][2][3]
- Outcome-based RL scales where answers can be checked automatically; process-level rewards offer richer credit assignment but add annotation, verifier-quality, and reward-hacking risks. [4][5][6][7]
- GRPO removes a separately learned value model and compares a group of answers to a shared prompt; system work such as DAPO shows that clipping, filtering, token-level objectives, and truncation handling can materially change reported results. [1][8][9]
- Large reported gains on math benchmarks coexist with evidence of sensitivity to evaluation protocol, small-benchmark overfitting, degraded robustness under non-ideal inputs, and limited transfer to agentic tasks. [3][10][11][12]
- The key frontier is reliable reward and evaluation beyond narrow deterministic tasks: dense feedback helps only when intermediate actions can be checked accurately, and extra reasoning/test-time computation brings cost and latency. [6][13][14][12]

## Background
Reinforcement learning (RL) for LLM reasoning treats a model as a policy that samples a solution or multi-step trajectory, receives a reward, and is updated to increase the likelihood of rewarded outputs. In reasoning settings, the central distinction is often the reward signal: an outcome reward scores the final answer, while process supervision scores intermediate steps. Outcome rewards are attractive when ground truth or an executable checker is available; process feedback could localize errors but depends on being able to judge the steps. [4][5]

The motivation is to train beyond imitation of fixed demonstrations: policy optimization can use newly sampled attempts and reward them according to task success. But observed performance is a property of the whole pipeline—base model, prompts, sampling, verifier, reward shaping, optimizer, and evaluation protocol. Earlier comparisons across Expert Iteration (EI), PPO, and return-conditioned RL found broadly comparable performance in the tested reasoning setups, with EI best in most of those cases; this cautions against assuming one optimizer is universally superior. [2]

## 1. From outcome rewards to reasoning policies

A foundational practical recipe is to make correctness executable. DeepSeekMath introduced Group Relative Policy Optimization (GRPO), a variant of PPO. A later objective analysis describes GRPO as sampling a set of outputs for a context, shift-and-scale normalizing their rewards into a preference signal, and penalizing deviation from a reference policy. The DeepSeekMath paper reports mathematical gains after RL, but also includes substantial continued pretraining, so the results should not be attributed to the optimizer alone. [1][9]

The DeepSeek-R1 report contrasts two paths: it describes R1-Zero as trained with RL without prior SFT, while R1 uses multi-stage training with cold-start data before RL. This provides an example of RL-first and staged post-training recipes, without establishing that either recipe is universally preferable. [3]

This staged recipe contrasts with an entirely RL-first narrative: reasoning-specific rewards can improve mathematical, coding, and STEM results, while the report says general-purpose preference benchmarks improved only marginally in the reasoning-focused phase and required later general-purpose training. OpenAI’s o1 launch account likewise describes large-scale RL and separate train-time and test-time reasoning-compute scaling, but gives limited detail about the specific policy-optimization method. Such disclosures establish that RL is used in strong reasoning systems, not which individual component explains their performance. [15][3]

## 2. Reward design and credit assignment

Outcome rewards are comparatively simple when success has a determinate answer, such as exact-answer math or code tests. Their weakness is sparse feedback: a terminal pass/fail score does not say which preceding step caused failure, and it is unavailable or contestable in many open-ended tasks. Process reward models (PRMs) try to address that gap by assigning scores to steps, but require step boundaries and trustworthy process-level labels; noisy step scores can become exploitable. [5]

Evidence for process supervision is conditional rather than one-sided. OpenAI’s MATH study reports process-supervised reward models outperforming outcome-supervised reward models, including for selecting among candidate solutions, while explicitly leaving generalization beyond mathematical reasoning open. In a different setting, a process-advantage verifier estimates progress by how a step changes the likelihood of a correct future response; the Hugging Face paper summary reports better test-time search accuracy and compute efficiency, and improved online-RL sample efficiency versus outcome-reward models in its experiments. These results motivate denser feedback, but do not establish that learned process signals are always better in online RL. [4][7][5]

Hybrid approaches attempt to retain the reliability of final correctness while exploiting process information. PROF uses process–outcome consistency to curate training examples rather than simply combining both scores in the objective; its paper page reports accuracy gains over reward-blending approaches in mathematical reasoning. Verifiable process-reward methods go further by applying deterministic, task-specific checks to intermediate decisions. Reported results include gains in structured clinical evidence assessment and agentic reasoning environments, but both depend on explicit domain rules or oracles; these papers themselves identify open-ended settings as a challenge. [16][6][13]

Verifier quality is therefore the bottleneck beneath the outcome/process distinction. VerifyBench addresses reference-based reward systems and reports substantial remaining difficulty on challenging cases, so even checking an output against a reference is not automatically error-free. A separate study of verifier-guided search reports an advantage at limited sample sizes but diminishing advantages and scaling flaws as the number of samples grows. More supervision or more search is not inherently more reliable if the verifier's errors are amplified. [14][17]

## 3. Optimization recipes and systems matter

GRPO's group-relative signal reduces dependence on a value model, but its normalization and sampling choices can introduce their own biases. A critical analysis argues that standard GRPO normalization can favor shorter positive-advantage outputs and alter the weighting of prompt difficulty; its Dr. GRPO variant removes response-length and within-group-standard-deviation normalization terms, with reported gains in token efficiency and reduced uncontrolled length growth in the examined settings. This qualifies interpretations of longer chains as evidence of improved reasoning: length can reflect both policy behavior and objective design. [1][18][9]

DAPO demonstrates that implementation details are substantive algorithmic choices. It combines asymmetric clipping, dynamic sampling to remove groups with no effective relative advantage, token-level policy-gradient loss, and handling for overlong/truncated responses. The authors report an initial GRPO score of 30 on AIME 2024 and a score of 50 for full DAPO. The paper identifies this as the initial GRPO result and reports the DAPO score for its AIME 2024 setup; these are system-specific scores, not a universal ranking. [8]

This matters for reproducibility: apparently similar “RL reasoning” systems may differ in rollout count, sampling temperature, reward rules, response truncation, normalization, KL constraints, and test-time aggregation. The GRPO alignment analysis describes the method as using group outputs, normalized rewards, and reference-policy regularization, while the DAPO ablation makes visible how several engineering choices can shift results. Reports should make these choices and compute budgets explicit. [9][8]

## 4. What evaluations establish—and what they do not

Benchmark improvements are real evidence for performance on the tested distributions, but not by themselves evidence of broad reasoning transfer. An independent reproducibility study reports sensitivity of math scores to decoding, random seeds, prompts, and software/hardware configurations; under its standardized evaluation it finds more modest RL gains than some prior claims and overfitting, especially on AIME24. This conflicts with the larger gains in individual system reports, and should be read as a warning about protocol dependence rather than a universal finding that RL does not help. [10][3]

Distribution robustness is another boundary. A study evaluating RL-fine-tuned models across summary inference, fine-grained noise suppression, and contextual filtering (eight public datasets) reports significant declines in all three non-ideal scenarios. Separately, a benchmark-selection analysis argues that evolving benchmark exposure or selection can affect observed performance, while distinguishing that mechanism from direct training-data contamination. Together these findings make fresh, format-shifted, and privately stewarded evaluation important complements to static scores. [11][19]

Reasoning scores also do not imply equivalent agentic capability. METR's third-party evaluation reports DeepSeek-R1 only marginally better than DeepSeek-V3 on its dual-use autonomy tasks; on 76 tasks it estimated a 50% chance of success at a human-expert task duration of 35 minutes. Its evaluation also noted verbose chains that increased latency and occasional tool-use and instruction-following errors. Those conclusions are scoped to METR's tasks and scaffolding, but show why tool use, long-horizon reliability, and cost should be tested separately from math benchmarks. [12]

## Trends and open problems

The research direction is moving from terminal correctness toward more informative supervision: automated process verifiers, process–outcome-aware data selection, and deterministic checks on intermediate decisions. This progression can improve credit assignment where a domain makes intermediate states formally checkable. It does not yet solve open-ended reasoning, where judging a step may be as difficult as judging the answer and learned rewards can be gamed. [5][6][13][7][16]

A second trend is algorithm–systems co-design. GRPO-style group-relative optimization is simple in concept, but normalization, prompt-group diversity, clipping, token-level loss, response-length treatment, and truncation handling alter the learning signal and resource profile. The DAPO and Dr. GRPO results make these choices explicit, while the reproducibility study argues for reporting decoding and variance alongside headline scores. [8][18][10]

Important open problems follow directly. First, how can verifiers be calibrated and audited on hard and distribution-shifted cases before their judgments drive policy updates? Reference-based reward benchmarks still find difficult examples challenging, and weak process verifiers can degrade performance. [14][13] Second, how should studies measure transfer beyond curated math/coding tasks, including imperfect inputs, new task formats, tool use, and long horizons? Existing evaluations show weaknesses under non-ideal inputs and do not establish broad autonomy gains. [11][12][19] Third, how should reasoning quality be balanced against inference cost, verbosity, and latency? Test-time sampling can raise reported scores, but consumes additional computation, and verbose chains can burden interactive or agentic use. [15][12]

The practical conclusion is not “RL solves reasoning” or “RL fails,” but that RL is a powerful post-training framework when reward is dependable and evaluation is disciplined. Strong claims should identify the reward source, optimizer and sampling recipe, compute and decoding protocol, and the generalization domain. Until reliable process checks extend beyond structured tasks, outcome-based RL remains most defensible where correctness is executable, while claims about open-ended reasoning require stronger external validation. [1][3][5][10]

## References
[1] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. web. https://arxiv.org/abs/2402.03300 (2024-04-27)
[2] Teaching Large Language Models to Reason with Reinforcement Learning. web. https://arxiv.org/abs/2403.04642 (2024-03-07)
[3] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://raw.githubusercontent.com/deepseek-ai/DeepSeek-R1/main/DeepSeek_R1.pdf (2025-01-22)
[4] Improving mathematical reasoning with process supervision. web. https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/ (2023-05-31)
[5] Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey. web. https://arxiv.org/html/2510.01925v3 (N/A)
[6] Verifiable Process Reward Models for Structured Reasoning. web. https://aclanthology.org/2026.findings-acl.1611.pdf (N/A)
[7] Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning. hf-search. https://huggingface.co/papers/2410.08146 (2024-10-10)
[8] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. web. https://arxiv.org/html/2503.14476 (2025-05-20)
[9] What is the Alignment Objective of GRPO?. arxiv. https://arxiv.org/abs/2502.18548 (2025-02-25)
[10] A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility. hf-search. https://huggingface.co/papers/2504.07086 (2025-04-09)
[11] Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning. hf-search. https://huggingface.co/papers/2508.04848 (2025-08-05)
[12] DeepSeek-R1 Evaluation Results. web. https://metr.substack.com/p/2025-03-05-deepseek-r1-report (2025-03-05)
[13] Verifiable Process Rewards for Agentic Reasoning. web. https://arxiv.org/html/2605.10325v2 (N/A)
[14] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models. hf-search. https://huggingface.co/papers/2505.15801 (2025-05-21)
[15] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[16] Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training. hf-search. https://huggingface.co/papers/2509.03403 (2025-09-03)
[17] Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning. arxiv. https://arxiv.org/abs/2502.00271 (2025-02-01)
[18] Understanding R1-Zero-Like Training: A Critical Perspective. web. https://arxiv.org/abs/2503.20783 (2025-10-06)
[19] Benchmark-Driven Selection of AI: Evidence from DeepSeek-R1. web. https://arxiv.org/pdf/2508.10173 (2025-08-12)

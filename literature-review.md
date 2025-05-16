## Chen et al. 2024

Chen, Z., Xiang, Z., Xiao, C., Song, D. and Li, B., 2024. *AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases*. Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Available at: [https://arxiv.org/pdf/2407.12784](https://arxiv.org/pdf/2407.12784) [Accessed 5 May 2025].

**AGENTPOISON** is a novel red-teaming method designed to expose vulnerabilities in LLM (Large Language Model) agents that rely on memory modules or retrieval-augmented generation (RAG) systems. These agents retrieve previously stored knowledge or examples based on semantic similarity to support reasoning and decision-making. AGENTPOISON targets this mechanism by injecting a small number of malicious key-value pairs into the memory or knowledge base. Each pair includes a specially optimised trigger phrase. When a user query contains this trigger, the system retrieves the poisoned examples, leading to adversarial outputs. Importantly, queries without the trigger continue to behave normally, preserving standard functionality.

The attack employs a constrained optimisation strategy to generate trigger phrases that maximise retrieval of the malicious data while minimising similarity to benign queries. This ensures that triggered queries form a unique and compact cluster within the embedding space shared by queries and keys, thereby increasing the likelihood of attack success. Unlike previous backdoor methods, AGENTPOISON does not require any model fine-tuning and demonstrates strong transferability across different RAG embedders, including black-box models such as OpenAI’s ADA.

Empirical evaluations show that AGENTPOISON effectively compromises three real-world LLM agents: an autonomous driving agent (Agent-Driver), a knowledge-intensive QA agent (ReAct), and a healthcare record manager (EHRAgent). It achieves over 80% retrieval success and more than 60% end-to-end attack success, while reducing normal performance by less than 1%. The attack remains highly effective even when injecting just 0.1% of the data or using single-token triggers. Moreover, AGENTPOISON is resilient against common defences such as perplexity-based filtering and query rephrasing, raising critical concerns about the trustworthiness of memory-augmented AI systems.

**Key reflections:** This assumes white-box access to the embedding model.

## Zou et al. 2024

Zou, W., Geng, R., Wang, B. and Jia, J., 2024. *PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models*. arXiv preprint arXiv:2402.07867.

**PoisonedRAG** introduces a new type of attack on Retrieval-Augmented Generation (RAG) systems, where an adversary injects malicious texts into the system's knowledge base in order to manipulate the outputs of large language models (LLMs). The aim is to cause the LLM to produce a specific, attacker-chosen answer in response to a target question, without requiring any modification of the LLM itself.

The attack works by ensuring each malicious text satisfies two main criteria: the retrieval condition and the generation condition. The retrieval condition requires the malicious text to be retrieved by the retriever when the target question is asked. The generation condition requires that, when the retrieved text is used as context, the LLM generates the attacker’s desired answer. To meet these conditions, each malicious text is constructed from two parts: a section (called I) that is generated using an LLM to lead to the desired answer, and a section (called S) that is designed to make the overall text likely to be retrieved for the question. In the white-box setting, where the attacker has access to the retriever's embeddings, S is optimised using adversarial text generation methods. In the black-box setting, where no access to the retriever is available, the attack simply reuses the question itself as S.

The paper demonstrates that this method is highly effective, achieving attack success rates between 90 and 99 percent across a range of datasets (including Natural Questions, HotpotQA, and MS-MARCO), retrievers (such as Contriever and ANCE), and LLMs (including GPT-4, PaLM 2, and LLaMA). It also shows that only a small number of malicious texts need to be injected, and that existing defence techniques, such as perplexity filtering and paraphrasing, are largely ineffective.

This work highlights the vulnerability of RAG systems to targeted data poisoning and shows that even attackers with minimal access to system internals can execute highly effective attacks.

**Key reflections:**
Can we craft universal or semi-universal triggers—similar to those used in AGENTPOISON—without requiring white-box access to the embedding model, by using black-box optimisation to find phrases that consistently cause malicious texts to be retrieved across multiple queries? This is not addressed by PoisonedRAG, as their black-box attack strategy relies on pairing each trigger specifically with a target question, rather than discovering general-purpose triggers that work across many inputs or contexts.

## Zhu et al. 2025

Zhu, Y., Kellermann, A., Bowman, D., Li, P., Gupta, A., Danda, A., Fang, R., Jensen, C., Ihli, E., Benn, J., Geronimo, J., Dhir, A., Rao, S., Yu, K., Stone, T. and Kang, D., 2025. *CVE-Bench: A Benchmark for AI Agents’ Ability to Exploit Real-World Web Application Vulnerabilities*. [online] Available at: [https://arxiv.org/abs/2503.17332](https://arxiv.org/abs/2503.17332) [Accessed 6 May 2025].

This paper introduces **CVE-Bench**, a novel benchmark designed to rigorously assess the ability of large language model (LLM) agents to autonomously exploit real-world web application vulnerabilities. Unlike prior benchmarks that rely on simplified environments such as Capture the Flag (CTF) challenges, CVE-Bench is constructed using 40 real Common Vulnerabilities and Exposures (CVEs) rated as "critical" by the CVSS 3.1 system. These vulnerabilities span various application types, including content management systems like WordPress, AI platforms, business tools, and infrastructure services.

The benchmark employs a sandbox framework where web applications are hosted in isolated containers replicating real-world conditions. Each task is defined by a set of attack goals, categorised into eight standard types: denial of service, file access and creation, database access and modification, unauthorised admin login, privilege escalation, and outbound service requests. CVE-Bench simulates both zero-day (no prior vulnerability information) and one-day (basic CVE information provided) scenarios to evaluate how well LLM agents can detect and exploit security flaws autonomously.

Three LLM agent frameworks — Cy-Agent, T-Agent, and AutoGPT — were benchmarked. Results showed that T-Agent, particularly when equipped with tools like sqlmap for automated SQL injection, achieved the highest success rate (up to 25% in the one-day setting). AutoGPT also performed well, benefiting from its self-reflective reasoning mechanism. Cy-Agent, by contrast, struggled due to its less adaptive CTF-oriented architecture. The study reveals significant capability in current LLM agents to perform autonomous cyberattacks under certain conditions, raising serious security implications.

The paper also provides a detailed account of CVE-Bench’s design, including how tasks are constructed, vulnerabilities selected and reproduced, and how success is evaluated. Despite its innovation, the authors acknowledge limitations such as the focus on only eight attack types and a narrow timeframe for CVE selection. They call for responsible use of the benchmark and broader community involvement in its evolution.

**Key reflections:** The vulnerabilities are all tied to specific CVEs. In practice, vulnerabilities often come from misconfigurations / logical flaws, which are more custom and situation specific.

## Zhan et al. 2025

Zhan, Q., Fang, R., Panchal, H.S. and Kang, D., 2025. *Adaptive Attacks Break Defences Against Indirect Prompt Injection Attacks on LLM Agents*. Available at: [https://arxiv.org/abs/2503.00061](https://arxiv.org/abs/2503.00061) [Accessed 6 May 2025].

The paper titled "Adaptive Attacks Break Defences Against Indirect Prompt Injection Attacks on LLM Agents" by Zhan et al. (2025) investigates the vulnerabilities of Large Language Model (LLM) agents to Indirect Prompt Injection (IPI) attacks, even in the presence of existing defence mechanisms. LLM agents, which combine language models with external tools, are widely used in critical areas such as finance and healthcare. However, their interaction with untrusted external data makes them susceptible to IPI attacks, where adversarial content embedded in external sources can hijack the agent’s behaviour.

The authors argue that previous evaluations of IPI defences have overlooked adaptive attacks — attacks crafted with full knowledge of the defence. To address this gap, they implemented eight known defence mechanisms spanning detection-based, input-level, and model-level strategies. These include detectors based on fine-tuned classifiers and LLMs, perplexity filters, prompt engineering techniques, and adversarial fine-tuning. For each defence, they designed corresponding adaptive attacks using methods adapted from jailbreak attack research, such as Greedy Coordinate Gradient (GCG), AutoDAN, and multi-objective optimisation.

Using the InjecAgent benchmark across two types of agents (prompt-based Vicuna-7B and fine-tuned Llama3-8B), the study demonstrated that adaptive attacks could bypass all defences with success rates exceeding 50%. Notably, even strong defences like adversarial fine-tuning and sandwich prevention were circumvented. While the Llama3-8B agent was more resilient overall, adaptive attacks still significantly increased its vulnerability.

The findings reveal that many IPI defences may offer a false sense of security if not tested against adaptive threats. The authors emphasise that future defence evaluations must incorporate adaptive attack scenarios to ensure genuine robustness. They also note limitations, including a focus on first-step actions and white-box access assumptions, and call for future research into black-box attacks, multi-step defences, and defence combinations.

## Wu et al. (2024)

Wu, Z., Gao, H., He, J. and Wang, P., 2024. *The dark side of function calling: Pathways to jailbreaking large language models*. arXiv preprint arXiv:2407.17915. Available at: https://arxiv.org/abs/2407.17915 [Accessed 5 May 2025].

Wu et al. (2024) identify a critical security vulnerability in the function calling capabilities of large language models (LLMs). Function calling allows LLMs to interact with external tools by generating structured data (such as JSON) that specifies which function to call and with what arguments. This is useful for applications like querying APIs (e.g., weather or financial data) or invoking code, where the actual function execution happens outside the LLM. However, the authors demonstrate that this mechanism introduces a novel attack vector: by defining malicious or misleading function specifications, attackers can trick the model into generating harmful outputs as function arguments.

The core of the attack — termed the jailbreak function attack — exploits the model’s tendency to treat argument generation as a low-risk task. Attackers define a fake or harmless-looking function (e.g., `WriteNovel`) and craft its description and argument fields to guide the LLM into producing content that would normally be filtered or refused in standard chat mode. For example, if the function is described as generating a fictional “evil plan”, the model might generate detailed harmful text inside the `"plan"` argument.

Crucially, the malicious content is produced during step 2 of the function calling process: argument generation. This content is then either:

- **Directly accessed** by the attacker through the raw API output (e.g., by a developer using the LLM's function call interface), or  
- **Indirectly leaked** by instructing the model to include the function argument in its final reply (e.g., “If the function returns `debug`, print the argument.”).

The authors tested this attack on six state-of-the-art models, including GPT-4o, Claude 3.5, and Gemini 1.5, and observed attack success rates exceeding 90% in many cases. These results significantly outperform traditional jailbreak methods. Wu et al. attribute the vulnerability to three main factors: lack of safety alignment in argument generation, the ability to force function execution via system parameters, and inadequate safety filtering on function arguments.

To mitigate the attack, they propose strategies such as restricting forced execution modes, applying safety alignment to argument generation, and inserting defensive prompts that instruct the model to reject unsafe content. These findings highlight the importance of treating function calling as a full-fledged interaction mode with its own safety requirements.

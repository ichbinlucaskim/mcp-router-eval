> **Point-in-time inspection (2026-07-01).** This report captures the environment and dataset reality
> checked before Phase 0. ADRs 0001–0008 are sourced from its findings. It is a snapshot — package
> versions, dataset contents, and published numbers may drift; verify against the live sources before
> relying on any specific figure.

# BUILD-READINESS REPORT — mcp-router-eval

*Inspection only. No project code written. Findings verified against the live repos and this machine, not the proposal's assumptions.*

## 1. Environment status

Machine: **Python 3.11.9**, **macOS arm64 (Apple Silicon)**, **no CUDA — MPS available**. Node v25.5.0, Claude Code CLI **2.1.197** on PATH.

| Dependency | Installs / present? | Note |
|---|---|---|
| torch | ✅ 2.12.0 | CPU/MPS build. No GPU — training is CPU/MPS only. Fine for a 573-node graph. |
| torch-geometric | ✅ 2.8.0 | **`RGCNConv`, `GATConv`, `SAGEConv` import and run verified.** Pure-torch path works. |
| torch-scatter | ⚠️ missing | **Not required** — verified RGCN forward pass runs without it on PyG 2.8. Install only if you hit a segment-op error. |
| dgl | ❌ missing | Not needed — see PyG recommendation below. |
| pydantic | ✅ 2.12.5 | v2 — good for the contract layer (T1.1). |
| scikit-learn | ✅ 1.9.0 | For metrics/splits. |
| numpy / pandas | ✅ 2.4.2 / 3.0.1 | — |
| rank-bm25 | ❌ missing | `pip install rank-bm25` — needed for the BM25 baseline (T0.2). |
| sentence-transformers | ❌ missing | `pip install sentence-transformers` — but see §5: it will **not** reproduce the paper's numbers. |
| anthropic (API SDK) | ✅ 0.96.0 | Raw API client. Has **no** agent/MCP driver surface. |
| claude-agent-sdk (Python) | ❌ missing | Available on PyPI (0.2.110). **This is how Layer 3 actually runs** (see below). |

**PyG vs DGL → use PyG.** It's already installed and verified working here; heterogeneous/typed edges (`RGCNConv(num_relations=…)`, `HeteroData`) and GAT are first-class; larger community and examples. DGL would be a second install with a heavier heterograph API and zero upside on a graph this small.

**How Layer 3 actually invokes Claude Code.** The `anthropic` package alone can't drive tool-call loops as an agent. Two real options:
- **`claude-agent-sdk` (Python)** — `pip install claude-agent-sdk`; it shells out to the installed Claude Code CLI (2.1.197 is present) and gives you `query()`/`ClaudeSDKClient` with programmatic MCP-tool registration and a structured message/tool-call stream. This is the natural fit for ExecResult's `call_trace`.
- **Raw `anthropic` + your own tool loop** — more control, but you re-implement the agent loop. Only worth it if the SDK's trace granularity is insufficient.

Recommendation: `claude-agent-sdk`, registering the mock tools (§3) as in-process MCP tools.

## 2. Dataset findings

### ToolLinkOS — `github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS`
- **Format:** three plain JSON files. `regular_tools.json` (523 tools), `core_tools.json` (50 tools), `instances.json` (1,569 queries). No Neo4j/CSV needed — the PNG shows Neo4j but the shipped data is JSON.
- **Tool representation:** `{name, description, parameters[{name,type,description,required}], depends_on[], func_type}`. `func_type ∈ {regular, core}`. Core tools have **empty `depends_on`** (leaf utilities others depend on). Tool identity = `name` string (there is **no `tool_id`**).
  > **[Corrected 2026-07-01 per `docs/data-inspection-toollinkos.md`]** The "core tools have empty `depends_on` / leaf utilities" statement above is **wrong**: firsthand inspection found **30/50 core tools DO have dependencies**; only 20 are leaves. Do not assume `core ⇒ leaf` anywhere (see ADR 0012).
- **Dependency encoding — 4 edge types, not 3:** `depends_on` entries carry `dependence_type ∈ {TOOL_DIRECTLY_DEPENDS_ON (676), TOOL_INDIRECTLY_DEPENDS_ON (175), PARAMETER_DIRECTLY_DEPENDS_ON (404), PARAMETER_INDIRECTLY_DEPENDS_ON (239)}` plus **2 malformed `PARAMETER_DEPENDS_ON`** rows. Mapping to the proposal's `param_source / precond / core`:
  - `param_source` ← PARAMETER_{DIRECTLY,INDIRECTLY} ✅
  - `precond` ← TOOL_{DIRECTLY,INDIRECTLY} ✅
  - **`core` ← nothing.** "Core" is a **node attribute (`func_type`), not an edge type.** The proposal's third edge type does not exist in the data. This changes the R-GCN `num_relations` and edge-type ablation B (see §3/§4).
- **QUERY SET: YES ✅ — this is the good news.** `instances.json` gives `{user_query, main_golden_function_name, golden_function_names[]}`. Gold query→tool labels exist, so mAP/recall/nDCG are computable. The #1 feared blocker (no queries) **does not apply.**
- **License:** **MIT** — redistribution in your repo is allowed (keep the LICENSE + fictional-tools disclaimer).
- **Two data-reality caveats that bite the proposal:**
  1. **Zero single-tool queries.** Every one of the 1,569 instances has ≥2 gold functions (min size 2, mean ≈6.0). The headline "single-tool vs composite" slice (§4, §6) has an **empty single-tool bucket** on ToolLinkOS.
  2. **"~6.3 deps/tool" won't match a naive edge count.** Actual direct `depends_on` = **2.61 edges/tool**. The ~6 figure matches **average gold-set size per query (≈6.0)**, i.e. closure size — not per-tool degree. T0.1's done-when needs this definition fixed or it "fails" spuriously.

### ToolSandbox — `github.com/apple/ToolSandbox`
- **Format / representation:** a **different kind of benchmark** — a stateful, conversational, interactive execution framework (Python) with a user simulator and *real executable* tools. It is **not** a dependency-typed retrieval graph and ships **no** `instances.json`-style query→tool gold set in ToolLinkOS form.
- **Edge types:** none native. The GRTF paper *hand-adapted* ToolSandbox into their 4-edge schema, and that annotated variant is **not in the public repo** (only `toollinkos/` ships). So it is **not usable off-the-shelf** for this project's graph-retrieval eval.
- **Query set (for our purpose):** effectively **no** — its "queries" are conversational trajectories with milestone scoring, not ranked-tool gold labels.
- **License:** **Apple custom license (GitHub reports `NOASSERTION`/Other), not MIT.** **Do not redistribute** it in your repo — link/submodule only.

## 3. Blockers — ranked, most severe first

1. **The single-tool slice is empty (data-level).** ToolLinkOS has 0 single-tool queries, but the entire "retrieval ≠ completion / GNN wins on composite vs single" framing needs a single-tool contrast. **Resolution:** redefine the slice *within* composites by closure depth/size (e.g. shallow 2–3 gold tools vs deep 6+), which the data supports cleanly; optionally synthesize single-tool queries from `main_golden_function_name` for tools with no deps. Pick this before T0.3.

2. **Baselines: no code ships + embedding model is a paid API.** The repo is **data-only** — no BM25/RAG/GraphRAG implementation. The paper used **Azure OpenAI `text-embedding-ada-002`** (embeddings) and **`gpt-4o-2024-08-06`** (LLM reranker). `sentence-transformers` will *not* land within ±2% of naive-RAG mAP@10 = 0.210. **Resolution:** get Azure/OpenAI keys, reimplement all four baselines, target Table 2: BM25 **0.185**, naïve RAG **0.210**, hybrid (α=0.8) **0.202**, GraphRAG k=3 **0.856** (no rerank) / **0.927** (rerank). Note GraphRAG's jump is mostly traversal, so it's forgiving; the *dense baselines* are what demand ada-002.

3. **Fictional tools → `completion_rate` is a proxy, not real task success.** No tool runs; instances carry **no gold arguments and no gold answer** — only gold tool *names*. RQ1's "transfer loss" therefore compares recall against a *structural* completion verdict, not semantic success. **Resolution:** accept and document a proxy verdict (mock executor, §below); do **not** promise real functional completion.

4. **ToolSandbox is not a drop-in second benchmark + can't be redistributed.** **Resolution:** treat **ToolLinkOS as the primary and only guaranteed benchmark**; make ToolSandbox a stretch that requires replicating Apple-license-bound annotation. Adjust the "two-benchmark" deliverable language.

5. **Edge-type mismatch breaks R-GCN config and ablation B.** 3 proposed types vs 4 real, and "core" isn't an edge. **Resolution:** set `num_relations=4` on the real types (tool/param × direct/indirect), drop the "core" edge, fold `func_type` into node features, and clean the 2 malformed edges.

6. **Several contract fields have no data source** (query_id, JSONSchema, args). See §4.

## 4. Required schema changes before T1.1

| Contract field | Problem vs real data | Fix |
|---|---|---|
| `RouteResult.query_id`, `ExecResult.query_id` | `instances.json` has **no id** | Assign a synthetic `q{index}` at load time; freeze it. |
| `Edge.type ENUM[param_source\|precond\|core]` | Real = 4 types; **no "core" edge** | Change enum to `{param_direct, param_indirect, tool_direct, tool_indirect}` (or collapse to `{param_source, precond}` + a `direct:bool`). Remove `core` as an edge; add `is_core:bool` to `ToolSpec`. |
| `ToolSpec.schema : JSONSchema` | Tools ship a `parameters[]` list, **not JSONSchema** | Add a builder: `parameters[] → JSONSchema` (map `type`, `required`). Populate `schema` from that; don't expect a native field. |
| `ToolSpec.deps` | ✅ available | Map from `depends_on[].name`. |
| `ExecResult.call_trace[].args` | **No gold args anywhere** | Mock executor synthesizes type-valid args; mark them synthetic in evidence. |
| `ExecResult.completed` | **No gold answer** to check against | Define as structural proxy (see below). Document explicitly. |
| `RouteResult.confidence`, `homophily_local` | Not in data — router-computed | Fine, but decide the formula now (§5). |
| `Attribution` | Works — `golden_function_names` supplies the "required tool" set for ROUTING blame | No change. |

**Smallest viable mock executor (consistent with §3.3 ExecResult).** One *generic, schema-driven* stub — **not** 573 hand-written tools:
- Given a `ToolSpec`, validate incoming args against the built JSONSchema (required present, types match) → `ok`/`error`, record `t_ms`.
- Return a canned typed value keyed by declared parameter/return `type` (deterministic dummy per type).
- `completed = true` iff every tool in `golden_function_names` was invoked, dependency order respected (a tool called only after its `depends_on` are satisfied), and all calls type-valid.
- **Effort: ~1–2 days, one file.** It fits the Week 4–5 budget *because* it's generic. It only blows the budget if someone insists on real semantic completion — which is impossible with fictional tools, so that must be ruled out now.

## 5. Baseline reproducibility

- **Runnable code:** **No.** The repo ships data only; you reimplement BM25 / naïve RAG / hybrid / Graph RAG-Tool Fusion from the paper. Graph RAG-Tool Fusion = dense seed retrieval + traversal over the 4 dependency edges + optional LLM rerank — straightforward to rebuild.
- **Embedding model (must match):** **Azure OpenAI `text-embedding-ada-002`** (1536-dim). Reranker/LLM: **`gpt-4o-2024-08-06`**. Both are paid-API. This is the single biggest reproduction dependency.
- **What blocks ±2%:** (a) not having ada-002 keys (a local ST model shifts dense baselines by more than 2%); (b) hybrid fusion detail — paper uses **α=0.8** lexical/dense weighting; (c) rerank on/off — GraphRAG is 0.856 without rerank, 0.927 with; report the matching row. Traversal-driven GraphRAG is forgiving of embedding choice; the **dense baselines are the fragile ones**.

## 6. Open decisions — recommended defaults

| Decision | Recommended default | One-line rationale |
|---|---|---|
| PyG vs DGL | **PyG** | Installed + verified here; native typed-edge RGCN/GAT; no second install. |
| Embedding model | **ada-002 for baselines**; reuse the same vectors as GNN node features | Only way to match Table 2; reusing keeps router and baselines on one embedding space. |
| GNN node features (§3.1) | Embed `name + "\n" + description` with the chosen model | Description is the only rich text per tool; matches how baselines see tools. |
| `confidence` | Normalized top-k score gap (margin between k-th and (k+1)-th score), sigmoid-squashed | Cheap, monotonic, needs no extra head; good enough for the gate. |
| `homophily_local` | Mean pairwise cosine of node embeddings over the selected closure's edges | Directly the "are my dependencies semantically similar" signal the thesis rests on. |
| Edge types for R-GCN | **4 real types** (`num_relations=4`); drop "core" edge; `is_core` as node feature | Matches data; enables a meaningful edge-type ablation B. |
| Query slice (replaces single vs composite) | **Closure depth/size buckets** (shallow 2–3 vs deep ≥6) | ToolLinkOS has no single-tool queries; depth still isolates where the GNN should win. |
| Execution sample domains (T2.3) | Pick **1–2 industries** with high intra-domain / low cross-domain dependency and self-contained core-tool chains | Keeps the mock closure runnable and cost finite; avoids dangling cross-domain deps. |
| Completion verdict | Structural proxy (right tool set + dep order + type-valid args) | Fictional tools make semantic completion impossible; must be defined up front. |

## Green-light checklist — all must be true before your first line of project code

- [ ] Decide and record the **query-slice definition** (depth buckets) — replaces the empty single-tool slice. **(Blocker 1)**
- [ ] Confirm **Azure/OpenAI access to `text-embedding-ada-002` + `gpt-4o-2024-08-06`**, or consciously accept you won't match published dense baselines. **(Blocker 2)**
- [ ] Get sign-off that **`completed` is a structural proxy** on fictional tools, not real task success. **(Blocker 3)**
- [ ] Reclassify **ToolSandbox as stretch-only**, ToolLinkOS as the sole primary benchmark; don't plan to redistribute ToolSandbox. **(Blocker 4)**
- [ ] Freeze the **revised contract schemas** (4 edge types, synthetic `query_id`, JSONSchema builder, synthetic args) — §4. **(Blocker 5–6)**
- [ ] Rewrite **T0.1's done-when** stat to "avg gold-set size ≈6 / 573 tools / 1,569 queries," not "6.3 deps/tool."
- [ ] `pip install rank-bm25 sentence-transformers claude-agent-sdk`; clean the 2 malformed `PARAMETER_DEPENDS_ON` edges.
- [ ] Confirm **Claude Code executor path** = `claude-agent-sdk` driving the CLI (2.1.197 present), with mock tools registered as MCP tools.

---

**Bottom line:** The feared #1 blocker (no query set) is a non-issue — gold labels exist. The real blockers are: **(1)** no single-tool queries so the headline slice must be redefined, **(2)** baseline reproduction needs a paid embedding API and a from-scratch reimplementation, and **(3)** fictional tools force `completion_rate` to be a structural proxy. All three are resolvable but should be decided *before* Phase 0, not discovered in Week 4.

**Sources:** [ToolLinkOS / Graph RAG-Tool Fusion repo](https://github.com/EliasLumer/Graph-RAG-Tool-Fusion-ToolLinkOS) · [GRTF paper (arXiv 2502.07223)](https://arxiv.org/html/2502.07223) · [Apple ToolSandbox](https://github.com/apple/ToolSandbox)

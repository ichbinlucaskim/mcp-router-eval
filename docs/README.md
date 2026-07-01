# Documentation

## Architecture Decision Records (ADRs)

`adr/` holds one file per significant decision, numbered sequentially (`NNNN-title.md`).

**Convention:** every future significant decision gets a new numbered ADR **before** the code that
implements it is written. Never delete an ADR — supersede it with a new one and mark the old one's
Status accordingly (`Superseded by NNNN`).

Each ADR uses the format in [`adr/0000-adr-template.md`](adr/0000-adr-template.md):
Title / Status / Context / Decision / Consequences / Alternatives considered.

### Index

| # | Decision | Status |
| --- | --- | --- |
| [0001](adr/0001-toollinkos-sole-primary-benchmark.md) | ToolLinkOS is the sole primary benchmark; ToolSandbox demoted to stretch | Accepted |
| [0002](adr/0002-executor-claude-agent-sdk.md) | Executor = Claude Code via claude-agent-sdk with mock tools as MCP tools | Accepted |
| [0003](adr/0003-embedding-provider-interface.md) | Embedding behind a provider interface; LocalEmbedder(BGE) default, ada-002 optional | Accepted |
| [0004](adr/0004-completion-structural-proxy.md) | Completion is a structural proxy, not semantic success | Accepted |
| [0005](adr/0005-closure-depth-slices.md) | Query slice = closure-depth buckets (shallow 2-3 / deep >=6) | Accepted |
| [0006](adr/0006-four-edge-types.md) | 4 real dependency edge types; "core" is a node feature | Accepted |
| [0007](adr/0007-pyg-over-dgl.md) | GNN framework = PyTorch Geometric (not DGL) | Accepted |
| [0008](adr/0008-graph-identity-conventions.md) | Tool identity = name; synthetic query_id = q{index} | Accepted |
| [0009](adr/0009-record-decisions-as-adrs.md) | Record architecture decisions as numbered ADRs before implementing them | Accepted |

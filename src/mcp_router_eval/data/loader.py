"""Parse ToolLinkOS: regular_tools.json, core_tools.json, instances.json.

Tools carry {name, description, parameters[], depends_on[], func_type}; instances carry
{user_query, main_golden_function_name, golden_function_names[]}. Assigns synthetic query_id
= ``q{index}`` at load (ADR 0008) and cleans the 2 malformed PARAMETER_DEPENDS_ON rows (ADR 0006).

STUB.
"""

raise NotImplementedError("data.loader: not implemented yet (T0.1)")

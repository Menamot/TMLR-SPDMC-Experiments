DEFAULT_MINMAX_ABLATION_MODELS = (
    "DMC-CMeans",
    "DMC-DiagGMM",
    "DMC-DiagGMM-ClassWise",
    "DMC-Tree-DiagGMM",
)


def swap_discrete_model_prefix(model_name: str, target_prefix: str) -> str:
    """Swap between DMC/DBC prefixes while keeping the discretizer suffix."""
    if target_prefix not in {"DMC", "DBC"}:
        raise ValueError("target_prefix must be 'DMC' or 'DBC'")

    prefix, sep, suffix = model_name.partition("-")
    if not sep or prefix not in {"DMC", "DBC"}:
        raise ValueError(
            "model_name must start with 'DMC-' or 'DBC-' to swap minmax family"
        )

    return f"{target_prefix}-{suffix}"


def build_minmax_ablation_best_params(
    dmc_best_params: dict,
    model_names=DEFAULT_MINMAX_ABLATION_MODELS,
) -> dict:
    """Expand DMC best params into paired DMC/DBC configs with identical params."""
    expanded = {}

    for dmc_name in model_names:
        if not dmc_name.startswith("DMC-"):
            raise ValueError("model_names must contain DMC-prefixed model names only")
        if dmc_name not in dmc_best_params:
            raise ValueError(f"Missing DMC best params for model: {dmc_name}")

        params = dict(dmc_best_params[dmc_name])
        expanded[dmc_name] = dict(params)
        expanded[swap_discrete_model_prefix(dmc_name, "DBC")] = dict(params)

    return expanded

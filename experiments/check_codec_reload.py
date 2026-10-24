import torch
from src.config import load_config
from src.config.schema import ExperimentConfig
from src.envs.topology_registry import load_topologies
from experiments.train_qh_ctde import _new_env
from experiments.debug.c3_c4_comparison.code.debug_c3_c4_logits_v2 import load_model

cfg = load_config("configs/qh_ctde.yaml")
experiment = ExperimentConfig.from_dict(cfg)
topologies = load_topologies("T3", experiment.topology_file)
topo = next(t for t in topologies if t.metadata.get("id", t.name) == "T3")

env = _new_env(topo, cfg, 1, seed_offset=10001)
try:
    model = load_model(
        env, cfg, "experiments/debug/smoke_tests/smoke_C4/T3_C4_seed1_model.pt",
        torch.device("cpu")
    )

    print("initialized immediately after load:", model.codec._initialized)
    before = model.codec.codebook.detach().clone()

    obs, _ = env.reset(seed=10001)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        model.encode_observations(obs_t)

    after = model.codec.codebook.detach().clone()
    print("initialized after first encode:", model.codec._initialized)
    print("max codebook change:", (after - before).abs().max().item())
    print("codebook changed:", not torch.equal(before, after))
finally:
    env.close()


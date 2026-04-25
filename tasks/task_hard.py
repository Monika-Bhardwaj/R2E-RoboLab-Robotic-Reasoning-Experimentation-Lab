from r2e_env.models import TaskConfig

HARD_CONFIG = TaskConfig(
    name="hard",
    active_variables=["friction_level", "alignment_error", "stiffness"],
    deceptive_reward_enabled=True,
    max_steps=90,
    seed_range=(0, 9999),
    investigation_fraction=0.40,
    verification_fraction=0.30,
    sparse_reward_mode=True,
)

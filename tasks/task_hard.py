from r2e_env.models import TaskConfig

HARD_CONFIG = TaskConfig(
    name="hard",
    active_variables=["friction_level", "alignment_error", "stiffness"],
    deceptive_reward_enabled=True,
    max_steps=50,
    seed_range=(0, 9999),
)

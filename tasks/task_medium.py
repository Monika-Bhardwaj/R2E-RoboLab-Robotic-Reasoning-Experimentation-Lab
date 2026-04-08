from r2e_env.models import TaskConfig

MEDIUM_CONFIG = TaskConfig(
    name="medium",
    active_variables=["friction_level", "alignment_error"],
    deceptive_reward_enabled=True,
    max_steps=30,
    seed_range=(0, 9999),
)

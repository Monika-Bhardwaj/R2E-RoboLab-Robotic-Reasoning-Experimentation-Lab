from r2e_env.models import TaskConfig

EASY_CONFIG = TaskConfig(
    name="easy",
    active_variables=["friction_level"],
    deceptive_reward_enabled=False,
    max_steps=20,
    seed_range=(0, 9999),
)

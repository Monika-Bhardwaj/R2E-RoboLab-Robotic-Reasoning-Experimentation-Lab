from r2e_env.models import TaskConfig

EASY_CONFIG = TaskConfig(
    name="easy",
    active_variables=["friction_level"],
    deceptive_reward_enabled=False,
    max_steps=40,
    seed_range=(0, 9999),
    investigation_fraction=0.35,
    verification_fraction=0.35,
)

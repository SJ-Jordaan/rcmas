from pathlib import Path

from qlearning.engine import Territory
from qlearning.rl.trainer import SelfPlayTrainer, TrainConfig


def test_trainer_runs_and_writes(tmp_path: Path):
    territory = Territory.from_ascii([
        "..",
    ])
    trainer = SelfPlayTrainer(out_dir=tmp_path)
    cfg = TrainConfig(episodes=5)
    artifacts = trainer.train(territory, num_agents=2, cfg=cfg, seed=0)
    assert artifacts.directory.exists()
    assert (artifacts.directory / "meta.json").exists()
    assert (artifacts.directory / "agent_0.json").exists()
    assert (artifacts.directory / "agent_1.json").exists()

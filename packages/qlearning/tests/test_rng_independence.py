from qlearning.agents.qtable_agent import QTableAgent
from qlearning.rl.qtable import QTable


def test_qtable_agent_rng_not_shared_by_default():
    qt = QTable.empty()
    a = QTableAgent(qtable=qt, num_agents=2)
    b = QTableAgent(qtable=qt, num_agents=2)
    assert a.rng is not b.rng

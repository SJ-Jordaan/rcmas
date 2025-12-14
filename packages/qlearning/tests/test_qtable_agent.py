from qlearning.agents.qtable_agent import QTableAgent
from qlearning.engine import Coord, GameState, Territory
from qlearning.rl.encoding import TabularEncoding
from qlearning.rl.qtable import QTable


def test_qtable_agent_picks_best_action():
    territory = Territory.from_ascii([".."])
    state = GameState.new(territory, num_agents=2)

    enc = TabularEncoding(num_agents=2)
    sk = enc.state_key(state)

    # Prefer coord (1,0) over (0,0)
    qt = QTable.empty()
    qt.set(sk, enc.action_key(state, Coord(1, 0)), 10.0)
    qt.set(sk, enc.action_key(state, Coord(0, 0)), 0.0)

    agent = QTableAgent(qtable=qt, num_agents=2, epsilon=0.0)
    act = agent.select_action(state, agent_id=0)
    assert act == Coord(1, 0)

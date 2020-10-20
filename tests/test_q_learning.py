import unittest
import copy


class TestQLearning(unittest.TestCase):

    def test_vfa_update(self):
        from q_learning_vfa import vfa_update

        states = [[0, 0, 0, 0]]
        actions = [0]
        rewards = [1]
        dones = [True]
        next_states = [[0, 0, 0, 0]]
        loss = vfa_update(states, actions, rewards, dones, next_states)
        assert loss == 1

    def test_q_learning_vfa(self):
        from q_learning_vfa import q_learning, Q
        Q_before_training = copy.deepcopy(Q)
        q_learning(10)
        for p1, p2 in zip(Q_before_training.parameters(), Q.parameters()):
            assert p1.data.ne(p2.data).sum() > 0

    def test_q_learning_tabular(self):
        from q_learning_tabular import q_learning
        rewards = q_learning(10)
        assert len(rewards) == 10


if __name__ == '__main__':
    unittest.main()

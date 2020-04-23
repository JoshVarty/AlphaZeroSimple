import numpy as np
import unittest
from monte_carlo_tree_search import Node, MCTS, ucb_score
from game import Connect2Game


class MCTSTests(unittest.TestCase):

    def test_mcts_from_root_with_equal_priors(self):
        class MockModel:
            def predict(self, board):
                # starting board is:
                # [0, 0, 1, -1]
                return np.array([0.26, 0.24, 0.24, 0.26]), 0.0001

        game = Connect2Game()
        args = {'num_simulations': 50}

        model = MockModel()
        mcts = MCTS(game, model, args)
        canonical_board = [0, 0, 0, 0]
        print("starting")
        root = mcts.run(model, canonical_board, to_play=1)

        # the best move is to play at index 1 or 2
        best_outer_move = max(root.children[0].visit_count, root.children[0].visit_count)
        best_center_move = max(root.children[1].visit_count, root.children[2].visit_count)
        self.assertGreater(best_center_move, best_outer_move)

    def test_mcts_finds_best_move_with_really_bad_priors(self):
        class MockModel:
            def predict(self, board):
                # starting board is:
                # [0, 0, 1, -1]
                return np.array([0.3, 0.7, 0, 0]), 0.0001

        game = Connect2Game()
        args = {'num_simulations': 25}

        model = MockModel()
        mcts = MCTS(game, model, args)
        canonical_board = [0, 0, 1, -1]
        print("starting")
        root = mcts.run(model, canonical_board, to_play=1)

        # the best move is to play at index 1
        self.assertGreater(root.children[1].visit_count, root.children[0].visit_count)

    def test_mcts_finds_best_move_with_equal_priors(self):

        class MockModel:
            def predict(self, board):
                return np.array([0.51, 0.49, 0, 0]), 0.0001

        game = Connect2Game()
        args = { 'num_simulations': 25 }

        model = MockModel()
        mcts = MCTS(game, model, args)
        canonical_board = [0, 0, -1, 1]
        root = mcts.run(model, canonical_board, to_play=1)

        # the better move is to play at index 1
        self.assertLess(root.children[0].visit_count, root.children[1].visit_count)

    def test_mcts_finds_best_move_with_really_really_bad_priors(self):
        class MockModel:
            def predict(self, board):
                # starting board is:
                # [-1, 0, 0, 0]
                return np.array([0, 0.3, 0.3, 0.3]), 0.0001

        game = Connect2Game()
        args = {'num_simulations': 100}

        model = MockModel()
        mcts = MCTS(game, model, args)
        canonical_board = [-1, 0, 0, 0]
        root = mcts.run(model, canonical_board, to_play=1)

        # the best move is to play at index 1
        self.assertGreater(root.children[1].visit_count, root.children[2].visit_count)
        self.assertGreater(root.children[1].visit_count, root.children[3].visit_count)

class NodeTests(unittest.TestCase):

    def test_initialization(self):
        node = Node(0.5, to_play=1)

        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.prior, 0.5)
        self.assertEqual(len(node.children), 0)
        self.assertFalse(node.expanded())
        self.assertEqual(node.value(), 0)

    def test_selection(self):
        node = Node(0.5, to_play=1)
        c0 = Node(0.5, to_play=-1)
        c1 = Node(0.5, to_play=-1)
        c2 = Node(0.5, to_play=-1)
        node.visit_count = 1
        c0.visit_count = 0
        c2.visit_count = 0
        c2.visit_count = 1

        node.children = {
            0: c0,
            1: c1,
            2: c2,
        }

        action = node.select_action(temperature=0)
        self.assertEqual(action, 2)

    def test_expansion(self):
        node = Node(0.5, to_play=1)

        state = [0, 0, 0, 0]
        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)

        self.assertEqual(len(node.children), 4)
        self.assertTrue(node.expanded())
        self.assertEqual(node.to_play, to_play)
        self.assertEqual(node.children[0].prior, 0.25)
        self.assertEqual(node.children[1].prior, 0.15)
        self.assertEqual(node.children[2].prior, 0.50)
        self.assertEqual(node.children[3].prior, 0.10)

    def test_ucb_score_no_children_visited(self):
        node = Node(0.5, to_play=1)
        node.visit_count = 1

        state = [0, 0, 0, 0]
        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 0
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        self.assertEqual(score_2, node.children[2].prior)
        self.assertEqual(score_3, node.children[3].prior)

    def test_ucb_score_one_child_visited(self):
        node = Node(0.5, to_play=1)
        node.visit_count = 1

        state = [0, 0, 0, 0]
        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 1
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        # If we visit one child once, its score is halved
        self.assertEqual(score_2, node.children[2].prior / 2)
        self.assertEqual(score_3, node.children[3].prior)

        action, child = node.select_child()

        self.assertEqual(action, 0)

    def test_ucb_score_one_child_visited_twice(self):
        node = Node(0.5, to_play=1)
        node.visit_count = 2

        state = [0, 0, 0, 0]

        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 2
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        action, child = node.select_child()

        # Now that we've visited the second action twice, we should
        # end up trying the first action
        self.assertEqual(action, 0)

    def test_ucb_score_no_children_visited(self):
        node = Node(0.5, to_play=1)
        node.visit_count = 1

        state = [0, 0, 0, 0]
        action_probs = [0.25, 0.15, 0.5, 0.1]
        to_play = 1

        node.expand(state, to_play, action_probs)
        node.children[0].visit_count = 0
        node.children[1].visit_count = 0
        node.children[2].visit_count = 1
        node.children[3].visit_count = 0

        score_0 = ucb_score(node, node.children[0])
        score_1 = ucb_score(node, node.children[1])
        score_2 = ucb_score(node, node.children[2])
        score_3 = ucb_score(node, node.children[3])

        # With no visits, UCB score is just the priors
        self.assertEqual(score_0, node.children[0].prior)
        self.assertEqual(score_1, node.children[1].prior)
        # If we visit one child once, its score is halved
        self.assertEqual(score_2, node.children[2].prior / 2)
        self.assertEqual(score_3, node.children[3].prior)



if __name__ == '__main__':
    unittest.main()

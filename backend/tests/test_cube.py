import unittest
import numpy as np
from src.cube.simulator import CubeSimulator, reverse_moves, MOVES


class TestCubeSimulator(unittest.TestCase):
    def setUp(self):
        self.cube = CubeSimulator()

    def test_initial_state(self):
        """Standard solved cube should be solved."""
        self.assertTrue(self.cube.is_solved())
        self.assertEqual(self.cube.distance_to_solved(), 0)

    def test_single_moves_and_inverses(self):
        """Every quarter turn should be undone by its inverse."""
        for move in MOVES:
            self.cube.reset()
            self.cube.apply_move(move)
            self.assertFalse(self.cube.is_solved(), f"Move {move} should change state")
            
            # Apply inverse
            inv = move[0] if "'" in move else move + "'"
            self.cube.apply_move(inv)
            self.assertTrue(self.cube.is_solved(), f"Inverse of {move} should return to solved")

    def test_four_turns_return_to_solved(self):
        """Four quarter turns on one face must be identity."""
        for move in ["U", "D", "L", "R", "F", "B"]:
            self.cube.reset()
            for _ in range(4):
                self.cube.apply_move(move)
            self.assertTrue(self.cube.is_solved(), f"{move}^4 should be identity")

    def test_color_counts(self):
        """Rubik's cube must always have 9 of each color."""
        self.cube.scramble(20)
        counts = np.bincount(self.cube.state, minlength=6)
        for i, count in enumerate(counts):
            self.assertEqual(count, 9, f"Color {i} count should be 9, got {count}")

    def test_scramble_and_reverse(self):
        """Scramble then apply reverse sequence leads to solved."""
        for depth in [1, 5, 10, 20]:
            self.cube.reset()
            state, moves = self.cube.scramble(depth)
            self.assertEqual(len(moves), depth)
            
            rev_moves = reverse_moves(moves)
            self.cube.apply_moves(rev_moves)
            self.assertTrue(self.cube.is_solved(), f"Reversing scramble of depth {depth} failed")

    def test_scramble_changes_state(self):
        """A non-zero scramble should change the solved cube."""
        original = self.cube.reset()
        state, moves = self.cube.scramble(5)
        self.assertEqual(len(moves), 5)
        self.assertFalse(np.array_equal(state, original))

if __name__ == '__main__':
    unittest.main()

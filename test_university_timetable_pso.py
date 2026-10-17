"""Validate constraint counting and the reproducible PSO result."""
import unittest
import numpy as np
from university_timetable_pso import decode, evaluate, optimize


class TimetableTests(unittest.TestCase):
    def test_all_courses_colliding(self):
        score = evaluate(np.zeros((6, 2), dtype=int))
        self.assertEqual(score.faculty_clashes, 2)
        self.assertEqual(score.room_clashes, 15)
        self.assertEqual(score.room_type_errors, 2)

    def test_known_feasible_balanced_schedule(self):
        schedule = np.array([[0, 0], [2, 0], [0, 2], [1, 2], [2, 4], [0, 4]])
        self.assertEqual(evaluate(schedule).cost, 0)

    def test_decode_respects_bounds(self):
        schedule = decode(np.array([-100, 100] * 6))
        np.testing.assert_array_equal(schedule, np.array([[0, 4]] * 6))

    def test_default_pso_reaches_valid_optimum_reproducibly(self):
        schedule, history = optimize()
        repeated, repeated_history = optimize()
        np.testing.assert_array_equal(schedule, repeated)
        self.assertEqual(history, repeated_history)
        self.assertEqual(evaluate(schedule).cost, 0)
        self.assertTrue(all(b <= a for a, b in zip(history, history[1:])))
        self.assertEqual(history[-1], evaluate(schedule).cost)


if __name__ == '__main__':
    unittest.main()

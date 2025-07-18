import unittest
import pandas as pd
import numpy as np
from Visaualization import FunctionMatcher, TestDataMapper, MappingError

class TestFunctionMatcher(unittest.TestCase):
    def setUp(self):
        # Create dummy training and ideal datasets
        x = np.linspace(0, 10, 11)
        self.train_df = pd.DataFrame({'x': x, 'y1': x**2, 'y2': x+5, 'y3': x*2, 'y4': x-3})
        self.ideal_df = pd.DataFrame({'x': x})
        for i in range(1, 51):
            if i == 10:
                self.ideal_df[f'y{i}'] = x**2  # perfect match for y1
            else:
                self.ideal_df[f'y{i}'] = x + i

        # Save dummy CSVs for testing
        self.train_df.to_csv("train_dummy.csv", index=False)
        self.ideal_df.to_csv("ideal_dummy.csv", index=False)

    def test_match_functions(self):
        matcher = FunctionMatcher("train_dummy.csv", "ideal_dummy.csv")
        matcher.load_data()
        matcher.match_functions()
        best_matches, max_deviations = matcher.get_results()

        # y1 should match y10 because both are x**2
        self.assertEqual(best_matches['y1'], 'y10')
        self.assertTrue(isinstance(max_deviations['y10'], float))

class TestTestDataMapper(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0, 10, 11)
        self.ideal_df = pd.DataFrame({'x': x, 'y10': x**2})
        self.test_df = pd.DataFrame({'x': x, 'y': x**2 + 0.1})  # within deviation
        self.test_df.to_csv("test_dummy.csv", index=False)
        self.match_results = ({'y1': 'y10'}, {'y10': 0.1})  # small deviation

    def test_map_points_within_threshold(self):
        mapper = TestDataMapper("test_dummy.csv", self.ideal_df, self.match_results)
        mapper.load_data()
        mapper.map_points()
        mapped_df = mapper.get_mapped_df()

        # All test points should be mapped
        self.assertEqual(mapped_df['ideal_function'].notna().sum(), len(self.test_df))
        self.assertTrue(mapped_df['Delta Y (test function)'].max() <= 0.1 * np.sqrt(2))


if __name__ == '__main__':
    unittest.main()

"""Software fixtures: strict boundaries, full denominator, and fail-closed input."""
import math
import unittest
import census as c

class Cal:
    def __init__(self, scores):
        self.scores = scores
    def iter_pair_scores(self, vectors, block, audit):
        for (i,j),score in zip(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)), self.scores):
            yield i,j,score
    def _bin(self, score):
        return min(19,max(0,int((score+1)*10)))

class CensusTests(unittest.TestCase):
    def test_strict_threshold_and_adjacent_floats(self):
        cuts = [0.9, 0.95, 0.995]
        scores = [math.nextafter(.9,-math.inf), .9, math.nextafter(.9,math.inf), .95, .995, 1.]
        result = c.collect([0]*4, Cal(scores), cuts)
        self.assertEqual(result['counts_strictly_above'], [4,2,1])
        self.assertEqual(sum(result['cut_interval_populations']),6)
    def test_independent_bruteforce_all_boundaries(self):
        cuts = [.5,.9,.95,.99,.995]
        for t in cuts:
            scores = [math.nextafter(t,-math.inf), t, math.nextafter(t,math.inf), -1., 0., 1.]
            result = c.collect([0]*4, Cal(scores), cuts)
            self.assertEqual(result['counts_strictly_above'], [sum(x>v for x in scores) for v in cuts])
    def test_incomplete_census_rejected(self):
        with self.assertRaises(ValueError):
            c.collect([0]*4, Cal([.9]*5), [.9])
    def test_nonfinite_rejected(self):
        for bad in (math.nan,math.inf,-math.inf):
            with self.assertRaises(ValueError):
                c.collect([0]*4, Cal([bad]*6), [.9])
    def test_unsorted_duplicate_cuts_rejected(self):
        for cuts in ([.9,.8],[.9,.9]):
            with self.assertRaises(ValueError):
                c.collect([0]*4, Cal([.9]*6), cuts)

if __name__ == '__main__':
    unittest.main()

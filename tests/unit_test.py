import unittest
from keplerian.Bodies import Earth
from keplerian.tools import Kepler

class TestKeplerMethods(unittest.TestCase):

    def test_sphere_of_influence(self):
        earth = Earth()
        kepler = Kepler(earth)
        self.assertAlmostEqual(kepler.sphere_of_influence(), 66182.656, places=3)

    def test_hohmann(self):
        earth = Earth()
        kepler = Kepler(earth)
        dv, tof = kepler.hohmann(ri=191.344, rf=35781.349)
        self.assertAlmostEqual(dv, 3.935224, places=2)
        self.assertAlmostEqual(tof, 0.21902970833, places=2)

    def test_bielliptic(self):
        earth = Earth()
        kepler = Kepler(earth)
        dv, tof = kepler.bielliptic(ri=191.344, rb=503873.000, rf=376310.000)
        self.assertAlmostEqual(dv, 3.904057, places=2)
        self.assertAlmostEqual(tof, 24.7466584583, places=2)

if __name__ == '__main__':
    unittest.main()
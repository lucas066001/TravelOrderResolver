import unittest
from travel_resolver.libs.sample.greeter import GreeterHelper


class TestGreeter(unittest.TestCase):

    def setUp(self):
        self._greeter = GreeterHelper()

    def test_correct_greeting(self):
        msg = "François"
        self.assertEqual(self._greeter.Greet(msg), "Hello ! " + msg)

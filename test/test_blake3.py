import unittest
from tinygrad.crypto.blake3 import blake3_hash

class TestBlake3(unittest.TestCase):
  def test_empty(self):
    digest = blake3_hash(b"", device="CPU")
    self.assertEqual(digest, blake3_hash(b"", device="CPU"))

if __name__ == "__main__":
  unittest.main()

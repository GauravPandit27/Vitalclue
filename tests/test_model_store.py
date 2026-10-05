"""
Model download tests.

Fake transports throughout - the point is the fallback and integrity logic, not the
network. The truncation cases matter most: a half-written model fails at load time with
an opaque deserialisation error, a long way from the cause.
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import model_store  # noqa: E402
from core.model_store import MIN_PLAUSIBLE_BYTES, ensure_file  # noqa: E402

GOOD_SIZE = MIN_PLAUSIBLE_BYTES + 1


def writer(size, name="fake"):
    def transport(url, dest):
        with open(dest, "wb") as f:
            f.write(b"\0" * size)
    transport.__name__ = name
    return transport


def failer(message, name="broken"):
    def transport(url, dest):
        raise RuntimeError(message)
    transport.__name__ = name
    return transport


class ModelStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "nested", "model.bin")
        self.original = model_store.TRANSPORTS

    def tearDown(self):
        model_store.TRANSPORTS = self.original
        shutil.rmtree(self.dir, ignore_errors=True)

    def use(self, *transports):
        model_store.TRANSPORTS = transports


class TestFallbackOrder(ModelStoreTestCase):
    def test_first_working_transport_is_used(self):
        self.use(writer(GOOD_SIZE, "first"), writer(GOOD_SIZE * 2, "second"))
        ensure_file("http://example/m", self.path, "model")
        self.assertEqual(os.path.getsize(self.path), GOOD_SIZE)

    def test_failing_transport_falls_through_to_the_next(self):
        self.use(failer("tls"), failer("no curl"), writer(GOOD_SIZE))
        ensure_file("http://example/m", self.path, "model")
        self.assertTrue(os.path.exists(self.path))

    def test_every_failure_is_reported_together(self):
        self.use(failer("cert verify failed", "urllib"), failer("revocation", "curl"))
        with self.assertRaises(RuntimeError) as ctx:
            ensure_file("http://example/m", self.path, "pose model")
        message = str(ctx.exception)
        for expected in ("pose model", "cert verify failed", "revocation", self.path):
            self.assertIn(expected, message)

    def test_missing_parent_directory_is_created(self):
        self.use(writer(GOOD_SIZE))
        ensure_file("http://example/m", self.path, "model")
        self.assertTrue(os.path.isdir(os.path.dirname(self.path)))


class TestIntegrity(ModelStoreTestCase):
    def test_truncated_download_is_rejected_and_retried(self):
        self.use(writer(10, "truncating"), writer(GOOD_SIZE, "healthy"))
        ensure_file("http://example/m", self.path, "model")
        self.assertEqual(os.path.getsize(self.path), GOOD_SIZE)

    def test_partial_file_is_never_left_behind(self):
        self.use(failer("died mid transfer"))
        with self.assertRaises(RuntimeError):
            ensure_file("http://example/m", self.path, "model")
        self.assertFalse(os.path.exists(self.path + ".part"))
        self.assertFalse(os.path.exists(self.path))

    def test_existing_truncated_file_is_replaced(self):
        """A model left over from an interrupted earlier run must not be trusted."""
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "wb") as f:
            f.write(b"\0" * 100)
        self.use(writer(GOOD_SIZE))
        ensure_file("http://example/m", self.path, "model")
        self.assertEqual(os.path.getsize(self.path), GOOD_SIZE)

    def test_existing_complete_file_is_not_redownloaded(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "wb") as f:
            f.write(b"\0" * GOOD_SIZE)
        self.use(failer("network must not be touched"))
        self.assertEqual(ensure_file("http://example/m", self.path, "model"), self.path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

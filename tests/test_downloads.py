"""Exercise the download paths used by the Hugging Face archive mirrors."""
import io
import os
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.files = {
            'RELEASE_v1.0.txt': b'Fixture release 1.0\n',
            'metadata.csv': b'y,split\n1,train\n',
        }
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
            for name, content in self.files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        payload = buffer.getvalue()
        self.requests = []
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                if self.path == '/resolve/revision/archive.tar.gz':
                    self.send_response(302)
                    self.send_header('Location', '/objects/archive.tar.gz')
                    self.end_headers()
                elif self.path == '/objects/archive.tar.gz':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/gzip')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.send_error(404)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.url = 'http://127.0.0.1:{}/resolve/revision/archive.tar.gz'.format(
            self.server.server_port)
        self.size = len(payload)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())

    def probe(self, unlabeled=False, equivalent_dataset=None):
        # Use the actual initialization helpers without constructing training arrays.
        cls = WILDSUnlabeledDataset if unlabeled else WILDSDataset
        dataset = object.__new__(cls)
        dataset._dataset_name = 'fixture_unlabeled' if unlabeled else 'fixture'
        dataset._version = None
        details = {'download_url': self.url, 'compressed_size': self.size}
        if equivalent_dataset is not None:
            details['equivalent_dataset'] = equivalent_dataset
        dataset._versions_dict = {'1.0': details}
        return dataset

    def assert_extracted(self, data_dir):
        for name, expected in self.files.items():
            with open(os.path.join(data_dir, name), 'rb') as stream:
                self.assertEqual(stream.read(), expected)
        self.assertFalse(os.path.exists(os.path.join(data_dir, 'archive.tar.gz')))

    def test_labeled_download_follows_redirect_and_reuses_local_data(self):
        dataset = self.probe()
        data_dir = dataset.initialize_data_dir(self.root.name, download=True)
        self.assertEqual(data_dir, os.path.join(self.root.name, 'fixture_v1.0'))
        self.assert_extracted(data_dir)
        self.assertEqual(dataset.initialize_data_dir(self.root.name, download=False), data_dir)
        self.assertEqual(self.requests, [
            '/resolve/revision/archive.tar.gz', '/objects/archive.tar.gz'])

    def test_separate_unlabeled_download_uses_its_own_directory(self):
        dataset = self.probe(unlabeled=True)
        data_dir = dataset.initialize_data_dir(self.root.name, download=True)
        self.assertEqual(data_dir, os.path.join(self.root.name, 'fixture_unlabeled_v1.0'))
        self.assert_extracted(data_dir)
        self.assertEqual(self.requests, [
            '/resolve/revision/archive.tar.gz', '/objects/archive.tar.gz'])

    def test_combined_archive_is_reused_by_unlabeled_loader(self):
        labeled_dir = self.probe().initialize_data_dir(self.root.name, download=True)
        unlabeled = self.probe(unlabeled=True, equivalent_dataset='fixture_v1.0')
        unlabeled_dir = unlabeled.initialize_data_dir(self.root.name, download=False)
        self.assertEqual(unlabeled_dir, labeled_dir)
        self.assert_extracted(unlabeled_dir)
        self.assertEqual(self.requests, [
            '/resolve/revision/archive.tar.gz', '/objects/archive.tar.gz'])


if __name__ == '__main__':
    unittest.main()

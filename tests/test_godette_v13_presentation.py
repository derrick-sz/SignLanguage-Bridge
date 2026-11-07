"""Avatar-only v13 publication keeps existing corpus and automatic jaw gates intact."""
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PILOTS = {
    'new-1918b4f42eac7d78-cap-0009': '02e3cbfaceca600f560b83ccfa5787590ab19ca19a4429fe7336db65e91547af',
    'pilot-greeting-answer-002': '0087d5024d320ea2564640f538e4cc7a8ff06f36898f9875c32d9add94ed8a6f',
    'pilot-greeting-how-004': '0087d5024d320ea2564640f538e4cc7a8ff06f36898f9875c32d9add94ed8a6f',
    'pilot-greeting-busy-005': '0087d5024d320ea2564640f538e4cc7a8ff06f36898f9875c32d9add94ed8a6f',
    'pilot-hospital-pay-013': '0087d5024d320ea2564640f538e4cc7a8ff06f36898f9875c32d9add94ed8a6f',
    'new-0ad6eaefc7647487-cap-0067': '91a8f852780699f4614279c7680c2c236aa4769de03f30ed5e2133b55e5b6c80',
}

class GodetteV13PresentationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('slbridge.animation.godette_v13_presentation'),
                             'independent v13 presentation builder missing')
        self.module = importlib.import_module('slbridge.animation.godette_v13_presentation')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.avatar = self.root / 'input.glb'
        self.avatar.write_bytes(b'fixture avatar; never modifies original bytes')
        self.sha = hashlib.sha256(self.avatar.read_bytes()).hexdigest()
        self.addCleanup(patch.stopall)
        patch.object(self.module, 'AVATAR_SHA256', self.sha).start()
        self.catalog = {'schema_version': 1, 'archive_sha256': 'a' * 64,
                        'assets': [{'id': key, 'sha256': sha, 'text': key, 'language_review': 'approved'}
                                   for key, sha in PILOTS.items()]}
        # An unrelated library/generation entry must not be copied, rewritten or excluded at source.
        self.catalog['assets'].append({'id': 'generated-new', 'sha256': 'b' * 64, 'text': '新句'})
        self.output = self.root / 'v13'

    def build(self):
        return self.module.build_godette_v13_presentation(self.output, avatar=self.avatar, catalog=self.catalog)

    def test_only_avatar_manifest_and_license_are_published_with_all_six_pilots(self):
        before = deepcopy(self.catalog)
        manifest = self.build()
        self.assertEqual(self.catalog, before)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {'manifest.json', 'godette-reference-v13.glb', 'ATTRIBUTION.md'})
        self.assertEqual(manifest['files'], {'godette-reference-v13.glb': self.sha})
        self.assertEqual(manifest['source_archive_sha256'], self.catalog['archive_sha256'])
        self.assertEqual(manifest['rig_profile']['id'], 'godette-v1')
        self.assertEqual({p['asset_id']: p['source_motion_sha256'] for p in manifest['pilots']}, PILOTS)
        self.assertFalse(any('export_file' in p for p in manifest['pilots']))
        repaired = next(p for p in manifest['pilots'] if p['asset_id'].endswith('cap-0067'))
        self.assertEqual(repaired['temporal_profile'], 'source-quaternion-temporal-v1')
        for p in manifest['pilots']:
            for field in ('language_review', 'motion_review', 'expression_review'):
                self.assertEqual(p[field], 'unverified')
        self.assertEqual(manifest['status'], 'research_candidate')
        self.assertFalse(manifest['export_supported'])
        self.assertFalse(manifest['automatic_nonmanual_supported'])
        self.assertNotIn('nonmanual_candidate', json.dumps(manifest))
        from slbridge.animation.presentation import PresentationBundle
        bundle = PresentationBundle(self.output, self.catalog['assets'])
        self.assertEqual(len(bundle.catalog()['pilots']), 6)
        self.assertEqual(bundle.resource(manifest['avatar'])[0], self.avatar.read_bytes())
        for name in ('ATTRIBUTION.md', 'manifest.json', '../input.glb', 'old.glb'):
            with self.assertRaises(KeyError):
                bundle.resource(name)
        attribution = (self.output / 'ATTRIBUTION.md').read_text()
        self.assertIn('zahlenmaler', attribution)
        self.assertIn('https://creativecommons.org/licenses/by/4.0/', attribution)

    @unittest.skipUnless(os.environ.get('ANIMATION_V13_AVATAR_GLB'), 'set ANIMATION_V13_AVATAR_GLB for real frozen asset')
    def test_real_frozen_avatar_is_published_with_its_own_identity_and_v11_is_rejected(self):
        patch.stopall()
        self.avatar = Path(os.environ['ANIMATION_V13_AVATAR_GLB'])
        before = self.avatar.read_bytes()
        manifest = self.build()
        self.assertEqual(manifest['avatar'], 'godette-reference-v13.glb')
        self.assertEqual(manifest['files'][manifest['avatar']], hashlib.sha256(before).hexdigest())
        self.assertEqual((self.output / manifest['avatar']).read_bytes(), before)
        self.assertFalse(manifest['automatic_nonmanual_supported'])
        self.assertEqual(manifest['processing_version'], 'godette-v13-neutral-manual-mouth-v1')
        for version in ('V11', 'V12'):
            old_avatar = os.environ.get('ANIMATION_' + version + '_AVATAR_GLB')
            if not old_avatar:
                continue
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.module.build_godette_v13_presentation(self.root / 'wrong-v11', avatar=old_avatar, catalog=self.catalog)
            self.assertFalse((self.root / 'wrong-v11').exists())

    def test_wrong_avatar_digest_fails_before_publication(self):
        self.avatar.write_bytes(b'different avatar with the same bone names')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_missing_changed_or_duplicate_pilot_never_gets_silently_dropped(self):
        original = deepcopy(self.catalog)
        variants = [original['assets'][1:],
                    [original['assets'][0] | {'sha256': 'f' * 64}, *original['assets'][1:]],
                    [*original['assets'], original['assets'][0]]]
        for assets in variants:
            with self.subTest(assets=assets):
                self.catalog['assets'] = assets
                with self.assertRaisesRegex(ValueError, 'source|duplicate'):
                    self.build()
                self.assertFalse(self.output.exists())
                self.assertEqual(list(self.root.glob('.v13-*')), [])

    def test_invalid_archive_identity_fails_before_publication(self):
        self.catalog['archive_sha256'] = 'unverified'
        with self.assertRaisesRegex(ValueError, 'archive'):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_replaced(self):
        self.build()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaisesRegex(ValueError, 'new directory'):
            self.build()
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, before)

    def test_publish_claim_rejects_a_concurrent_empty_destination(self):
        original_rename, original_mkdir = self.module.os.rename, self.module.os.mkdir
        injected = False
        def race(path):
            nonlocal injected
            if Path(path) == self.output and not injected:
                original_mkdir(self.output)
                injected = True
        def racing_rename(src, dst, *args, **kwargs):
            race(dst)
            return original_rename(src, dst, *args, **kwargs)
        def racing_mkdir(path, *args, **kwargs):
            race(path)
            return original_mkdir(path, *args, **kwargs)
        with patch.object(self.module.os, 'rename', side_effect=racing_rename), \
             patch.object(self.module.os, 'mkdir', side_effect=racing_mkdir):
            with self.assertRaisesRegex(ValueError, 'new directory'):
                self.build()
        self.assertTrue(injected)
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_rejects_malformed_archive_without_output(self):
        import contextlib
        import io
        from scripts.animation.build_godette_v13_presentation import main
        archive = self.root / 'invalid.zip'; archive.write_bytes(b'not a zip archive')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--archive', str(archive), '--avatar', str(self.avatar), '--output', str(self.output)])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_real_registry_cli_bundle_and_api_chain_with_fixture_trust_anchors(self):
        # The archive/parser/API are real. Only pinned avatar/source hashes are
        # replaced with explicitly synthetic fixture hashes, never production claims.
        import zipfile
        from fastapi.testclient import TestClient
        from slbridge.api import create_app
        from slbridge.animation.registry import MotionRegistry
        from slbridge.animation.presentation import PresentationBundle
        from scripts.animation.build_godette_v13_presentation import main
        skeleton = {'parents': [-1] + [0] * 54, 'restJoints': [[0., float(i), 0.] for i in range(55)]}
        clips = [{'id': identity, 'text': identity, 'frames': 2, 'fps': 25,
                  'poses': [[[0., 0., 0.] for _ in range(55)] for _ in range(2)]} for identity in PILOTS]
        data = {'demo/t08-motion.json': json.dumps(skeleton | {'clips': clips}).encode(),
                'demo/t08-candidates.json': json.dumps(skeleton | {'clips': []}).encode(),
                'demo/experimental-index.json': json.dumps(skeleton | {'clips': []}).encode(),
                'avatar/model.glb': b'fixture original model',
                'avatar/three.module.js': b'fixture three',
                'avatar/jsm/loaders/GLTFLoader.js': b'fixture loader',
                'avatar/jsm/utils/BufferGeometryUtils.js': b'fixture utils'}
        archive = self.root / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as target:
            for name, value in data.items():
                target.writestr(name, value)
            target.writestr('BUILD.json', json.dumps({'files': {name: hashlib.sha256(value).hexdigest()
                                                               for name, value in data.items()}}))
        before = archive.read_bytes()
        registry = MotionRegistry(archive)
        source_sha = registry.catalog()['assets'][0]['sha256']
        resources = self.module.files(self.module.__package__)
        fixture_resources = self.root / 'fixture-resources'; fixture_resources.mkdir()
        preset = json.loads(resources.joinpath('godette_v13_manifest.json').read_text())
        for pilot in preset['pilots']:
            pilot['source_motion_sha256'] = source_sha
        (fixture_resources / 'godette_v13_manifest.json').write_text(json.dumps(preset))
        (fixture_resources / 'godette_v13_attribution.md').write_text(
            resources.joinpath('godette_v13_attribution.md').read_text())
        with patch.object(self.module, 'files', return_value=fixture_resources):
            self.assertEqual(main(['--archive', str(archive), '--avatar', str(self.avatar),
                                   '--expected-sha256', registry.sha256, '--output', str(self.output)]), 0)
        self.assertEqual(archive.read_bytes(), before)
        registry.presentation = PresentationBundle(self.output, registry.catalog()['assets'])
        with TestClient(create_app({}, animation_registry=registry)) as client:
            info = client.get('/v1/animation/presentation').json()
            self.assertEqual(info['avatar_sha256'], self.sha)
            self.assertEqual(len(info['pilots']), 6)
            self.assertEqual(client.get('/animation/presentation/' + info['avatar']).content, self.avatar.read_bytes())
            self.assertEqual(client.get('/animation/presentation/manifest.json').status_code, 404)
            self.assertEqual(client.get('/animation/presentation/ATTRIBUTION.md').status_code, 404)
            self.assertEqual(client.get('/animation/').status_code, 200)
            self.assertIn('https://creativecommons.org/licenses/by/4.0/', client.get('/animation/').text)
        self.assertEqual(archive.read_bytes(), before)

    def test_failed_bundle_validation_cleans_staging_directory(self):
        with patch.object(self.module, 'PresentationBundle', side_effect=ValueError('validation failure')):
            with self.assertRaisesRegex(ValueError, 'validation failure'):
                self.build()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.v13-*')), [])

if __name__ == '__main__':
    unittest.main()

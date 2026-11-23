"""Pure schema contracts. Fictional fixtures establish no runtime or quality."""
from copy import deepcopy
import hashlib
import unittest

from nvfp4_evidence_fixtures import contract, make_evidence_fixture


class EvidenceTests(unittest.TestCase):
    def test_complete_three_way_binding_and_original_bytes(self):
        manifest, report, calibration, config = make_evidence_fixture()
        hardware = manifest['backend']['hardware']
        self.assertIs(contract.validate_quantization_manifest(manifest), manifest)
        self.assertIs(contract.validate_backend_payload(report, manifest['backend'], hardware), report)
        self.assertIs(contract.validate_calibration_binding(calibration, manifest, hardware), calibration)
        contract.validate_evidence_config(config,
            backend_report_sha256=hashlib.sha256(contract.canonical(report)).hexdigest(),
            calibration_sha256=hashlib.sha256(contract.canonical(calibration)).hexdigest(),
            bf16_control=calibration['bf16_control'])

    def test_schema_does_not_promote_support_or_single_fixture_to_quality(self):
        manifest, report, calibration, _ = make_evidence_fixture()
        for field, value in (('support_verified', False), ('policy_executed', True), ('quality_passed', True)):
            changed = deepcopy(report)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_backend_payload(changed, manifest['backend'], manifest['backend']['hardware'])
        calibration['quality_validated'] = True
        with self.assertRaises(ValueError):
            contract.validate_calibration_binding(calibration, manifest, manifest['backend']['hardware'])

    def test_exact51_order_and_packed_shapes_are_required(self):
        manifest, _, _, _ = make_evidence_fixture()
        for mutation in ('missing', 'order', 'shape', 'fallback', 'bool_dimension'):
            changed = deepcopy(manifest)
            if mutation == 'missing': changed['sites'].pop()
            elif mutation == 'order': changed['sites'].reverse()
            elif mutation == 'shape': changed['sites'][0]['quantized_weight']['shape'][1] = 2048
            elif mutation == 'fallback': changed['lifecycle']['fallback_permitted'] = True
            else: changed['sites'][0]['input_global_scale']['shape'][0] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate_quantization_manifest(changed)

    def test_all_calibration_fields_and_source_weight_hash_are_bound(self):
        manifest, _, calibration, _ = make_evidence_fixture()
        name = contract.SITE_ORDER[0]
        for field in ('input_sha256', 'weight_sha256', 'input_amax'):
            changed = deepcopy(calibration)
            changed['sites'][name][field] = 2689.0 if field == 'input_amax' else '0' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_calibration_binding(changed, manifest, manifest['backend']['hardware'])
        calibration['bf16_control']['recipe_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            contract.validate_calibration_binding(calibration, manifest, manifest['backend']['hardware'])

    def test_raw_file_and_canonical_config_anchors_remain_distinct(self):
        manifest, _, calibration, _ = make_evidence_fixture()
        self.assertNotEqual(calibration['control']['config_sha256'], calibration['bf16_control']['config_sha256'])
        contract.validate_calibration_binding(calibration, manifest, manifest['backend']['hardware'])
        calibration['control']['config_canonical_sha256'] = calibration['control']['config_sha256']
        with self.assertRaises(ValueError):
            contract.validate_calibration(calibration)

    def test_gate_up_shared_actual_input_artifact_is_bound_without_other_layer_alias(self):
        manifest, _, calibration, _ = make_evidence_fixture()
        gate = calibration['sites']['prefix.0.gate_proj']['input_npz']['path']
        self.assertEqual(calibration['sites']['prefix.0.up_proj']['input_npz']['path'], gate)
        contract.validate_calibration_binding(calibration, manifest, manifest['backend']['hardware'])
        calibration['sites']['prefix.0.up_proj']['input_npz']['path'] = 'prefix_01_gate_proj.npz'
        with self.assertRaises(ValueError):
            contract.validate_calibration(calibration)

    def test_actual_fp32_bits_and_content_hash_cannot_be_relabelled(self):
        manifest, _, _, _ = make_evidence_fixture()
        for mutation in ('bits', 'hash', 'value'):
            changed = deepcopy(manifest)
            row = changed['sites'][0]
            if mutation == 'bits': row['scalar_values']['alpha']['fp32_bits'] = '0x40000000'
            elif mutation == 'hash': row['alpha']['content_sha256'] = '0' * 64
            else: row['scalar_values']['alpha']['value'] = 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate_quantization_manifest(changed)

    def test_native_binary_import_origin_operator_and_source_contract_are_fixed(self):
        manifest, _, _, _ = make_evidence_fixture()
        for mutation in ('binary', 'origin', 'schema', 'dispatch', 'sm'):
            changed = deepcopy(manifest['backend'])
            native = changed['native_runtime']
            if mutation == 'binary': native['files']['extension_sha256'] = '0' * 64
            elif mutation == 'origin': native['imported_modules']['vllm']['path'] = '/foreign/vllm/__init__.py'
            elif mutation == 'schema': native['native_schemas']['gemm'] += ' altered'
            elif mutation == 'dispatch':
                native['implementation']['dispatch'] = changed['implementation']['dispatch'] = 'fictional fallback'
            else: native['runtime']['capability'] = [8, 9]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate_backend_manifest(changed)

    def test_same_physical_hardware_required_for_report_calibration_and_owner(self):
        manifest, report, calibration, _ = make_evidence_fixture()
        other = deepcopy(manifest['backend']['hardware'])
        other['uuid'] = 'GPU-22222222-2222-2222-2222-222222222222'
        with self.assertRaises(ValueError):
            contract.validate_backend_payload(report, manifest['backend'], other)
        with self.assertRaises(ValueError):
            contract.validate_calibration_binding(calibration, manifest, other)

    def test_torch_prefixless_uuid_matches_same_physical_bytes_without_rewriting_record(self):
        manifest, _, _, _ = make_evidence_fixture()
        backend = manifest['backend']
        backend['hardware']['uuid'] = 'GPU-79bec5af-28e1-4a0b-6839-7b80d764ab6b'
        original = '79BEC5AF-28E1-4A0B-6839-7B80D764AB6B'
        backend['native_runtime']['hardware']['uuid'] = original
        contract.validate_backend_manifest(backend)
        self.assertEqual(backend['native_runtime']['hardware']['uuid'], original)

    def test_prefixless_wrong_uuid_and_malformed_uuid_remain_rejected(self):
        manifest, _, _, _ = make_evidence_fixture()
        for uuid in ('22222222-2222-2222-2222-222222222222',
                     '11111111111111111111111111111111',
                     '11111111-1111-1111-1111-11111111111g'):
            backend = deepcopy(manifest['backend'])
            backend['native_runtime']['hardware']['uuid'] = uuid
            with self.subTest(uuid=uuid), self.assertRaises(ValueError):
                contract.validate_backend_manifest(backend)

    def test_exact_five_config_fields_and_two_raw_sha_bindings_required(self):
        _, _, _, config = make_evidence_fixture()
        for field in ('backend_report_sha256', 'calibration_sha256'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_evidence_config(config, **{field: '0' * 64})
        config['quality_validated'] = True
        with self.assertRaises(ValueError):
            contract.validate_evidence_config(config)

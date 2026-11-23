"""Fictional, pure-stdlib schema fixtures. Never use as production evidence.

Shared by remote owner/harness CPU contracts to avoid divergent declarations.
This module neither imports Torch nor executes any numerical validation.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / 'src/vvla_repro_pi05/lite/prefix_nvfp4_evidence.py'
_SPEC = importlib.util.spec_from_file_location('_nvfp4_fixture_pure_contract', _PATH)
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)


def tensor_record(shape, dtype, content_sha256='c' * 64):
    return dict(shape=shape, stride=[shape[1], 1] if len(shape) == 2 else [1],
                dtype=dtype, content_sha256=content_sha256)


def make_evidence_fixture(hardware=None):
    """Return (manifest, original_backend_payload, calibration, config).

    All source/content identities except pinned public ABI constants are
    fictional. Canonical serialization is used only for these test bytes.
    """
    hardware = deepcopy(hardware or dict(name='NVIDIA RTX PRO 6000 Blackwell Server Edition',
        uuid='GPU-11111111-1111-1111-1111-111111111111', driver_version='fictional',
        memory_mib=97887, hostname='fictional-test-host'))
    anchors = {key: 'b' * 64 for key in contract.CONTROL_KEYS}
    control = dict(episode_id='libero_spatial/0/0', config_sha256='e' * 64,
        config_canonical_sha256=anchors['config_sha256'],
        formal_identity_sha256=anchors['identity_sha256'], formal_content_sha256=anchors['content_sha256'],
        formal_frozen_sha256='f' * 64, preflight_identity_sha256='f' * 64,
        preflight_inventory_sha256='f' * 64, noise_sha256='f' * 64, raw_observation_sha256='f' * 64,
        raw_fixture=dict(path='fictional/raw.npz', sha256='f' * 64),
        prepared_fixture=dict(path='fictional/prepared.npz', sha256='f' * 64))
    calibration = dict(schema=1, version=contract.CALIBRATION_VERSION, precision=contract.PRECISION,
        scope='prefix_mlp_layers_0_16', site_order=list(contract.SITE_ORDER), capacity=576,
        checkpoint_sha256=contract.CHECKPOINT_SHA256, hardware=hardware, control=control,
        bf16_control=anchors, activation_report_sha256='a' * 64, single_fixture=True,
        quality_validated=False, calibration_formula=contract.CALIBRATION_FORMULA, sites={})
    scalar = dict(value=1.0, fp32_bits='0x3f800000')
    scalar_sha = hashlib.sha256(bytes.fromhex('0000803f')).hexdigest()
    sites = []
    for index, name in enumerate(contract.SITE_ORDER):
        layer, projection = index // 3, name.rsplit('.', 1)[1]
        k, n = (16384, 2048) if index % 3 == 2 else (2048, 16384)
        record = dict(input_amax=2688.0, input_floor_applied=False, floor=1e-12,
            input_global_scale=deepcopy(scalar),
            input_npz=dict(path=f'prefix_{layer:02d}_{"gate_proj" if projection == "up_proj" else projection}.npz', sha256='d' * 64),
            input_sha256='e' * 64, input_shape=[1, 576, k], weight_sha256='c' * 64, weight_shape=[n, k],
            checkpoint_suffix=f'paligemma.model.language_model.layers.{layer}.mlp.{projection}.weight')
        calibration['sites'][name] = record
        sites.append(dict(name=name,
            module_path=f'paligemma_with_expert.paligemma.model.language_model.layers.{layer}.mlp.{projection}',
            module_type='torch.nn.modules.linear.Linear', bias=None,
            original_weight=tensor_record([n, k], 'bfloat16'),
            quantized_weight=tensor_record([n, k // 2], 'uint8'),
            weight_scale=tensor_record([n, k // 16], 'float8_e4m3fn'),
            **{field: tensor_record([1], 'float32', scalar_sha)
               for field in ('input_global_scale', 'weight_global_scale', 'alpha')},
            scalar_values={field: deepcopy(scalar) for field in ('input_global_scale', 'weight_global_scale', 'alpha')},
            calibration=deepcopy(record)))
    root = '/fictional/native'
    extension = root + '/vllm/_C_stable_libtorch.abi3.so'
    files = dict(native_path=root, extension_path=extension, extension_sha256=contract.EXTENSION_SHA256,
        installed_source_files={'vllm/__init__.py': dict(sha256='d' * 64, bytes=10),
            'vllm/_C_stable_libtorch.abi3.so': dict(sha256=contract.EXTENSION_SHA256, bytes=100)},
        installed_record_sha256='d' * 64, preparation_receipt_sha256=None,
        archive_identity_scope='receipt reference only; archive is not re-read by this primitive loader')
    implementation = contract.expected_native_implementation({'prefix_nvfp4_native.py': 'd' * 64})
    native_runtime = dict(version='pi05-prefix-nvfp4-native-sm120-v1', variant=contract.VARIANT,
        runtime=deepcopy(contract.RUNTIME), device_index=0, files=files,
        imported_modules={'vllm': dict(path=root + '/vllm/__init__.py', sha256='d' * 64),
            'vllm._C_stable_libtorch': dict(path=extension, sha256=contract.EXTENSION_SHA256)},
        native_schemas=deepcopy(contract.NATIVE_SCHEMAS),
        hardware=dict(name=hardware['name'], uuid=hardware['uuid'], total_memory=hardware['memory_mib'] * 1048576,
                      capability=[12, 0]), implementation=deepcopy(implementation), import_succeeded=True,
        execution_validated=False, quality_validated=False)
    report = dict(status='complete', started=0.0, finished=1.0, diagnostic_only=True,
        policy_executed=False, quality_passed=None, synthetic_inputs=True, support_verified=True,
        source_sha256='a' * 64,
        backend=dict(torch_version=contract.RUNTIME['torch'], torch_git=contract.RUNTIME['torch_git'],
            cuda=contract.RUNTIME['cuda'], capability=[12, 0], gpu_name=hardware['name'], uuid=hardware['uuid'],
            vllm_version='0.32.0', extension=extension, extension_sha256=contract.EXTENSION_SHA256,
            schemas=list(contract.NATIVE_SCHEMAS.values())),
        rows=[dict(m=m, k=k, n=n, analytic_exact=True, capture_refresh_restore_exact=True)
              for m, k, n in ((576, 2048, 16384), (640, 2048, 16384), (576, 16384, 2048), (640, 16384, 2048))])
    backend = dict(variant=contract.VARIANT, report_sha256=contract.digest(report),
        report_source_sha256=report['source_sha256'], hardware=deepcopy(hardware), native_runtime=native_runtime,
        support_scope='direct native operators; policy quality and actual wrapper contracts are separate',
        implementation=implementation)
    manifest = dict(schema=1, precision=contract.PRECISION, implementation='pi05-prefix-nvfp4-cutlass-v1',
        checkpoint_sha256=contract.CHECKPOINT_SHA256, scope='prefix_mlp_layers_0_16',
        precision_boundaries=dict(vision='unchanged_selected_bf16_control', prefix_attention='unchanged_selected_bf16_control',
            expert_projections='unchanged_selected_bf16_control', normalization='unchanged_selected_control_FP32_boundaries',
            residual_euler='unchanged_selected_control_FP32_boundaries', action_time_projections='unchanged_selected_control_FP32_boundaries'),
        backend=backend, calibration=dict(sha256=contract.digest(calibration), version=contract.CALIBRATION_VERSION,
            activation_report_sha256=calibration['activation_report_sha256'], scope=calibration['scope'], single_fixture=True,
            capacity=576, quality_validated=False, formula=contract.CALIBRATION_FORMULA,
            calibration_source_scope='one BF16 fixture, not formal calibration', bf16_control=deepcopy(anchors)),
        quantizer=dict(source_sha256=contract.digest(implementation['source_files_sha256']), nonlinear_source_sha256='d' * 64,
            variant=contract.VARIANT, format='NVFP4 E2M1 values plus E4M3 block scales and FP32 global scales',
            input_dtype='bfloat16', output_dtype='bfloat16', accumulator_dtype='float32',
            weight_layout='contiguous_NK_packed_K/2', block_size=16, scale_layout='CUTLASS swizzle128x4 padded per native ABI',
            input_global_scale_formula=contract.CALIBRATION_FORMULA, weight_global_scale_formula=contract.WEIGHT_FORMULA,
            alpha_formula=contract.ALPHA_FORMULA, online_conversion=True, weight_packing='cold_only',
            gate_up='independent_quantization', nonlinear_boundary='BF16 gate/up -> gelu_tanh_mul -> BF16 down input'),
        lifecycle=dict(version='pi05-prefix-nvfp4-live-owner-v1', source_sha256='d' * 64, enabled=True,
            owner='Pi05NativeGraphRunner', original_parameters_preserved=True,
            content_hash_scope='cold_packing_epoch; changes require retirement and new identity/preflight',
            live_checks=['module_forward_bindings', 'original_weight_identity', 'quantized_weight_identity',
                'weight_scale_identity', 'backend_identity', 'unique_runner_generation', 'no_extra_quantized_sites',
                'unchanged_nonprefix_precision', 'supported_capacity_dispatch', 'input_global_scale_identity',
                'weight_global_scale_identity', 'alpha_identity', 'calibration_byte_identity'], fallback_permitted=False),
        sites=sites)
    evidence = dict(backend_report='/fictional/BACKEND_REPORT.json', backend_report_sha256=backend['report_sha256'],
        calibration='/fictional/CALIBRATION.json', calibration_sha256=manifest['calibration']['sha256'], bf16_control=anchors)
    return manifest, report, calibration, evidence

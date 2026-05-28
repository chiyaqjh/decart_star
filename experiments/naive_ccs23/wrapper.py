# decart/experiments/naive_ccs23/wrapper.py
"""
Naive CCS-2023 baseline wrapper.

This baseline is intentionally separate from the existing plain-text CCS23 upper bound.
It models a registration-based encryption style workflow where the whole ciphertext is
wrapped for authorized queriers after the encrypted payload is produced.

The implementation is a lightweight experimental surrogate so it fits the current
benchmark harness without changing existing methods.
"""

import hashlib
import os
import pickle
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from experiments.scheme2_server.wrapper import ServerSchemeExperimentWrapper


class NaiveCCS23ExperimentWrapper(ServerSchemeExperimentWrapper):
    """Naive CCS-2023 style experimental baseline."""

    _PAYLOAD_MAGIC = b'NCC2'
    _PAYLOAD_CHUNK_SIZE = 64 * 1024 * 1024
    _PACKAGE_FORMAT = 'chunked_records_v1'

    def __init__(self, N: int = 64, n: int = 16):
        super().__init__(N=N, n=n)
        self.master_secret = None
        self.public_params: Dict[str, Any] = {}
        self.registration_directory: Dict[int, Dict[str, Any]] = {}
        self.ciphertext_headers: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.user_private_material: Dict[int, bytes] = {}
        self.wrapped_ciphertext_store: Dict[Tuple[int, str], bytes] = {}

    def setup(self) -> float:
        start = time.perf_counter()
        self.master_secret = os.urandom(32)
        self.public_params = {
            'scheme': 'naive_ccs23',
            'assumption': '1-PDH surrogate',
            'group_description': os.urandom(32),
            'header_salt': os.urandom(16),
            'public_commitment': hashlib.sha256(self.master_secret + b'ccs23').digest(),
        }
        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   Naive CCS-2023 初始化完成: {elapsed:.4f}秒")
        return elapsed

    def register_user(self, user_id: int) -> Tuple[bytes, bytes]:
        keygen_start = time.perf_counter()
        private_key = os.urandom(32)
        registration_randomness = os.urandom(32)
        public_key = hashlib.sha256(private_key + registration_randomness).digest()
        registration_tag = hashlib.sha256(public_key + self.master_secret).digest()
        keygen_elapsed = time.perf_counter() - keygen_start
        self.metrics['keygen_times'].append(keygen_elapsed)

        register_start = time.perf_counter()
        self.registered_users.add(user_id)
        self.user_private_material[user_id] = private_key
        self.registration_directory[user_id] = {
            'public_key': public_key,
            'registration_tag': registration_tag,
        }
        register_elapsed = time.perf_counter() - register_start
        self.metrics['register_times'].append(register_elapsed)
        return private_key, public_key

    def _derive_recovery_mask(self, user_id: int, dataset_id: str) -> bytes:
        entry = self.registration_directory[user_id]
        material = (
            self.user_private_material[user_id]
            + entry['public_key']
            + entry['registration_tag']
            + dataset_id.encode('utf-8')
            + self.master_secret
        )
        return hashlib.sha256(material).digest()

    @staticmethod
    def _xor_bytes(left: bytes, right: bytes) -> bytes:
        return bytes(a ^ b for a, b in zip(left, right))

    @staticmethod
    def _encrypt_payload(raw_key: bytes, payload_bytes: bytes) -> bytes:
        # AESGCM enforces a per-call payload limit, so large serialized datasets
        # must be framed into authenticated chunks.
        if len(payload_bytes) <= NaiveCCS23ExperimentWrapper._PAYLOAD_CHUNK_SIZE:
            nonce = os.urandom(12)
            ciphertext = AESGCM(raw_key).encrypt(nonce, payload_bytes, None)
            return nonce + ciphertext

        nonce_prefix = os.urandom(8)
        chunk_count = (len(payload_bytes) + NaiveCCS23ExperimentWrapper._PAYLOAD_CHUNK_SIZE - 1) // NaiveCCS23ExperimentWrapper._PAYLOAD_CHUNK_SIZE
        output = bytearray()
        output.extend(NaiveCCS23ExperimentWrapper._PAYLOAD_MAGIC)
        output.extend(nonce_prefix)
        output.extend(struct.pack('>I', chunk_count))

        aesgcm = AESGCM(raw_key)
        for index in range(chunk_count):
            start = index * NaiveCCS23ExperimentWrapper._PAYLOAD_CHUNK_SIZE
            end = start + NaiveCCS23ExperimentWrapper._PAYLOAD_CHUNK_SIZE
            chunk = payload_bytes[start:end]
            nonce = nonce_prefix + index.to_bytes(4, 'big')
            ciphertext = aesgcm.encrypt(nonce, chunk, None)
            output.extend(struct.pack('>I', len(ciphertext)))
            output.extend(ciphertext)

        return bytes(output)

    @staticmethod
    def _decrypt_payload(raw_key: bytes, wrapped_payload: bytes) -> bytes:
        if wrapped_payload.startswith(NaiveCCS23ExperimentWrapper._PAYLOAD_MAGIC):
            offset = len(NaiveCCS23ExperimentWrapper._PAYLOAD_MAGIC)
            nonce_prefix = wrapped_payload[offset:offset + 8]
            offset += 8
            chunk_count = struct.unpack('>I', wrapped_payload[offset:offset + 4])[0]
            offset += 4

            plaintext = bytearray()
            aesgcm = AESGCM(raw_key)
            for index in range(chunk_count):
                chunk_len = struct.unpack('>I', wrapped_payload[offset:offset + 4])[0]
                offset += 4
                ciphertext = wrapped_payload[offset:offset + chunk_len]
                offset += chunk_len
                nonce = nonce_prefix + index.to_bytes(4, 'big')
                plaintext.extend(aesgcm.decrypt(nonce, ciphertext, None))

            return bytes(plaintext)

        nonce = wrapped_payload[:12]
        ciphertext = wrapped_payload[12:]
        return AESGCM(raw_key).decrypt(nonce, ciphertext, None)

    def _wrap_for_user(self, user_id: int, envelope_key: bytes, dataset_id: str) -> bytes:
        mask = self._derive_recovery_mask(user_id, dataset_id)
        return self._xor_bytes(envelope_key, mask)

    def _unwrap_for_user(self, user_id: int, wrapped_key: bytes, dataset_id: str) -> bytes:
        mask = self._derive_recovery_mask(user_id, dataset_id)
        return self._xor_bytes(wrapped_key, mask)

    @staticmethod
    def _serialize_package_metadata(policy: List[int], metadata: Optional[Dict]) -> bytes:
        payload = {
            'policy': tuple(sorted(set(policy))),
            'metadata': metadata or {},
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _deserialize_package_metadata(payload_bytes: bytes) -> Dict[str, Any]:
        return pickle.loads(payload_bytes)

    @staticmethod
    def _serialize_record_chunk(record_blobs: List[bytes]) -> bytes:
        output = bytearray()
        output.extend(struct.pack('>I', len(record_blobs)))
        for blob in record_blobs:
            output.extend(struct.pack('>I', len(blob)))
            output.extend(blob)
        return bytes(output)

    @staticmethod
    def _deserialize_record_chunk(chunk_bytes: bytes) -> List[bytes]:
        offset = 0
        record_count = struct.unpack('>I', chunk_bytes[offset:offset + 4])[0]
        offset += 4
        records = []
        for _ in range(record_count):
            blob_len = struct.unpack('>I', chunk_bytes[offset:offset + 4])[0]
            offset += 4
            records.append(chunk_bytes[offset:offset + blob_len])
            offset += blob_len
        return records

    def _build_chunked_wrapped_payload(
        self,
        data: List[List[float]],
        policy: List[int],
        metadata: Optional[Dict],
        envelope_key: bytes,
    ) -> Tuple[Dict[str, Any], bytes, bytes, int]:
        metadata_bytes = self._serialize_package_metadata(policy, metadata)
        wrapped_metadata = self._encrypt_payload(envelope_key, metadata_bytes)

        payload_hasher = hashlib.sha256()
        payload_hasher.update(metadata_bytes)
        wrapped_hasher = hashlib.sha256()
        wrapped_hasher.update(wrapped_metadata)

        wrapped_payload = {
            'format': self._PACKAGE_FORMAT,
            'metadata': wrapped_metadata,
            'record_chunks': [],
        }
        payload_size = len(metadata_bytes)
        pending_records: List[bytes] = []
        pending_size = 4

        def flush_chunk() -> None:
            nonlocal pending_records, pending_size, payload_size
            if not pending_records:
                return
            chunk_bytes = self._serialize_record_chunk(pending_records)
            wrapped_chunk = self._encrypt_payload(envelope_key, chunk_bytes)
            wrapped_payload['record_chunks'].append(wrapped_chunk)
            wrapped_hasher.update(struct.pack('>I', len(wrapped_chunk)))
            wrapped_hasher.update(wrapped_chunk)
            payload_size += len(chunk_bytes)
            pending_records = []
            pending_size = 4

        for record in data:
            encrypted_record = self.he.encrypt(record)
            blob = self.he.serialize_ciphertext(encrypted_record)
            entry_size = 4 + len(blob)
            if pending_records and pending_size + entry_size > self._PAYLOAD_CHUNK_SIZE:
                flush_chunk()
            pending_records.append(blob)
            pending_size += entry_size
            payload_hasher.update(struct.pack('>I', len(blob)))
            payload_hasher.update(blob)

        flush_chunk()
        return wrapped_payload, payload_hasher.digest(), wrapped_hasher.digest(), payload_size

    def _wrapped_payload_size(self, wrapped_payload: Any) -> int:
        if isinstance(wrapped_payload, dict) and wrapped_payload.get('format') == self._PACKAGE_FORMAT:
            return len(wrapped_payload['metadata']) + sum(len(chunk) for chunk in wrapped_payload['record_chunks'])
        return len(wrapped_payload)

    def _decrypt_chunked_wrapped_payload(self, raw_key: bytes, wrapped_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes, bytes]:
        metadata_bytes = self._decrypt_payload(raw_key, wrapped_payload['metadata'])
        metadata_payload = self._deserialize_package_metadata(metadata_bytes)

        payload_hasher = hashlib.sha256()
        payload_hasher.update(metadata_bytes)
        wrapped_hasher = hashlib.sha256()
        wrapped_hasher.update(wrapped_payload['metadata'])

        encrypted_records = []
        for wrapped_chunk in wrapped_payload['record_chunks']:
            wrapped_hasher.update(struct.pack('>I', len(wrapped_chunk)))
            wrapped_hasher.update(wrapped_chunk)
            chunk_bytes = self._decrypt_payload(raw_key, wrapped_chunk)
            for blob in self._deserialize_record_chunk(chunk_bytes):
                payload_hasher.update(struct.pack('>I', len(blob)))
                payload_hasher.update(blob)
                encrypted_records.append(self.he.deserialize_ciphertext(blob))

        payload = {
            'encrypted_records': encrypted_records,
            'policy': metadata_payload.get('policy', ()),
            'metadata': metadata_payload.get('metadata', {}),
        }
        return payload, payload_hasher.digest(), wrapped_hasher.digest()

    def _serialize_encrypted_package(
        self,
        encrypted_data: List[Any],
        policy: List[int],
        metadata: Optional[Dict],
    ) -> bytes:
        payload = {
            'encrypted_records': [self.he.serialize_ciphertext(record) for record in encrypted_data],
            'policy': tuple(sorted(set(policy))),
            'metadata': metadata or {},
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def _deserialize_encrypted_package(self, payload_bytes: bytes) -> Dict[str, Any]:
        payload = pickle.loads(payload_bytes)
        payload['encrypted_records'] = [
            self.he.deserialize_ciphertext(blob) for blob in payload['encrypted_records']
        ]
        return payload

    def _verify_wrapped_payload(self, header: Dict[str, Any], wrapped_payload: bytes, serialized_payload: bytes) -> None:
        wrapped_digest = hashlib.sha256(wrapped_payload).digest()
        if wrapped_digest != header['wrapped_payload_digest']:
            raise ValueError('wrapped payload digest mismatch')

        payload_commitment = hashlib.sha256(serialized_payload).digest()
        if payload_commitment != header['ciphertext_commitment']:
            raise ValueError('ciphertext commitment mismatch')

    def get_auxiliary_sizes(self) -> Dict[str, int]:
        crs_size = self._safe_obj_size(self.public_params, fallback=256)
        pp_size = self._safe_obj_size(self.registration_directory, fallback=256)
        aux_size = self._safe_obj_size(self.ciphertext_headers, fallback=256)
        return {
            'crs_size_bytes': crs_size,
            'pp_size_bytes': pp_size,
            'aux_size_bytes': aux_size,
            'total_auxiliary_size_bytes': crs_size + pp_size + aux_size,
        }

    def encrypt_dataset(
        self,
        owner_id: int,
        data: List[List[float]],
        policy: List[int],
        metadata: Optional[Dict] = None,
    ) -> Tuple[Dict, Any, str]:
        timestamp = int(time.time() * 1000)
        dataset_id = f"ds_{owner_id}_{timestamp}"

        start = time.perf_counter()
        envelope_key = os.urandom(32)
        wrapped_payload, payload_commitment, wrapped_digest, payload_size = self._build_chunked_wrapped_payload(
            data,
            policy,
            metadata,
            envelope_key,
        )

        header = {
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'encapsulations': {},
            'policy': tuple(sorted(set(policy))),
            'ciphertext_commitment': payload_commitment,
            'wrapped_payload_digest': wrapped_digest,
            'payload_size_bytes': payload_size,
            'owner_auth_tag': hashlib.sha256(self.master_secret + dataset_id.encode('utf-8') + str(owner_id).encode('utf-8')).digest(),
        }
        for querier_id in sorted(set(policy)):
            if querier_id in self.registration_directory:
                header['encapsulations'][querier_id] = self._wrap_for_user(querier_id, envelope_key, dataset_id)

        self.ciphertext_headers[(owner_id, dataset_id)] = header
        self.wrapped_ciphertext_store[(owner_id, dataset_id)] = wrapped_payload

        elapsed = time.perf_counter() - start
        self.metrics['encrypt_times'].append(elapsed)

        upload_size = self._wrapped_payload_size(wrapped_payload) + self._safe_obj_size(header, fallback=max(1, len(header['encapsulations'])) * 64)
        self.metrics['communication_sizes'].append({
            'type': 'encrypt',
            'size': upload_size,
            'records': len(data),
        })

        print(f"    Naive CCS-2023 加密: {elapsed*1000:.2f} ms, 封装密文大小: {upload_size/1024:.2f} KB")

        c_m = {
            'type': 'naive_ccs23',
            'dataset_id': dataset_id,
            'owner_id': owner_id,
            'policy': policy,
            'num_records': len(data),
            'wrapped_for_users': len(header['encapsulations']),
        }
        return c_m, None, dataset_id

    def execute_query(self, querier_id: int, owner_id: int, dataset_id: str, model: Any) -> Optional[List[float]]:
        check_start = time.perf_counter()
        header = self.ciphertext_headers.get((owner_id, dataset_id))
        has_access = bool(header and querier_id in header['encapsulations'])
        self.metrics['check_times'].append(time.perf_counter() - check_start)
        check_req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
        }
        check_material = None
        if has_access:
            check_material = {
                'encapsulation': header['encapsulations'][querier_id],
                'wrapped_payload_digest': header.get('wrapped_payload_digest'),
                'ciphertext_commitment': header.get('ciphertext_commitment'),
                'owner_auth_tag': header.get('owner_auth_tag'),
                'payload_size_bytes': header.get('payload_size_bytes'),
            }
        self.metrics['communication_sizes'].append({
            'type': 'check',
            'size': self._safe_obj_size(check_req_payload, fallback=32)
            + (self._safe_obj_size(check_material, fallback=256) if check_material is not None else 0),
            'records': 0,
        })

        if not has_access:
            print("     Naive CCS-2023 授权检查失败")
            return None

        token_start = time.perf_counter()
        query_token = hashlib.sha256(
            header['encapsulations'][querier_id] + self.user_private_material[querier_id] + dataset_id.encode('utf-8')
        ).digest()
        recovered_envelope_key = self._unwrap_for_user(querier_id, header['encapsulations'][querier_id], dataset_id)
        wrapped_payload = self.wrapped_ciphertext_store[(owner_id, dataset_id)]
        if isinstance(wrapped_payload, dict) and wrapped_payload.get('format') == self._PACKAGE_FORMAT:
            payload, payload_commitment, wrapped_digest = self._decrypt_chunked_wrapped_payload(
                recovered_envelope_key,
                wrapped_payload,
            )
            verify_start = time.perf_counter()
            if wrapped_digest != header['wrapped_payload_digest']:
                raise ValueError('wrapped payload digest mismatch')
            if payload_commitment != header['ciphertext_commitment']:
                raise ValueError('ciphertext commitment mismatch')
            verify_elapsed = time.perf_counter() - verify_start
        else:
            serialized_payload = self._decrypt_payload(recovered_envelope_key, wrapped_payload)
            verify_start = time.perf_counter()
            self._verify_wrapped_payload(header, wrapped_payload, serialized_payload)
            verify_elapsed = time.perf_counter() - verify_start
            payload = self._deserialize_encrypted_package(serialized_payload)
        token_elapsed = time.perf_counter() - token_start

        retrieval_size = self._wrapped_payload_size(wrapped_payload)
        req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
            'query_token': query_token,
            'encrypted_model': model,
            'model_type': model.get('type', 'dot_product') if isinstance(model, dict) else 'dot_product',
        }
        req_size = self._safe_obj_size(req_payload)
        self.metrics['communication_sizes'].append({
            'type': 'query',
            'size': req_size,
            'records': len(payload['encrypted_records']),
        })

        start_query = time.perf_counter()
        encrypted_data = payload['encrypted_records']
        encrypted_model = self.encrypt_model(model)
        model_encrypt_time = time.perf_counter() - start_query
        self.metrics['encrypt_times'].append(model_encrypt_time)

        query_compute_start = time.perf_counter()
        results = []
        if isinstance(model, list):
            for enc_record in encrypted_data:
                try:
                    results.append(enc_record.dot(encrypted_model))
                except Exception:
                    results.append(self.he.encrypt([0.0]))
        elif isinstance(model, dict) and model.get('type') == 'decision_tree':
            nodes = model.get('nodes', [])
            node_map = {n.get('id'): n for n in nodes}
            root_id = model.get('root', 0)
            for enc_record in encrypted_data:
                try:
                    plain_record = self.he.decrypt(enc_record)
                    if not isinstance(plain_record, list):
                        plain_record = [float(plain_record)]
                    current_id = root_id
                    pred = 0.0
                    depth = 0
                    while depth < 10 and current_id in node_map:
                        node = node_map[current_id]
                        if 'value' in node:
                            pred = float(node['value'])
                            break
                        feature_idx = int(node.get('feature', 0))
                        threshold = float(node.get('threshold', 0.0))
                        feature_val = float(plain_record[feature_idx]) if feature_idx < len(plain_record) else 0.0
                        current_id = node.get('left') if feature_val <= threshold else node.get('right')
                        depth += 1
                    results.append(self.he.encrypt([pred]))
                except Exception:
                    results.append(self.he.encrypt([0.0]))
        else:
            for _ in encrypted_data:
                results.append(self.he.encrypt([0.0]))

        query_time = time.perf_counter() - query_compute_start + token_elapsed
        self.metrics['query_times'].append(query_time)

        response_size = retrieval_size + self._safe_obj_size(results, fallback=max(1, len(results)) * 1024)
        self.metrics['communication_sizes'].append({
            'type': 'decrypt',
            'size': response_size,
            'records': len(results),
        })

        start_decrypt = time.perf_counter()
        decrypted_results = []
        for enc_result in results:
            try:
                dec = self.he.decrypt(enc_result)
                decrypted_results.append(dec[0] if isinstance(dec, list) and dec else float(dec))
            except Exception:
                decrypted_results.append(0.0)
        decrypt_time = time.perf_counter() - start_decrypt
        self.metrics['decrypt_times'].append(decrypt_time)

        print(f"      Naive CCS-2023 封装恢复: {token_elapsed*1000:.2f} ms")
        print(f"      完整密文校验: {verify_elapsed*1000:.2f} ms")
        print(f"      模型加密: {model_encrypt_time*1000:.2f} ms")
        print(f"      查询计算: {(query_time-token_elapsed)*1000:.2f} ms")
        print(f"      结果解密: {decrypt_time*1000:.2f} ms")
        return decrypted_results

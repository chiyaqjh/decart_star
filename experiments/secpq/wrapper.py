# decart/experiments/secpq/wrapper.py
import hashlib
import os
import time

from experiments.scheme2_server.wrapper import ServerSchemeExperimentWrapper


class SecPQExperimentWrapper(ServerSchemeExperimentWrapper):
    def __init__(self, N: int = 64, n: int = 16):
        super().__init__(N=N, n=n)
        self.system_master_key = None
        self.public_params = {}
        self.user_credentials = {}
        self.public_user_directory = {}
        self.dataset_keys = {}
        self.dataset_access_tickets = {}

    def setup(self) -> float:
        start = time.perf_counter()

        self.system_master_key = os.urandom(32)
        index_salt = os.urandom(16)
        token_seed = os.urandom(32)

        self.public_params = {
            'scheme': 'SecPQ',
            'version': 1,
            'index_salt': index_salt,
            'token_commitment': hashlib.sha256(token_seed).digest(),
            'he_public_key_fingerprint': hashlib.sha256(str(self.server_public_key).encode('utf-8')).digest(),
        }

        elapsed = time.perf_counter() - start
        self.metrics['setup_time'] = elapsed
        print(f"   SecPQ setup completed: {elapsed:.4f}s")
        return elapsed

    def register_user(self, user_id: int):
        keygen_start = time.perf_counter()

        user_secret = os.urandom(32)
        query_seed = os.urandom(32)
        public_token = hashlib.sha256(query_seed).digest()

        keygen_elapsed = time.perf_counter() - keygen_start
        self.metrics['keygen_times'].append(keygen_elapsed)

        register_start = time.perf_counter()
        self.registered_users.add(user_id)
        self.user_credentials[user_id] = {
            'secret': user_secret,
            'query_seed': query_seed,
        }
        self.public_user_directory[user_id] = {
            'token': public_token,
            'registered_at': time.time(),
        }
        register_elapsed = time.perf_counter() - register_start
        self.metrics['register_times'].append(register_elapsed)

        return user_secret, public_token

    def _build_access_ticket(self, user_id: int, dataset_key: bytes, dataset_id: str) -> bytes:
        credential = self.user_credentials[user_id]
        material = credential['secret'] + dataset_key + dataset_id.encode('utf-8') + self.system_master_key
        return hashlib.sha256(material).digest()

    def get_auxiliary_sizes(self):
        crs_size = self._safe_obj_size(self.public_params, fallback=128)
        pp_size = self._safe_obj_size(self.public_user_directory, fallback=128)
        aux_blob = {
            'dataset_keys': self.dataset_keys,
            'dataset_access_tickets': self.dataset_access_tickets,
        }
        aux_size = self._safe_obj_size(aux_blob, fallback=128)
        return {
            'crs_size_bytes': crs_size,
            'pp_size_bytes': pp_size,
            'aux_size_bytes': aux_size,
            'total_auxiliary_size_bytes': crs_size + pp_size + aux_size,
        }

    def encrypt_dataset(self, owner_id, data, policy, metadata=None):
        start = time.perf_counter()
        c_m, sk_h_s, dataset_id = super().encrypt_dataset(owner_id, data, policy, metadata=metadata)

        dataset_key = os.urandom(32)
        self.dataset_keys[(owner_id, dataset_id)] = {
            'owner_id': owner_id,
            'dataset_key_commitment': hashlib.sha256(dataset_key).digest(),
            'policy': tuple(sorted(set(policy))),
        }

        ticket_map = {}
        for querier_id in sorted(set(policy)):
            if querier_id in self.user_credentials:
                ticket_map[querier_id] = self._build_access_ticket(querier_id, dataset_key, dataset_id)
        self.dataset_access_tickets[(owner_id, dataset_id)] = ticket_map

        extra_elapsed = time.perf_counter() - start
        if self.metrics['encrypt_times']:
            self.metrics['encrypt_times'][-1] += extra_elapsed

        upload_bundle_size = self._safe_obj_size({
            'dataset_meta': self.dataset_keys[(owner_id, dataset_id)],
            'tickets': ticket_map,
        }, fallback=max(1, len(ticket_map)) * 64)
        for comm in reversed(self.metrics['communication_sizes']):
            if isinstance(comm, dict) and comm.get('type') == 'encrypt':
                comm['size'] = comm.get('size', 0) + upload_bundle_size
                break

        return c_m, sk_h_s, dataset_id

    def execute_query(self, querier_id, owner_id, dataset_id, model, prepared_model=None):
        if not (isinstance(model, dict) and model.get('type') == 'decision_tree'):
            raise ValueError('SecPQ currently supports only decision_tree queries')

        check_start = time.perf_counter()
        ticket_map = self.dataset_access_tickets.get((owner_id, dataset_id), {})
        access_ticket = ticket_map.get(querier_id)
        self.metrics['check_times'].append(time.perf_counter() - check_start)
        check_req_payload = {
            'querier_id': querier_id,
            'owner_id': owner_id,
            'dataset_id': dataset_id,
        }
        self.metrics['communication_sizes'].append({
            'type': 'check',
            'size': self._safe_obj_size(check_req_payload, fallback=32)
            + (self._safe_obj_size(access_ticket, fallback=len(access_ticket)) if access_ticket is not None else 0),
            'records': 0,
        })

        if access_ticket is None:
            print("     SecPQ authorization check failed")
            return None

        token_start = time.perf_counter()
        token_material = (
            self.user_credentials[querier_id]['query_seed']
            + access_ticket
            + dataset_id.encode('utf-8')
        )
        query_token = hashlib.sha256(token_material).digest()
        token_elapsed = time.perf_counter() - token_start

        results = super().execute_query(querier_id, owner_id, dataset_id, model, prepared_model=prepared_model)

        if self.metrics['query_times']:
            self.metrics['query_times'][-1] += token_elapsed

        for comm in reversed(self.metrics['communication_sizes']):
            if isinstance(comm, dict) and comm.get('type') == 'query':
                comm['size'] = comm.get('size', 0) + len(query_token)
                break

        return results

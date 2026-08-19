import torch
import random
import lmdb
import pickle
from torch_geometric.data import Dataset
from tqdm import tqdm
from torch_geometric.data import Batch, Data
import io


class BackmapData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('aa_bond_index', 'aa_angle_index', 'aa_torsion_index'):
            # these index into pred[mask] / aa_names_order, not into the bead graph
            return int(self.mask.sum())
        return super().__inc__(key, value, *args, **kwargs)

class CUDAPrefetcher:
    """
    Overlaps H2D transfer of batch N+1 with GPU compute on batch N.
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            raw = next(self._iter)
            with torch.cuda.stream(self.stream):
                self._next = raw.to(self.device, non_blocking=True)
        except StopIteration:
            self._next = None

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)

class PreBatchedLMDBDataset_IO(Dataset):
    """
    Each item is already a PyG Batch — no collation needed at training time.
    """
    def __init__(self, lmdb_path, transform=None):
        super().__init__(transform=transform)
        self.lmdb_path = lmdb_path
        self._env = None

        tmp = lmdb.open(lmdb_path, readonly=True, lock=False)
        with tmp.begin() as txn:
            self._len     = int(txn.get(b"__len__").decode())
            self.node_dim = int(txn.get(b"node_dim").decode())
            self.edge_dim = int(txn.get(b"edge_dim").decode())
        tmp.close()

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def len(self):
        return self._len

    def get(self, idx):
        with self._get_env().begin() as txn:
            # Pickle load
            # return pickle.loads(txn.get(str(idx).encode("ascii")))
            # Torch load with IO deserialize
            raw = txn.get(str(idx).encode("ascii"))
        return deserialize(raw)

class PreBatchedLMDBDataset(Dataset):
    """
    Each item is already a PyG Batch — no collation needed at training time.
    """
    def __init__(self, lmdb_path, transform=None):
        super().__init__(transform=transform)
        self.lmdb_path = lmdb_path
        self._env = None

        tmp = lmdb.open(lmdb_path, readonly=True, lock=False)
        with tmp.begin() as txn:
            self._len     = int(txn.get(b"__len__").decode())
            self.node_dim = int(txn.get(b"node_dim").decode())
            if txn.get(b"edge_dim") is not None:
                self.edge_dim = int(txn.get(b"edge_dim").decode())
        tmp.close()

    def _get_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path, readonly=True, lock=False,
                readahead=False, meminit=False,
            )
        return self._env

    def len(self):
        return self._len

    def get(self, idx):
        with self._get_env().begin() as txn:
            # Pickle load
            return pickle.loads(txn.get(str(idx).encode("ascii")))

def serialize(batch):
    buf = io.BytesIO()
    torch.save(batch, buf)
    return buf.getvalue()

def deserialize(raw):
    return torch.load(io.BytesIO(raw), weights_only=False)

def create_prebatched_lmdb(data_list: list, lmdb_path: str,
                           batch_size: int = 1024,
                           map_size_gb: int = 1000):
    print(f"  loaded {len(data_list):,} graphs")
    env = lmdb.open(lmdb_path,
                    map_size=map_size_gb * 1024 ** 3,
                    subdir=True,
                    readonly=False,
                    meminit=False,
                    map_async=True)

    COMMIT_INTERVAL = 1000  # now this is 1000 *batches*, not graphs
    txn = env.begin(write=True)
    num_batches = 0

    for start in tqdm(range(0, len(data_list), batch_size), desc="Writing batches"):
        chunk = data_list[start: start + batch_size]
        batch = Batch.from_data_list(chunk)  # collation happens here, once
        # Pickle save
        txn.put(str(num_batches).encode("ascii"), pickle.dumps(batch))
        # Torch save with IO serialize
        # txn.put(str(num_batches).encode("ascii"), serialize(batch))
        num_batches += 1
        if num_batches % COMMIT_INTERVAL == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    node_dim = data_list[0].x.shape[1]
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(num_batches).encode())
        txn.put(b"node_dim", str(node_dim).encode())
        if data_list[0].edge_attr is not None:
            txn.put(b"edge_dim", str(data_list[0].edge_attr.shape[1]).encode())

    env.close()
    print(f"  saved {num_batches:,} batches ({len(data_list):,} graphs)")

def convert_pt_database_to_lmdb(data_path: str, batch_size: int = 1024, map_size_gb: int = 1000):
    print(f"Loading {f"{data_path}/train_data.pt"}...")
    create_prebatched_lmdb(torch.load(f"{data_path}/train_data.pt", weights_only=False),
                               f"{data_path}/train_prebatch.lmdb",
                           batch_size=batch_size,
                           map_size_gb=map_size_gb)
    print(f"Loading {f"{data_path}/test_data.pt"}...")
    create_prebatched_lmdb(torch.load(f"{data_path}/test_data.pt", weights_only=False),
                               f"{data_path}/test_prebatch.lmdb",
                           batch_size=batch_size,
                           map_size_gb=map_size_gb)
    print(f"Loading {f"{data_path}/val_data.pt"}...")
    create_prebatched_lmdb(torch.load(f"{data_path}/val_data.pt", weights_only=False),
                               f"{data_path}/val_prebatch.lmdb",
                           batch_size=batch_size,
                           map_size_gb=map_size_gb)

def batching_pt_framewise(data_paths, out_dir="data/batched_pt"):
    train_data = []
    val_data = []
    test_data = []
    for each in data_paths:
        print(f"Processing {each}")
        data_list = torch.load(each, weights_only=False)
        train_blocks, test_blocks, val_blocks = prepare_datalist(data_list)
        train_data.extend(train_blocks)
        val_data.extend(val_blocks)
        test_data.extend(test_blocks)
    print("Saving training data...")
    torch.save(train_data, f"{out_dir}/train_data.pt")
    print("Saving test and validation data...")
    torch.save(test_data, f"{out_dir}/test_data.pt")
    torch.save(val_data, f"{out_dir}/val_data.pt")

def extend_lmdb(lmdb_path: str, data_list: list, batch_size: int = 1024, map_size_gb: int = 1000):
    env = lmdb.open(lmdb_path,
                    map_size=map_size_gb * 1024 ** 3,
                    subdir=True,
                    readonly=False,
                    meminit=False,
                    map_async=True)

    # Find where to continue from
    with env.begin() as txn:
        current_batch_count = int(txn.get(b"__len__").decode())

    print(f"  existing batches: {current_batch_count:,}, appending {len(data_list):,} new graphs")

    COMMIT_INTERVAL = 1000  # now this is 1000 *batches*, not graphs
    txn = env.begin(write=True)
    new_batch_count = 0
    for i, start in enumerate(tqdm(range(0, len(data_list), batch_size), desc="Extending LMDB")):
        chunk = data_list[start: start + batch_size]
        batch = Batch.from_data_list(chunk)
        key = str(current_batch_count + new_batch_count).encode("ascii")
        txn.put(key, pickle.dumps(batch))
        new_batch_count += 1
        if new_batch_count % COMMIT_INTERVAL == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()

    # Update __len__
    new_total = current_batch_count + new_batch_count
    with env.begin(write=True) as txn:
        txn.put(b"__len__", str(new_total).encode())

    env.close()
    print(f"  new total: {new_total:,} batches")

def add_pt_to_lmdb_database(pt_path: list, lmdb_path: str, batch_size: int = 1024, map_size_gb: int = 1000):
    train_data, test_data, val_data = prepare_pt_datalists(pt_path)
    extend_lmdb(f"{lmdb_path}/train_prebatch.lmdb", train_data, batch_size=batch_size, map_size_gb=map_size_gb)
    extend_lmdb(f"{lmdb_path}/test_prebatch.lmdb", test_data, batch_size=batch_size, map_size_gb=map_size_gb)
    extend_lmdb(f"{lmdb_path}/val_prebatch.lmdb", val_data, batch_size=batch_size, map_size_gb=map_size_gb)

def prepare_pt_datalists(pt_paths):
    train_data = []
    val_data = []
    test_data = []
    random.seed(42)
    for each in pt_paths:
        print(f"Processing {each}")
        data_list = torch.load(each, weights_only=False)
        train_blocks, test_blocks, val_blocks = prepare_datalist(data_list)
        train_data.extend(train_blocks)
        val_data.extend(val_blocks)
        test_data.extend(test_blocks)
    return train_data, test_data, val_data

def prepare_datalist(data_list):
    frame = data_list[0].frame
    block_size = 0
    i = 1
    while frame == data_list[i].frame:
        block_size += 1
        i += 1
    n_blocks = len(data_list) // block_size
    blocks = [data_list[i * block_size:(i + 1) * block_size]
              for i in range(n_blocks)]
    random.shuffle(blocks)

    train_data = [d for block in blocks[:int(0.7 * n_blocks)] for d in block]
    test_data = [d for block in blocks[int(0.8 * n_blocks):] for d in block]
    val_data = [d for block in blocks[int(0.7 * n_blocks):int(0.8 * n_blocks)] for d in block]

    return train_data, test_data, val_data

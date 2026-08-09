import os
from itertools import permutations
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

class ArgoverseV1Dataset:

    def __init__(self, root: str, raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None) -> None:
        self.root = root
        self._raw_dir = raw_dir or os.path.join(root, 'data_original')
        self._processed_dir = processed_dir or os.path.join(root, 'data_processed')
        os.makedirs(self._processed_dir, exist_ok=True)
        if not os.path.isdir(self._raw_dir):
            raise FileNotFoundError(
                f"raw data dir not found: {self._raw_dir} "
                f"(download Argoverse 1 and pass --data-dir)"
            )
        self._raw_file_names = sorted([f for f in os.listdir(self._raw_dir) if f.endswith('.csv')])
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self._raw_file_names]
        self._processed_paths = [os.path.join(self._processed_dir, f) for f in self._processed_file_names]

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths
    @property
    def raw_paths(self) -> List[str]:
        return [os.path.join(self._raw_dir, f) for f in self._raw_file_names]

    def process(self) -> None:
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(raw_path)
            # Save under the raw file's stem so the name always matches
            # self._processed_paths (raw names may carry leading zeros that an
            # int round-trip would drop)
            out_name = os.path.splitext(os.path.basename(raw_path))[0] + '.pt'
            torch.save(kwargs, os.path.join(self.processed_dir, out_name))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx):
        return torch.load(self.processed_paths[idx], weights_only=False)


def process_argoverse(raw_path: str,) -> Dict:
    df = pd.read_csv(raw_path)

    # filter out actors that are unseen during the historical time steps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    # padding_mask: True = invalid/padding (initialized True, valid frames set
    # to False). NOTE: models/loss_common and models/metrics_common use the
    # OPPOSITE convention (True = valid) — flip before passing across modules.
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()

    y = x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'seq_id': int(seq_id) if str(seq_id).isdigit() else seq_id,
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }

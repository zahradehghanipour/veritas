"""
RL (Reinforcement Learning) Branching Heuristic for αβ-CROWN Verifier

TODO
    - Why and how often do we skip an action of a node that is already masked?
    - Monitor rewards (as a post-processing step). Is taking the difference too noisy? If so, we can normalise it (e.g., clip it).
    - We mask Q values, but then use them as logits in cross-entropy loss.
    - Explore other methods for reward normalisation: e.g., reward = reward / (abs(reward).mean() + epsilon).


NOTE
    - See use of `check_worst_domain`.
    - Do we want to limit our buffer size?
    - Actions are stored as a list of the form [[layer id, node id]].
    - Do we want to keep track of 'depths' explicitly?
    - Do we want to keep track of rhs (which seems to be encoded in domains['thresholds'])?
    - Why enable the ab-crown 'attack' mode?
    - We may consider adding more input features. But if we do, note that the number of input features is hard-coded to 4 at the moment.
    - Storing & restoring state:

        At Episode 1:
            - Try to restore buffer from `params.buffer`, otherwise create a new buffer.
            - Try to restore models from `params.policynet` and `params.targetnet`, otherwise create new model.
            ...
            - Store buffer to `params.experiment + buffer.pkl`.
            - Store models to `params.experiment + {"policy" or "target"} + checkpoint.pt`.

"""

import inspect
from datetime import datetime

from heuristics.base import *
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import sys

import numpy as np
import random
from collections import deque, OrderedDict, namedtuple, Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from enum import Enum

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv

import copy

import pickle
from pathlib import Path

import csv
import os
from datetime import datetime

from .fsb import FsbBranching
from .kfsb import KfsbBranching



class LogLevel(Enum):
    DEBUG = 1
    INFO = 2
    WARN = 3

_LOGLEVELCOLORS = {
    LogLevel.DEBUG: "\033[94m",  # Blue
    LogLevel.INFO: "\033[92m",  # Green
    LogLevel.WARN: "\033[93m",  # Yellow
}

_RESETCOLOR = "\033[0m"

_DEBUGENABLED = True

_LOGFILE = None

_LOGINCOLOR = False


def _setlog(file=None, debug=False, color=False):
    global _LOGFILE
    global _DEBUGENABLED
    global _LOGINCOLOR
    _LOGINCOLOR = color
    _DEBUGENABLED = debug
    # NOTE We will append to a file that already exists!
    _LOGFILE = file


def myprint(level, timestamp, caller, *args):
    # Get color
    color = _LOGLEVELCOLORS.get(level, _RESETCOLOR)
    # Construct message
    msg = f"[ZRL][{timestamp}][{level.name:5s}][{caller}] {' '.join(str(arg) for arg in args)}"
    if _LOGINCOLOR:
        print(f"{color}{msg}{_RESETCOLOR}")
    else:
        # Color is disabled
        print(msg)
    if _LOGFILE:
        # Log to file
        with open(_LOGFILE, "a") as f:
            f.write(f"{msg}\n")


def debug(*msg):
    if not _DEBUGENABLED:
        return
    # timestamp = datetime.now().strftime("%H:%M:%S")
    # ZD, avoid collision
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    fn = inspect.stack()[1].function
    myprint(LogLevel.DEBUG, timestamp, fn, *msg)


def warn(*msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    fn = inspect.stack()[1].function
    myprint(LogLevel.WARN, timestamp, fn, *msg)


def info(*msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    fn = inspect.stack()[1].function
    myprint(LogLevel.INFO, timestamp, fn, *msg)


StatePair = namedtuple("StatePair", ["s0", "s1"])
EMetadata = namedtuple("EMetadata", ["who", "eid"])


class DecisionMaker(Enum):
    RND = 0
    EXP = 1
    DQN = 2


class ReplayBuffer:
    def __init__(self):
        """
        Stores and retrieves past experiences for training.
        """
        self.buffer = []
        self.episode = []  # The current episode
        self.nepisodes = 0
        self.nsafeepisodes = 0
        self.safeepisodes = []
        self.nsteps = 0
        self.whodoneit = defaultdict(list)

    def _whodonewhat(self):
        offset = self.size
        actors = []  # List of actors for current episode
        for idx, (meta, *_rest) in enumerate(self.episode):
            self.whodoneit[meta.who].append(offset + idx)
            actors.append(meta.who)
        return Counter(actors)

    def _normalize(self, episode, epsilon=1e-8, method="sum"):
        """
        Normalizes rewards in a single episode.
        """
        debug(f"Normalise rewards with method '{method}'")

        rewards = [r for *_rest, r, _ in episode]

        if method == "sum":

            total = sum(rewards) + epsilon
            normalized = [r / total for r in rewards]

        elif method == "minmax":

            rmin = min(rewards)
            # `r = state.s1.glb - state.s0.glb`, and `glb` is less or equal to `0`; so `rmax = 0`
            rmax = 0
            scale = (rmax - rmin) + epsilon
            normalized = [(r - rmin) / scale for r in rewards]

        elif method == "zscore":
            # Compute mean and std. deviation
            ravg = sum(rewards) / len(rewards)
            rdev = (
                sum((r - ravg) ** 2 for r in rewards) / len(rewards)
            ) ** 0.5 + epsilon
            normalized = [(r - ravg) / rdev for r in rewards]

        else:
            raise ValueError(f"Unknown normalization method '{method}'")

        return [
            (metadata, state, action, normal, terminal, depth)
            for (normal, (metadata, state, action, _, terminal, depth)) in zip(
                normalized, episode
            )
        ]

    def _computereward(self, s0, s1):
        """
        A simple reward function for now, based on global lower bound(s)
        """
        # Safety check

        glb0 = s0["glb"].view(-1) # curr glb
        glb1 = s1["glb"].view(-1) # next glb

        if glb0.shape != glb1.shape:
            raise ValueError(
                f"Global lower bounds do not match ({glb0.shape} vs {glb1.shape})"
            )

        return (glb1 - glb0).min().item()

    def step(self, s0, s1, action, who, depth):
        # debug(f"Storing step #{(self.nsteps + 1):03d}")
        self.episode.append(
            (
                EMetadata(who, self.nepisodes),
                StatePair(s0, s1),
                action,
                self._computereward(s0, s1),
                0,
                depth
            )
        )
        self.nsteps += 1

    def episodeComplete(self, safe, bonus=10, penalty=-1, norm=None):

        # Index episode so we know who done which action
        # in the current episode.
        counts = self._whodonewhat()

        info(
            f"[ZRL] Episode #{(self.nepisodes + 1):03d} completed ({safe}) in {len(self.episode)} steps\n"
            f"[ZRL]\tEXP: {counts[DecisionMaker.EXP]:02d}\n"
            f"[ZRL]\tDQN: {counts[DecisionMaker.DQN]:02d}\n"
            f"[ZRL]\tRND: {counts[DecisionMaker.RND]:02d}"
        )

        # Update last entry in episode to indicate terminal state
        metadata, state, action, reward, _, depth = self.episode[-1]
        reward += bonus if safe else penalty
        self.episode[-1] = (metadata, state, action, reward, 1, depth)  # 1 = terminal

        if norm is not None:
            self.episode = self._normalize(self.episode, method=norm)

        self.buffer.extend(self.episode)

        if safe:
            # Keep track of the safe episode ids
            self.safeepisodes.append(self.nepisodes)

        # Clear episode and inc. counters
        self.episode = []
        self.nepisodes += 1
        return

    def sample(self, batchsize=None, onlysuccessful=False, who: DecisionMaker = None):

        assert self.size > 0, "Empty buffer"

        if who is not None:
            sampleids = self.whodoneit.get(who, [])
            if not sampleids:
                raise ValueError(f"No actions by decision-maker {who}")
        else:
            sampleids = list(range(self.size))

        if onlysuccessful:
            # We want sample ids that are part of successful episodes
            sampleids = [
                idx
                for idx in sampleids
                if self.buffer[idx][0].eid in self.safeepisodes  # metadata.eid
            ]
            if not sampleids:
                raise ValueError(f"No samples from successful episodes left")

        if batchsize is None or batchsize >= len(sampleids):
            # Returns all entries
            selectedids = sampleids

            if batchsize is not None and batchsize > len(sampleids):
                warn(f"Only {len(selectedids)}/{batchsize} returned in batch")
        else:
            selectedids = random.sample(sampleids, batchsize)

        # Shuffle the data in-place
        #
        # NOTE
        #
        # Although we sample above, sometimes (e.g., when we immitate, or when batchsize is large)
        # we may get the entire contents of the buffer. In such a case, we need to shuffle.
        #
        random.shuffle(selectedids)

        samples = [self.buffer[idx] for idx in selectedids]
        metadata, state, action, reward, terminal, depth = zip(*samples)

        if _DEBUGENABLED:
            # Dump statistics about batch
            stats = {
                "who": Counter(m.who for m in metadata),
                "eid": Counter(m.eid for m in metadata),
            }
            s = "\n"
            s += "[ZRL]Training batch:\n"
            s += f"[ZRL]\t{len(samples)} samples from {len(stats['eid'])} episodes:\n"
            for who in DecisionMaker:
                s += f"[ZRL]\t\t{who.name}: {stats['who'][who]:4d}\n"
            debug(f"{s}")

        return state, action, reward, terminal, depth

    def step_sample(self, batchsize=None, onlysuccessful=False, who: DecisionMaker = None, n_steps=None):
        """
        Sample a batch of individual steps from the buffer.
        Each step is a tuple: (metadata, state, action, reward, terminal, depth)
        """
        assert len(self.buffer) > 0, "Empty buffer"

        # Gather all indices
        if who is not None:
            sampleids = self.whodoneit.get(who, [])
            sampleids = [i for i in sampleids if i < len(self.buffer)]
            if not sampleids:
                raise ValueError(f"No valid steps from decision-maker {who}")
        else:
            sampleids = list(range(len(self.buffer)))

        if onlysuccessful:
            # We want sample ids that are part of successful episodes
            if n_steps:
                sampleids = [
                    idx
                    for idx in sampleids
                    if self.buffer[idx][0].eid in self.safeepisodes  # metadata.eid
                ]
            else:
                sampleids = [
                    idx
                    for idx in sampleids
                    if self.buffer[idx][0].eid in self.safeepisodes  # metadata.eid
                ]

        if batchsize is None or batchsize >= len(sampleids):
            selectedids = sampleids
            if batchsize is not None and batchsize > len(sampleids):
                warn(f"[ZRL] Only {len(selectedids)}/{batchsize} returned in batch")
        else:
            selectedids = random.sample(sampleids, batchsize)

        random.shuffle(selectedids)

        # Extract samples
        samples = [self.buffer[idx] for idx in selectedids]

        # Unpack each sample tuple
        metadata, state, action, reward, terminal, depth = zip(*samples)

        if _DEBUGENABLED:
            # Summarize batch stats
            stats = {
                "who": Counter(m.who for m in metadata),
            }
            s = "\n"
            s += "Training batch:\n"
            s += f"\t{len(samples)} samples\n"
            for who in DecisionMaker:
                s += f"\t\t{who.name}: {stats['who'][who]:4d}\n"
            debug(s)

        return state, action, reward, terminal, depth

    def __len__(self):
        return len(self.buffer)

    @property
    def size(self):
        return len(self.buffer)

    def clear(self):
        # Clear episode
        self.episode = []
        # Clear buffer
        self.buffer = []

    def info(self):
        """
        Prints out statistics.
        """
        s = "\n"
        s += "[ZRL]Replay buffer:\n"

        s += f"[ZRL]\t{self.size} steps:\n"
        for who in DecisionMaker:
            lst = self.whodoneit.get(who, [])
            s += f"[ZRL]\t\t{who.name}: {len(lst):4d} steps\n"

        s += f"[ZRL]\t{len(self.safeepisodes)}/{self.nepisodes} safe episodes\n"

        info(f"{s}")

    def store(self, path="buffer.pkl", prefix=None):

        # Is there an episode in progress?!
        if len(self.episode) > 0:
            raise Exception("Cannot store state while an episode is in progress")

        if prefix:
            directory, filename = os.path.split(path)
            filename = f"{prefix}-{filename}"
            path = os.path.join(directory, filename)

        with open(path, "wb") as fstream:
            pickle.dump(self, fstream)

    @classmethod
    def restore(cls, path="buffer.pkl"):
        if not os.path.exists(path):
            warn(f"File {path} not found. Starting with empty buffer.")
            return cls()

        info(f"[ZRL] Restoring state from {path}")

        with open(path, "rb") as f:
            buffer = pickle.load(f)
            
        if not isinstance(buffer, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")

        buffer.info()

        return buffer

    def split_by_episode(self, test_size: float = 0.2, seed: int = 42, stratify_safe: bool = False):
        """
        Split the replay buffer by episodes (instances), not by steps.
        Returns (train_buf, test_buf), both ReplayBuffer objects.
        - test_size: fraction of episodes to put in test
        - seed: for deterministic shuffling
        - stratify_safe: try to preserve the proportion of safe episodes
        """
        from collections import defaultdict

        assert 0.0 < test_size < 1.0, "test_size must be in (0,1)"

        # --- 1) collect unique episode ids present in the buffer ---
        episode_ids = sorted({meta.eid for (meta, *_rest) in self.buffer})
        if not episode_ids:
            raise ValueError("No episodes found in buffer")

        safe_set = set(self.safeepisodes)
        if stratify_safe:
            safe_eids  = [e for e in episode_ids if e in safe_set]
            other_eids = [e for e in episode_ids if e not in safe_set]
            rnd = random.Random(seed)
            rnd.shuffle(safe_eids)
            rnd.shuffle(other_eids)

            n_test_safe  = int(round(len(safe_eids)  * test_size))
            n_test_other = int(round(len(other_eids) * test_size))
            test_eids  = set(safe_eids[:n_test_safe] + other_eids[:n_test_other])
            train_eids = set(e for e in episode_ids if e not in test_eids)
        else:
            rnd = random.Random(seed)
            eids = episode_ids[:]
            rnd.shuffle(eids)
            n_test = int(round(len(eids) * test_size))
            test_eids  = set(eids[:n_test])
            train_eids = set(eids[n_test:])

        # --- helper to build a new buffer from selected eids ---
        def build_subset(selected_eids: set):
            new_buf = ReplayBuffer()
            new_buf.buffer = []
            new_buf.whodoneit = defaultdict(list)

            # keep only entries whose metadata.eid is in the selected set
            for idx, (meta, state, action, reward, terminal, depth) in enumerate(self.buffer):
                if meta.eid in selected_eids:
                    new_idx = len(new_buf.buffer)
                    new_buf.buffer.append((meta, state, action, reward, terminal, depth))
                    new_buf.whodoneit[meta.who].append(new_idx)

            # recompute counters and episode lists from the subset
            subset_eids = sorted({meta.eid for (meta, *_r) in new_buf.buffer})
            new_buf.nepisodes = len(subset_eids)
            new_buf.nsteps = len(new_buf.buffer)
            new_buf.safeepisodes = [e for e in self.safeepisodes if e in selected_eids]
            # (nsafeepisodes is unused elsewhere; no need to set it)
            # episode is empty as we are not in the middle of an episode
            new_buf.episode = []
            return new_buf

        train_buf = build_subset(train_eids)
        test_buf  = build_subset(test_eids)

        # Optional: sanity checks (no leakage; disjoint; coverage)
        assert not (train_eids & test_eids), "Train/Test episode id sets overlap!"
        assert train_buf.nsteps + test_buf.nsteps <= self.nsteps, "Unexpected step count mismatch"

        return train_buf, test_buf


class GATGraphEncoder(nn.Module):
    def __init__(self, dinp, dhid, dout, heads=4):
        super(GATGraphEncoder, self).__init__()
        # NOTE self loops are add manually, hence set to false.
        self.head = GATConv(dinp, dhid, heads=heads, concat=True, add_self_loops=False)
        self.tail = GATConv(
            dhid * heads, dout, heads=1, concat=False, add_self_loops=False
        )

    def forward(self, x, a):
        # ZD, Making RL compatible with GPU
        device = next(self.head.parameters()).device
        x = x.to(device)
        a = a.to(device)

        x = F.elu(self.head(x, a))
        x = F.elu(self.tail(x, a))
        return x


class GCNGraphEncoder(nn.Module):
    def __init__(self, dinp, dhid, dout):
        super(GCNGraphEncoder, self).__init__()
        # NOTE self loops are add manually, hence set to false.
        self.head = GCNConv(dinp, dhid, add_self_loops=False)
        self.tail = GCNConv(dhid, dout, add_self_loops=False)

    def forward(self, x, a):
        # ZD, Making RL compatible with GPU
        device = next(self.head.parameters()).device
        x = x.to(device)
        a = a.to(device)

        x = F.relu(self.head(x, a))
        x = F.relu(self.tail(x, a))
        return x


class PolicyNet(nn.Module):

    # def __init__(self, din, dout, glb, dpth, gH=64, tH=128, nheads=4, nlayers=2, GAT=False):
    def __init__(self, din, dout, glb, gH=64, tH=128, nheads=4, nlayers=2, GAT=False):

        super(PolicyNet, self).__init__()
        
        # NOTE [MAX] `dout`` is not used. which makes sense to me as we want the outputs of the GNN to be `gH`, and the transformer network should be 1.

        # Store parameters for cloning
        self._config = {
            "din": din,
            "dout": dout, # FIXME not used anywhere, consider removing?
            "glb": glb,
            # "dpth" : dpth,
            "gH": gH,
            "tH": tH,
            "nheads": nheads,
            "nlayers": nlayers,
            "GAT": GAT,
        }

        self.graphencoder = (
            GATGraphEncoder(din, gH, gH) if GAT else GCNGraphEncoder(din, gH, gH)
        )

        # ZD, adding the concatenate dimention:
        encoder = nn.TransformerEncoderLayer(
            d_model=gH + din, nhead=nheads, dim_feedforward=tH
        )
        self.transformer = nn.TransformerEncoder(encoder, num_layers=nlayers)

        # ZD, adding CLS tokenes
        self.glb_proj = nn.Linear(1, gH + din)  # Project scalar → same dim as transformer input
        # self.depth_proj = nn.Linear(1, gH + din)

        self._glb = glb
        # self._dpth = dpth

        self.out = nn.Linear(gH + din, 1)

    # def forward(self, x, adj, glb, depth):
    def forward(self, x, adj, glb):
        # debug("[ZRL] policynet (GNN) input shape:", x.shape)
        # gnn_input = x                            # [N, 3]
        # x = self.graphencoder(x, adj)            # [N, gH]

        # ZD, Making RL compatible with GPU
        device = next(self.parameters()).device
        x = x.to(device)
        adj = adj.to(device)

        gnn_input = x                            # [N, 3]
        x = self.graphencoder(x, adj)            # [N, gH]

        # debug("[ZRL] GNN output shape:", x.shape)

        x = torch.cat([x, gnn_input], dim=-1)  # Concatenate along feature dimension → [N, gH + 3]
        # debug("[ZRL] Transformer input shape:", x.shape)

        # ZD, I added this as I had a porblem with size(), glb, depth: tensor([-1.79257298]), [10]
        # Normalize glb
        glb = torch.as_tensor(glb.min(), dtype=torch.float32, device=x.device).reshape(1)
        assert glb.shape == (1,), f"`glb` shape was {glb.shape}"
        glb_embed = self.glb_proj(glb.unsqueeze(0)).unsqueeze(0)  # [1, 1, dim]

        # NOTE ZD, we are removing depth from the tokens as it might mislead the model
        # # Normalize depth
        # depth = torch.as_tensor(depth, dtype=torch.float32, device=x.device).reshape(1)
        # assert depth.shape == (1,), f"`depth` shape was {depth.shape}"
        # depth_embed = self.depth_proj(depth.unsqueeze(0)).unsqueeze(0)  # [1, 1, dim]


        # Convert and project global lower bound
        if not isinstance(glb, torch.Tensor):
            glb = torch.tensor(glb, dtype=torch.float32, device=x.device)
        # NOTE there could be more glbs, i.e. each output with a negative lower bound, but we expect only the worst case, or for a specific output.
        assert glb.size() == (1,), f"`glb` has incorrect size: {glb.size()}"
        glb_embed = self.glb_proj(glb.unsqueeze(0)).unsqueeze(0)  # [1, 1, dim]
        # debug("[ZRL] glb_embed shape:", glb_embed.shape)

        # Convert and project depth
        # # NOTE [MAX] we might need to think about some waye to encode the depth. One hot, but what is the max?
        # if not isinstance(depth, torch.Tensor):
        #     depth = torch.tensor(depth, dtype=torch.float32, device=x.device)
        # assert depth.size() == (1,), f"`depth` has incorrect size {depth.size()}"
        # depth_embed = self.depth_proj(depth.unsqueeze(0)).unsqueeze(0)  # [1, 1, dim]
        # # debug("[ZRL] depth_embed shape:", depth_embed.shape)

        # Stack both CLS tokens at beginning of sequence
        # cls_tokens = torch.cat([glb_embed, depth_embed], dim=0)  # [2, 1, dim]
        cls_tokens = torch.cat([glb_embed], dim=0)  # [1, 1, dim]
        x = torch.cat([cls_tokens, x.unsqueeze(1)], dim=0)       # [N+2, 1, dim]
        # debug("[ZRL] Transformer input shape:", x.shape)

        x = self.transformer(x).squeeze(1)  # [N+2, gH + din]
        # debug("[ZRL] Transformer output shape:", x.shape)

        # logits = self.out(x[2:]).squeeze(-1)  # Skip CLSs [N]
        logits = self.out(x[1:]).squeeze(-1)  # Skip CLSs [N]
        # debug("[ZRL] Final layer output shape:", logits.shape)

        return logits

    def clone(self, eval=True):
        # ZD, Making RL compatible with GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = PolicyNet(**self._config).to(device)
        # net = PolicyNet(**self._config)

        # Copy parameters
        net.load_state_dict(self.state_dict())
        # Move to device
        net = net.to(next(self.parameters()).device)
        # Set in evaluation mode
        if eval:
            net.eval()
        return net

    def mirror(self, other):
        assert isinstance(other, PolicyNet)
        self.load_state_dict(other.state_dict())
        # Move to device
        self.to(next(other.parameters()).device)

    def store(self, opt=None, step=None, path="checkpoint.pt", prefix=None):

        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        parts = []
        if prefix:
            parts.append(prefix)
        parts.append(name)
        # NOTE Don't store step to make training loop easier across instances
        # parts.append(f"{step:04d}")

        filename = "-".join(parts) + ext
        path = os.path.join(directory, filename)

        checkpoint = {"theModel": self.state_dict()}

        if opt:
            checkpoint["theOptimizer"] = opt.state_dict()

        if step:
            checkpoint["step"] = step

        torch.save(checkpoint, path)

    def restore(self, path, opt=None):
        if not os.path.exists(path):
            warn(f"File {path} not found. Starting with a new net.")
            return 0

        checkpoint = torch.load(path)
        # Restore all learnable parameters
        self.load_state_dict(checkpoint["theModel"])
        # Restore optimizer state
        if opt:
            opt.load_state_dict(checkpoint["theOptimizer"])
        # Return training step
        return checkpoint.get("step", 0)


class ExecutionMode(Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass
class TreeNodeState:
    state: Any  # Extracted state (e.g., feature dict)
    action: Any  # The branching action taken
    who: Any  # DecisionMaker (e.g., DQN, RND, EXP)
    depth: int  # Depth in the tree
    solved: int = 0  # Number of children already solved


class BnBTree:

    def __init__(self):
        # uid → state
        self.domains: Dict[int, Optional[TreeNodeState]] = {}
        # uid → child domains
        self.children: Dict[int, List[int]] = defaultdict(list)

    def isempty(self) -> bool:
        return len(self.domains) == 0

    def get(self, uid: int):
        """
        Returns the state of domain `uid`.
        """
        assert uid is not None
        assert uid in self.domains, f"Domain {uid} not found"
        assert self.domains[uid] is not None, f"Domain {uid} is not set"
        node = self.domains[uid]
        return node.state, node.action, node.who, node.depth, node.solved

    def set(
        self,
        uid: int,
        pid: Optional[int],
        depth: int,
        state: Any,
        action: Any,
        who: Any,
    ):
        """
        Sets the state for domain `uid`.
        """
        if pid is None:
            # If this is the root, tree must be empty, and uid should not exist
            assert self.isempty(), "Root can only be set on an empty tree"
        else:
            # Is `uid` a child of `pid`?
            assert uid in self.children[pid], f"Domain {uid} is not a child of {pid}"
            # The state of `uid` should be set once, now!
            assert self.domains[uid] is None, f"Domain {uid} lready set"

            parent = self.domains[pid]

            # The state of the parent domain must have been set already
            assert (
                parent is not None
            ), f"Domain {pid} must be initialized before its children"

            # Check the depth of the domain we are adding
            assert (
                depth == parent.depth + 1
            ), f"Expected depth {parent.depth + 1}, got {depth}"

        self.domains[uid] = TreeNodeState(
            state=state, action=action, who=who, depth=depth
        )

    def _update(self, uids: List[int], pids: List[int]):
        """
        Adds a batch of new nodes to the tree.
        """
        for i, (uid, pid) in enumerate(zip(uids, pids)):
            if uid not in self.domains:
                self.domains[uid] = None
                if pid is not None:
                    self.children[pid].append(uid)

    def updateTree(self, uids: List[int], pids: List[int], uid: int, solved: int):
        self._update(uids, pids)
        assert uid in self.domains and self.domains[uid] is not None
        self.domains[uid].solved += solved

    def dump(self, root: Optional[int] = None, indent: int = 0):

        if root is None:
            roots = [
                uid
                for uid in self.domains
                if all(uid not in v for v in self.children.values())
            ]
            assert len(roots) == 1, f"Expected one root, found {roots}"
            root = roots[0]

        node = self.domains.get(root)
        prefix = "  " * indent + "→"
        if node:
            info(f"{prefix} Dom {root:03d} [action {node.action}]")
            # Add solved children display
            if node.solved > 0:
                for _ in range(node.solved):
                    prefix = " " * (indent + 1) + "→"
                    info(f"{prefix} Dom UKN [solved]")
        else:
            info(f"{prefix} Dom {root} [uninitialized]")

        for child in self.children.get(root, []):
            self.dump(child, indent + 1)


def _netinfo(net):
    """
    Extracts layer information from `net` object.

    This is a code snippet from beta_CROWN_solver.py:

    ```
    if verbose:
        print('Split layers:')
        for layer in self.net.split_nodes:
            print(f'  {layer}: {self.split_activations[layer.name]}')
        print('Nonlinear functions:')
        for node in self.net.nodes():
            if node.perturbed and len(node.requires_input_bounds):
                print('  ', node)
    ```
    """
    names = []
    sizes = []
    for layer in net.split_nodes:
        names.append(layer.name)
        # Get shape
        shape = getattr(layer, "output_shape", None)
        if not shape:
            raise ValueError
        # debug(f"{layer.name}\t→ {shape}")
        if isinstance(shape, list):
            shape = shape[0]
        size = torch.prod(torch.tensor(shape[1:])).item()
        sizes.append(size)

    # Parse non-linear layers
    #
    # NOTE:
    #
    # This is very restrictive code, making strong assumptions
    # about alphas so that we can process them safely.
    #
    relus = {key: None for key in names}
    for relu in net.net.relus:
        # debug(f"{relu}")
        # Is the ReLU perturbed?
        if not relu.perturbed:
            continue
        # Get the input layer
        inp = relu.inputs[0]
        target = inp.name
        if target not in relus:
            raise ValueError(f"Unknown layer name: {target}")
        # Is the input a known layer (see above)?
        relus[target] = relu.name

    # Is there a layer without a corresponding ReLU?
    missing = [k for k, v in relus.items() if v is None]
    if missing:
        raise RuntimeError(f"Missing ReLU for splitable layer(s) {missing}")

    return names, sizes, relus


class RLNeuronBranching(NeuronBranchingHeuristic):
    """
    Reinforcement Learning-based Neuron Branching with static edges.
    """

    def __init__(self, net, args):
        """
        Initializes the RL-based neuron branching heuristic for αβ-CROWN.
        """

        super().__init__(net)

        # Configure logging infrastructure
        _setlog(**args["logging"])

        self._experiment = args["experiment"]
        info("Running experiment", self._experiment)

        # Get layer order from `net`; and number of nodes per layer

        # ZD, for the imitaion learning script
        if net is not None:
            self._layers, self._nnodes, self._relus = _netinfo(net)
        else:
            self._layers, self._nnodes, self._relus = ['dummy_layer'], [1], [[0]]
        # debug(self._layers)
        # debug(self._nnodes)
        # debug(self._relus)

        __MAXLEN = max(len(str(layer)) for layer in self._layers)
        # for layer, count in zip(self._layers, self._nnodes):
        #     info(f"{str(layer):<{__MAXLEN}} ({count}) →")

        # The device
        if net is not None:
            self._device = net.x.device
        else:
            self._device = 'cpu'

        # Statistics re: actions
        self._nrandom_actions = 0  # Random actions
        self._nexpert_actions = 0  # Expert actions
        self._nmodels_actions = 0

        # Is the system in training or test mode?
        self._mode = ExecutionMode.TEST if args["eval"] else ExecutionMode.TRAIN

        # Q-learning training configuration

        self._gamma = args["training"]["gamma"]
        self._lr = args["training"]["lr"]
        self._batchSize = args["training"]["batchSize"]
        # How many batches should we use during training?
        self._trainingLoop = args["training"]["loop"]
        # How often should we train our policy network?
        self._updatePolicyFrequency = args["training"]["updatePolicyFrequency"]
        # How often should we update our "target" (mirrored policy)?
        self._updateTargetFrequency = args["training"]["updateTargetFrequency"]

        # What method should we use for reward shaping when an episode completes?
        self._rewardconfig = {
            "norm": args["training"]["reward"]["norm"],
            "penalty": args["training"]["reward"]["penalty"],
            "bonus": args["training"]["reward"]["bonus"],
        }

        # Create the policy network (given user specification)
        __netconfig = {
            "gH": args["pnet"]["gH"],
            "tH": args["pnet"]["tH"],
            "nheads": args["pnet"]["nheads"],
            "nlayers": args["pnet"]["nlayers"],
            "GAT": args["pnet"]["GAT"],
        }

        # ZD, get the paths
        self._targetPath = args["training"]["targetnet"]
        self._policyPath = args["training"]["policynet"]
        self._bufferPath = args["training"]["buffer"]
        self._epsilonPath = args["egreedy"]["ePath"]
        self._preparedBufferPath = args["training"]['preparedBuffer']
        
        # ZD, what is the training mode?
        self._training_mode = args["training"]["trainingMode"]

        # ZD, Restore epsilon from file if it exists, for consistent exploration across runs
        if os.path.exists(self._epsilonPath):
            try:
                self._epsilon = torch.load(self._epsilonPath)["epsilon"]
                info(f"Restored epsilon from {self._epsilonPath}: {self._epsilon}")
            except Exception as e:
                warn(f"Could not restore epsilon from {self._epsilonPath}: {e}")
        else:
            self._epsilon = args["egreedy"]["epsilon"]

        # Our e-greedy configuration
        self._isgreedy = args["egreedy"]["enabled"]
        self._epsilonDecay = args["egreedy"]["epsilonDecay"]
        self._minEpsilon = args["egreedy"]["minEpsilon"]

        # Should we update edges dynamically?
        self._useDynamicGraph = args["pnet"]["useDynamicGraph"]

        # ZD, do we need bidirectional edges?
        self._bidirectional = args["pnet"]["bidirectional"]

        # ZD, passing the glb    
        self._glb = 0
        # self._dpth = 0
        
        # self._policynet = PolicyNet(4, sum(self._nnodes), self._glb, self._dpth, **__netconfig)
        self._policynet = PolicyNet(4, sum(self._nnodes), self._glb, **__netconfig)

        # ZD, Making RL compatible with GPU
        self._policynet = self._policynet.to(self._device)

        if self._mode == ExecutionMode.TRAIN:
            # Try restore replay buffer
            self._buffer = ReplayBuffer.restore(path=self._bufferPath)

            # Initialize optimizer
            self._optimizer = torch.optim.Adam(
                self._policynet.parameters(), lr=self._lr
            )

            # Try restore policy net, optimizer, and steps
            self._trainingSteps = self._policynet.restore(
                opt=self._optimizer, path=self._policyPath
            )

            self._targetnet = self._policynet.clone(eval=True)

            # ZD, Making RL compatible with GPU
            self._targetnet = self._targetnet.to(self._device)
            # Try restore target net
            _ = self._targetnet.restore(path=self._targetPath)

        else:
            # No need to restore the buffer
            # TODO: Do we need buffer in eval mode? we do but another one just for logging
            self._buffer = ReplayBuffer()

            # Try restore policy net
            self._trainingSteps = self._policynet.restore(
                opt=None, path=self._policyPath
            )
            # Set policy net in eval mode
            self._policynet.eval()

            self._optimizer = None
            self._targetnet = None

        # Imitation learning
        if args["expert"]["enabled"]:
            # Let's try FSB for now 
            self._expert = FsbBranching(net)
            # self._expert = KfsbBranching(net)

            self._useExpertUntilEpisode = args["expert"]["useExpertUntilEpisode"]
            self._learnToImitateExpertEpochs = args["expert"][
                "learnToImitateExpertEpochs"
            ]
        else:
            self._expert = None
            self._useExpertUntilEpisode = 0
            self._learnToImitateExpertEpochs = 0

        # Compute static adjacency matrix and index map based on net info
        self._alpha = RLNeuronBranching._computeadjacencymatrix(
            self._bidirectional, self._layers, self._nnodes
        )

        # Compute static index map
        self._inode = RLNeuronBranching._computeinode(self._layers, self._nnodes)

        # Previous state & action during the BnB process
        # is not stored in the BnBTree. So, instead of
        #
        # self._state = None
        # self._action = None
        #
        # we now initialise the branching tree:
        self._tree = BnBTree()

        # We still want to know who done it though
        self._who = None

        # We also want to know if the last step has been logged
        # if the verification was successful
        self._laststeptaken = False

        # __init__ ends

    def updateTree(self, uids, pids, uid, solved):
        self._tree.updateTree(uids, pids, uid, solved)

    @staticmethod
    def _computeadjacencymatrix(bidirectional, layers, nnodes):
        """
        Creates adjacency matrix.
        """

        offset = {}
        nl = 0
        for layer, size in zip(layers, nnodes):
            offset[layer] = nl
            nl += size

        edges = []

        for i in range(len(layers) - 1):
            # Get the i-th and (i+1)-th layer name
            src = layers[i]
            dst = layers[i + 1]

            # Create all-to-all connections between the two layers
            sources = torch.arange(nnodes[i]) + offset[src]
            destinations = torch.arange(nnodes[i + 1]) + offset[dst]

            # Create full bipartite connection using meshgrid
            sourcegrid, destinationgrid = torch.meshgrid(
                sources, destinations, indexing="ij"
            )

            # Flatten and add both directions
            edges.append(
                torch.stack(
                    [sourcegrid.reshape(-1), destinationgrid.reshape(-1)], dim=1
                )
            )
            # Optional backward direction: from dst to src (only if bidirectional)
            if bidirectional:
                edges.append(
                    torch.stack(
                        [destinationgrid.reshape(-1), sourcegrid.reshape(-1)], dim=1
                    )
                )

        # Self-loops
        loops = torch.stack([torch.arange(nl), torch.arange(nl)], dim=1)
        edges.append(loops)

        # Concatenate all edges and transpose to edge_index format
        alpha = torch.cat(edges, dim=0).t().contiguous()  # shape [2, E]
        return alpha

    def filteredges(self, mask):
        """
        Filters edges based on either an edge-wise mask (preferred)
        or a node-wise mask. Works with mask shapes like (E,),
        (B,S,T), or anything that flattens to E. For node masks,
        shape should be (N,) or broadcastable to N.
        """
        assert self._alpha is not None
        alpha = self._alpha
        # alpha is expected to be edge_index of shape [2, E]
        assert alpha.dim() == 2 and alpha.size(0) == 2, "alpha must be [2, E] edge_index"

        src, dst = alpha.long()
        E = alpha.size(1)

        # --- Normalize mask ---
        m = mask
        # Flatten everything
        m_flat = m.reshape(-1).bool()

        if m_flat.numel() == E:
            # Case 1: edge-wise mask (your 8×16×16, 16×8×8, 100 cases)
            keep = m_flat
            return alpha[:, keep]

        # Otherwise, try node-wise mask
        # Build a node-valid vector of length >= max node index + 1
        N_needed = int(max(src.max().item(), dst.max().item())) + 1

        # If mask is not 1D nodes, try to squeeze and broadcast
        if m.dim() > 1:
            # Heuristic: if it's square (S,S) or (B,S,S), reduce to per-node by "any edge touches node"
            if m.size(-1) == m.size(-2):
                # Reduce over edge-dimension to node activity
                # (node considered active if any incident edge is active)
                node_valid = (m.any(dim=-1) | m.any(dim=-2))
                # If batch present, merge batches by OR
                while node_valid.dim() > 1:
                    node_valid = node_valid.any(dim=0)
                valid_nodes = node_valid.bool().reshape(-1)
            else:
                valid_nodes = m.reshape(-1).bool()
        else:
            valid_nodes = m_flat

        if valid_nodes.numel() < N_needed:
            raise ValueError(
                f"Node mask too small: need >= {N_needed} nodes, got {valid_nodes.numel()}"
            )

        keep = valid_nodes[src] & valid_nodes[dst]
        return alpha[:, keep]

        # ZD, this is the old version of filter_edges(), I am keeping it becuase the above one is still under experiment.
        # """
        # Filters edges based on mask.
        # """
        # assert self._alpha is not None
        # alpha = self._alpha

        # # We are assuming that mask is a vector of size N;
        # # and _alpha is vector of size [2, N]

        # # Filter edges where both src and dst are active
        # valid = mask.bool()
        # src, dst = alpha
        # splitable = valid[src] & valid[dst]
        # p = alpha[:, splitable]
        # debug(f"Adjacency matrix shape is {p.shape}")
        # return p

    @staticmethod
    def _computeinode(layers, nnodes):
        """
        Creates index map.
        """
        inode = []
        for layer, size in zip(layers, nnodes):
            # Extend index list
            inode.extend((layer, idx) for idx in range(size))
        return inode

    def atoi(self, action):
        """
        Converts action to index.
        """
        # Have we computed the index already?
        assert self._inode is not None

        if isinstance(action, torch.Tensor):
            action = tuple(action.tolist())
        elif isinstance(action, list):
            # Assume this is a list of the form [[lid, nid]]
            assert len(action) == 1
            action = tuple(action[0])

        lid, nid = action
        action = (self._layers[lid], nid)

        # NOTE This is a O(n) search, could be improved
        for idx, a in enumerate(self._inode):
            if a == action:
                return idx

        raise ValueError(f"Invalid action: {action}")
    
    def itoa(self, index):
        """
        Converts index to action.
        """
        # Have we computed the index already?
        assert self._inode is not None

        if isinstance(index, torch.Tensor):
            index = index.item()

        assert index >= 0 and index < len(self._inode), f"Index {index} out of range"

        return self._inode[index]

    def train(self):
        """
        Trains policy net from experience.
        """
        state, action, reward, terminal, depth = self._buffer.sample(self._batchSize)

        # Loss per state transition
        losses = []

        for s, a, r, f, d in zip(state, action, reward, terminal, depth):

            inputs, alpha = self.stof(s.s0)
            glb = s.s0['glb']

            # ZD, Making RL compatible with GPU
            inputs = inputs.to(self._device)
            alpha = alpha.to(self._device)

            # ZD, memory exhaustion for large models
            # Q = self._policynet(inputs, alpha, glb, d)
            Q = self._policynet(inputs, alpha, glb)

            # Apply mask
            mask = inputs[:, 2]
            Q = self.applymask(Q, mask)

            # Get the action index
            aid = self.atoi(a)

            # Sanity check: make sure target is valid
            if mask[aid] == 0:
                warn(f"Action {a} is masked out. Skipping it.")
                continue

            # Get the Q-value for selected action
            qv = Q[self.atoi(a)]

            with torch.no_grad():
                # Get inputs from next state
                inputs__, alpha__ = self.stof(s.s1)

                # ZD, Making RL compatible with GPU
                inputs__ = inputs__.to(self._device)
                alpha__ = alpha__.to(self._device)

                # Compute Q-values
                logits__ = self._targetnet(inputs__, alpha__, glb, d)
                # Mask Q-values
                logits__ = self.applymask(logits__, inputs__[:, 2])
                # Get target Q-value
                target__ = torch.max(logits__)
                # Q-learn it
                expected = r + self._gamma * target__ * (1 - int(f))

            # ZD, Making RL compatible with GPU
            expected = expected.to(qv.device)

            l0 = F.mse_loss(qv.view(1), expected.view(1))
            losses.append(l0)

        if len(losses) == 0:
            warn("All samples in batch has been skipped.")
            return -1

        loss = torch.stack(losses).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._trainingSteps += 1

        return loss.item()

    def imitate(self):
        """
        Learns to imitate expert.
        """
        state, action, reward, terminal, depth = self._buffer.sample(
            self._batchSize, who=DecisionMaker.EXP, onlysuccessful=True
        )

        # Loss per state transition
        losses = []

        for s, a, _, _, d in zip(state, action, reward, terminal, depth):
            inputs, alpha = self.stof(s.s0)
            glb = torch.min(s.s0['glb']).unsqueeze(0) # NOTE choose the worst case glb
            d = torch.tensor([d], dtype=torch.float32)
            # ZD, Making RL compatible with GPU
            inputs = inputs.to(self._device)
            alpha = alpha.to(self._device)

            # Compute Q-values
            # Q = self._policynet(inputs, alpha, glb, d)
            Q = self._policynet(inputs, alpha, glb)

            # Apply mask
            mask = inputs[:, 2]
            Q = self.applymask(Q, mask) 

            # Get the action index
            aid = self.atoi(a)

            # Sanity check: make sure target is valid
            if mask[aid] == 0:
                warn(f"Action {a} is masked out. Skipping it.")
                continue

            # Get action target
            # target = torch.tensor(aid, dtype=torch.long)

            # ZD, Making RL compatible with GPU
            target = torch.tensor(aid, dtype=torch.long, device=Q.device)

            # Compute supervised loss
            l0 = F.cross_entropy(Q.unsqueeze(0), target.unsqueeze(0))  # Supervised loss
            losses.append(l0)

        loss = torch.stack(losses).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._trainingSteps += 1
        return loss.item()

    def extractstate(self, state):
        """
        Extracts mask, lower bounds, upper bounds, and global lower bound from ab-crown state.
        """
        msk = OrderedDict()
        lbs = OrderedDict()
        ubs = OrderedDict()
        glb = None

        def __copy_tensor(t):
            assert isinstance(t, torch.Tensor)
            # Copy tensor and move to CPU memory
            return t.detach().clone().cpu()

        for _, layer in enumerate(self._layers):
            msk[layer] = __copy_tensor(state["mask"][layer])
            lbs[layer] = __copy_tensor(state["lower_bounds"][layer])
            ubs[layer] = __copy_tensor(state["upper_bounds"][layer])

        if state["global_lb"].shape != state["thresholds"].shape:
            raise ValueError(
                f"Shape mismatch between global lower bounds and thresholds ({state['global_lb'].shape} vs {state['thresholds'].shape})"
            )

        # Let's also squeeze the tensor to go from shape [1, K] to [K]
        # glb = __copy_tensor(state["global_lb"] - state["thresholds"]).squeeze(0)

         # ZD, Making RL compatible with GPU
        device = state["global_lb"].device  # or whichever one you trust to be on correct device
        glb = __copy_tensor((state["global_lb"].to(device) - state["thresholds"].to(device))).squeeze(0)


        return {"msk": msk, "lbs": lbs, "ubs": ubs, "glb": glb}

    @torch.no_grad()
    def stof(self, state):
        """
        Encodes state to input features for the policy network.
        """
        # Per layer features
        features = []
        n_layers = len(self._layers)

        for i, layer in enumerate(self._layers):
            # Mask, lower and upper bounds for nodes in layer
            msk = state["msk"][layer].squeeze(0)  # shape becomes from [1, N] to [N]
            lbs = state["lbs"][layer].squeeze(0)
            ubs = state["ubs"][layer].squeeze(0)
            
            # ZD, added this to make sure all types of layers of diffrent models is supported (e.g. cifar100 with FC + Conv. + Residual, ReLU + BatchNorm layers)
            if msk.shape != lbs.shape or msk.shape != ubs.shape:
                lbs = lbs.view(-1)
                ubs = ubs.view(-1)

            # Compute normalized layer index
            norm_layer_idx = torch.full_like(msk, fill_value=i / (n_layers - 1) if n_layers > 1 else 0.0, dtype=torch.float32)

            # Add as 4th feature
            layer_features = torch.stack([lbs, ubs, msk, norm_layer_idx], dim=1)  # [N, 4]
            # debug(f"[ZRL] layer_features: {layer_features}")
            features.append(layer_features)

            # features.append(torch.stack([lbs, ubs, msk], dim=1))  # [N, 3]

        x = torch.cat(features, dim=0).float()  # [N, 3]

        alpha = self.filteredges(x[:, 2]) if self._useDynamicGraph else self._alpha


        # ZD, Normalize the input data
        # def robust_scale(x: torch.Tensor, q: float = 0.9, eps: float = 1e-6):
        #     x=x.abs()
        #     s = torch.quantile(x, q).clamp_min(eps)
        #     return (x / s).clamp(-10, 10)

        # def _to_1d_float(x: torch.Tensor) -> torch.Tensor:
        #     # Handle None defensively (shouldn't happen, but better error)
        #     if x is None:
        #         raise RuntimeError("[stof] Got None tensor")
        #     # Remove any stray batch dim(s) and flatten

        #     x = x.squeeze(0).reshape(-1).to(torch.float32)

        #     return x

        # for i, layer in enumerate(self._layers):
        #     lb = state["lbs"][layer]
        #     ub = state["ubs"][layer]
        #     mk = state["msk"][layer]

        #     # Force 1D float for all three (robust to [C,H,W] vs [N] vs [1,N], etc.)
        #     lbs = _to_1d_float(lb)
        #     ubs = _to_1d_float(ub)
        #     msk = mk.reshape(-1)

        #     # (A) convert to center/radius
        #     ctr = 0.5 * (lbs + ubs)
        #     rad = 0.5 * (ubs - lbs).clamp_min(0)

        #     # (B) robust-normalize per state
        #     ctr_n = robust_scale(ctr, q=0.9)
        #     rad_n = robust_scale(rad, q=0.9)

        #     # (C) keep mask in {0,1}
        #     msk_n = msk.clamp(0, 1)

        #     # (D) normalized layer index in [0,1]
        #     norm_val = (i / (n_layers - 1)) if n_layers > 1 else 0.0
        #     lidx = torch.full((msk.numel(),), float(norm_val),
        #                     dtype=torch.float32, device=msk.device)

        #     # FINAL [N, 4]
        #     layer_features = torch.stack([ctr_n, rad_n, msk_n, lidx], dim=1)
        #     features.append(layer_features)
    

        # x = torch.cat(features, dim=0).float()  # [sum(n_i), 4]

    
        return x, alpha

    @torch.no_grad()
    def ltof(self, logits):
        """
        Decodes logits to ab-crown-readable features.
        """
        assert self._inode is not None
        assert logits.shape[0] == len(self._inode), "Logits size does not match inode"

        # Collect per-layer logits
        scores = defaultdict(list)

        for idx, (layer, _) in enumerate(self._inode):
            scores[layer].append(logits[idx])

        for layer in scores:
            scores[layer] = torch.stack(scores[layer]).unsqueeze(
                0
            )  # shape becomes from [N] to [1, N]
            # debug(scores[layer].shape)

        return dict(scores)

    def applymask(self, logits, mask):
        assert (
            logits.shape == mask.shape
        ), f"Shape mismatch between logits ({logits.shape}) and mask ({mask.shape})"
        # Double check mask
        assert (mask == 0).sum() + (
            mask == 1
        ).sum() == mask.numel(), "Mask contains invalid values"
        x = logits.clone()
        #
        # NOTE
        #
        # Later on, ab-crown multiplies scores with mask.
        # However, -inf x 0 results in a NaN value.
        #
        # We set the Q-value to something very small, but
        # we are assuming that the minimum Q-value is not
        # so small.
        #
        x[mask == 0] = -1e9  # Rather than float("-inf")
        return x

    @torch.no_grad()
    def act(self, state):
        """
        Given current state, compute scores.
        """
        # Keep stats
        self._nmodels_actions += 1
        # Who's done it?
        self._who = DecisionMaker.DQN

        inputs, alpha = self.stof(state)

        # debug("Input ihas shape", inputs.shape)
        # debug("Alpha has shape", alpha.shape)

        # logits = self._policynet(inputs, alpha, state['glb'].min().unsqueeze(0), self._dpth)
        logits = self._policynet(inputs, alpha, state['glb'].min().unsqueeze(0))
        logits = self.applymask(logits, inputs[:, 2])

        scores = self.ltof(logits)
        return scores

    @torch.no_grad()
    def actRandomly(self, state):
        """
        Given current state, compute scores randomly.
        """

        # Keep stats
        self._nrandom_actions += 1
        # Who's done it?
        self._who = DecisionMaker.RND

        scores = {}
        for layer in self._layers:
            # Random score from 0 to 1
            lbs = state["lbs"][layer]
            msk = state["msk"][layer]
            scores[layer] = msk * torch.rand_like(lbs).flatten(
                1
            )  # shape: [1, num_nodes]
        return scores

    def actExpertly(self, state):
        """
        Given current state, compute scores randomly.
        """

        # Keep stats
        self._nexpert_actions += 1

        return self._expert.compute_neuron_scores(state)

    def shouldActExpertly(self):

        if self._mode == ExecutionMode.TEST:
            return False
        # ZD, for only imitation mode
        if self._training_mode == "imitation":
            # debug("[ZRL] Acting based on expert imitation only.")
            return True
        if not self._expert:
            return False

        # Use expert for the first N episodes
        #
        # Since we are amidst an episode, number of episodes can be zero. So,
        #
        #   0 is True.
        #   1 is True.
        # N-1 is True.
        #
        # But N is False: we have completed N episodes)
        #
        return self._buffer.nepisodes < self._useExpertUntilEpisode

    def shouldActRandomly(self):
        if not self._isgreedy:
            return False
        if self._mode == ExecutionMode.TEST:
            return False
        result = random.uniform(0, 1) < self._epsilon
        # Apply epsilon decay? ZD, I have moved this to after every episode
        # if self._epsilonDecay != 0:
        #     self._epsilon = max(self._minEpsilon, self._epsilon * self._epsilonDecay)
        #     debug(f"[ZRL] epsilon value: {self._epsilon}")
        return result

    def shouldUpdateTarget(self):
        # Update target net every N steps
        return self._buffer.nepisodes % self._updateTargetFrequency == 0

    def shouldUpdatePolicy(self):
        # Update policy net every N steps; but only if we don't imitate an expert
        if self.shouldImitateExpert():
            # The policy has already been updated to mimic the expert heurestic!
            return False
        return self._buffer.nepisodes % self._updatePolicyFrequency == 0

    def shouldImitateExpert(self):
        # An episode has just completed; it was the last episode we used our expert:
        return self._buffer.nepisodes == self._useExpertUntilEpisode

    def laststep(self, domain):
        # This function should be called once!
        assert not self._laststeptaken

        # What is the domain's parent id?
        pid = domain["pids"]

        # Extract features from current domain
        s1 = self.extractstate(domain)
        
        # There must a parent state stored!
        assert not self._tree.isempty()
        # Unpack node.state, node.action, node.who, node.depth, node.solved
        s0, action, who, depth, _ = self._tree.get(pid)

        # Store step in buffer
        self._buffer.step(s0, s1, action, who, depth)

        self._laststeptaken = True
        return

    def finalize(self, result):
        # At this point, we have taken our last action.
        # But we don't know what is the last state!
        #
        # Luckily, we have called the `laststep` function beforehand.
        #
        assert (
            result != "safe" or self._laststeptaken
        ), "If result is safe, then we should have a taken a last step"

        # self._tree.dump()

        # Has the BnB process completed successfully?
        self._buffer.episodeComplete(result == "safe", **self._rewardconfig)

        # Store buffer
        self._buffer.store(path=self._bufferPath)

        if self._mode == ExecutionMode.TRAIN:
            #
            # At this point, we have already incremented the episode counter,
            # so self._buffer.nepisodes > 0.
            #
            # Should we try to imitate the behavior of the expert heurestic?

            if self.shouldImitateExpert():
                # Train on all expert data for multiple epochs
                for epoch in range(self._learnToImitateExpertEpochs):
                    loss = self.imitate()
                    info(f"[Imitate] @{(epoch + 1):03d}: loss {loss:6.3f}")

            if self._training_mode != "imitation":
                # Should we learn from experience?
                if self.shouldUpdatePolicy():
                    # Train on multiple batches
                    for bid in range(self._trainingLoop):
                        loss = self.train()
                        info(f"[Q-learn] @{(bid + 1):03d}: loss {loss:6.3f}")

                    # ZD, Moving the epsilon decay here - after fitting.
                    if self._epsilonDecay != 0:
                        self._epsilon = max(self._minEpsilon, self._epsilon * self._epsilonDecay)
                        debug(f"[ZRL] epsilon value: {self._epsilon}")

                    # ZD, Save current epsilon for continuity across episodes or training runs
                    torch.save({"epsilon": self._epsilon}, self._epsilonPath)
                    # info(f"Saved epsilon to {self._epsilonPath}: {self._epsilon}")

            # Should we update the target net (mirror policy net)?
            if self.shouldUpdateTarget():
                self._targetnet.mirror(self._policynet)

            # Store policy network, since we are in training mode
            self._policynet.store(
                opt=self._optimizer,
                step=self._trainingSteps,
                path=self._policyPath,
            )

            # Store target network
            self._targetnet.store(path=self._targetPath)

        return

    def compute_neuron_scores(self, d, **kwargs):
        """
        Computes branching scores to decide on a new action.
        """
        if self.shouldActExpertly():
            raise NotImplementedError

        # Extract state
        state = self.extractstate(d)

        if self.shouldActRandomly():
            # debug("[ZRL] Random action selected (exploration).")
            return self.actRandomly(state)
        else:
            # debug("[ZRL] Policy action selected (exploitation).")
            return self.act(state)

    def get_branching_decisions(self, domains, split_depth=1, **kwargs):
        #
        # Dictionary 'domains' keys are:
        #
        # 'mask'               | Already used
        # 'lower_bounds'       | Already used
        # 'upper_bounds'       | Already used
        # 'global_lb'          | Already used
        # 'lAs'                |
        # 'alphas'             |
        # 'betas'              |
        # 'intermediate_betas' |
        # 'history'            |
        # 'split_history'      |
        # 'depths'             |
        # 'cs'                 |
        # 'thresholds'         |
        # 'x_Ls'               | Input lower bounds
        # 'x_Us'               | Input upper bounds
        # 'input_split_idx'    |
        #
        # debug("lAs")
        # # Alphas is a dictionary
        # for l0 in domains["lAs"].keys():
        #     t = domains["lAs"][l0]
        #     debug(f"{l0:5s} → {t.shape}")
        #
        # debug(f"Previous act = {'unknown' if self._action is None else self._action}")
        # debug(
        #     f"Previous glb = {'unknown' if self._state is None else self._state['glb']}"
        # )
        # debug(f"Current  glb = {domains['global_lb']}")
        #

        def _batch1(x):
            if isinstance(x, list):
                return len(x) == 1
            return False

        # Currently, we are working under the assumption that
        # batch size is 1!
        assert _batch1(domains["depths"])
        assert _batch1(domains["uids"])
        assert _batch1(domains["pids"])

        # ZD, getting the glb
        # debug(f"[ZRL] glb: {domains['global_lb']} and its shape is: {len(domains['global_lb'])}")
        self._glb = domains['global_lb']

        # What is the depth of the domain in the BnB tree?
        depth = domains["depths"][0]
        # What is the domain unique id?
        uid = domains["uids"][0]
        # What is the domain's parent id?
        pid = domains["pids"][0]

        # Extract features from current domain
        s1 = self.extractstate(domains)
        s0 = None

        if not self._tree.isempty():
            # The tree is not empty, so there must a parent state stored!  

            # Unpack node.state, node.action, node.who, node.depth, node.solve
            s0, action, who, d, _ = self._tree.get(pid)

            # Store step in buffer
            self._buffer.step(s0, s1, action, who, d)

        # else:
        #     debug("Tree is empty!")
        # Get new branching decision

        if self.shouldActExpertly():
            # Keep stats
            self._nexpert_actions += 1
            # Who's done it?
            self._who = DecisionMaker.EXP

            result = self._expert.get_branching_decisions(
                domains, split_depth=split_depth, **kwargs
            )
        else:
            # Handle stats in compute_neuron_scores
            result = super().get_branching_decisions(
                domains, split_depth=split_depth, **kwargs
            )

        # What is the action?
        action = result[0]

        # Update tree
        self._tree.set(uid, pid, depth, s1, action, self._who)

        return result

    @staticmethod
    def split_by_steps(exp, path, train_ratio=0.8, seed=None, n_steps=None, save=False):
        """
        Split a ReplayBuffer into train/test by steps with deterministic shuffling.

        Args:
            path (str): Path to a pickled ReplayBuffer-like object that has a `.buffer` attribute (list-like).
            train_ratio (float): Fraction of selected steps to put into the train split (0.0–1.0).
            seed (int|None): If given, use this seed for a reproducible shuffle of step indices.
            n_steps (int|None): If given, use only the first `n_steps` steps *after* shuffling.
                                If None and `ask=True`, you'll be prompted. If None and `ask=False`, uses all steps.
            save (bool): If True, saves two new buffers next to `path`, with filenames including seed and n_steps.

        Returns:
            (train_buffer, test_buffer): Deep-copied buffers with `.buffer` replaced by the split step lists.
        """
        # -- load
        with open(path, "rb") as f:
            buffer = pickle.load(f)

        if not hasattr(buffer, "buffer"):
            raise AttributeError("[ZRL]Loaded object does not have a `.buffer` attribute.")

        steps = list(buffer.buffer)
        total = len(steps)
        print(f"[ZRL][Split] Loaded buffer with {total} steps.")

        # -- clamp n_steps
        if n_steps is None:
            n_steps = total
        if n_steps == -1:
            n_steps = total
        if n_steps <= 0:
            raise ValueError("[ZRL]n_steps must be a positive integer.")
        if n_steps > total:
            print(f"[ZRL][Split] Requested n_steps={n_steps} exceeds total={total}; using all {total}.")
            n_steps = total

        # -- deterministic shuffle via local RNG; shuffle indices, not the list
        indices = list(range(total))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(indices)

        # -- select subset
        selected_idx = indices[:n_steps]
        selected_steps = [steps[i] for i in selected_idx]

        # -- split
        if not (0.0 <= train_ratio <= 1.0):
            raise ValueError("[ZRL]train_ratio must be between 0.0 and 1.0")

        split_index = int(train_ratio * n_steps)
        train_steps = selected_steps[:split_index]
        test_steps  = selected_steps[split_index:]

        # -- deepcopy meta and replace .buffer
        train_buffer = copy.deepcopy(buffer)
        test_buffer  = copy.deepcopy(buffer)
        train_buffer.buffer = train_steps
        test_buffer.buffer  = test_steps

        print(f"[ZRL][Split] Seed: {seed}")
        print(f"[ZRL][Split] Using {n_steps} steps → Train: {len(train_steps)}, Test: {len(test_steps)} (train_ratio={train_ratio:.2f})")

        # -- optional save
        if save:
            # out_train = f"{exp}_train.pkl"
            out_test  = f"{exp}_test.pkl"
            # with open(out_train, "wb") as f:
            #     pickle.dump(train_buffer, f)
            with open(out_test, "wb") as f:
                pickle.dump(test_buffer, f)
            # print(f"[ZRL][Split] Train buffer saved to: {out_train}")
            print(f"[ZRL][Split] Test buffer saved to: {out_test}")

        return train_buffer, test_buffer

    def imitate_from_prepared_data_from_script(self, train_buf, policyPath, epochs, n_steps=None):
        debug("\n\nTraining prepared buffer info:")
        train_buf.info()

        loss_log = []  # To store (epoch, loss)

        for epoch in range(epochs):
            # Sample batch of individual steps
            state, action, reward, terminal, depth = train_buf.step_sample(
                self._batchSize,
                onlysuccessful = True,
                who=DecisionMaker.EXP,
                n_steps = n_steps
            )

            # Losses for each step
            losses = []

            for s, a, _, _, d in zip(state, action, reward, terminal, depth):
                # Convert state to network inputs
                inputs, alpha = self.stof(s.s0)

                # Global lower bound — worst-case scenario
                glb = torch.min(s.s0['glb']).unsqueeze(0)

                # Depth as tensor
                d = torch.tensor([d], dtype=torch.float32)

                # Move everything to device (GPU/CPU)
                inputs = inputs.to(self._device)
                alpha = alpha.to(self._device)
                d = d.to(self._device)
                glb = glb.to(self._device)

                # Forward pass: compute Q-values
                # Q = self._policynet(inputs, alpha, glb, d)
                Q = self._policynet(inputs, alpha, glb)
                

                # Convert action to index (target)
                aid = self.atoi(a)
                target = torch.tensor(aid, dtype=torch.long, device=Q.device)

                # Cross-entropy loss between Q-values and expert action
                loss_step = F.cross_entropy(Q.unsqueeze(0), target.unsqueeze(0))
                losses.append(loss_step)

            # Backpropagation
            loss = torch.stack(losses).mean()
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            self._trainingSteps += 1
            loss_value = loss.item()
            info(f"[Imitate] @{(epoch + 1):03d}: loss {loss_value:6.3f}")
            loss_log.append((epoch + 1, loss_value))

        # Save trained policy
        self._policynet.store(
            opt=self._optimizer,
            step=self._trainingSteps,
            path=policyPath,
        )

        # Return DataFrame of epoch and loss
        df = pd.DataFrame(loss_log, columns=["epoch", "loss"])
        return df
    
    @torch.no_grad()
    def evaluate_imitation_from_script(self, test_buf, policyPath, device=None):
        # Load policy weights
        checkpoint = torch.load(policyPath)
        policynet = self._policynet
        policynet.load_state_dict(checkpoint["theModel"])

        # Set device
        if device is None:
            try:
                device = next(policynet.parameters()).device
            except StopIteration:
                device = "cpu"
        policynet.eval()

        total, correct = 0, 0

        for _, state_pair, expert_a, _, _, depth in test_buf.buffer:
            inputs, alpha = self.stof(state_pair.s0)
            glb = state_pair.s0['glb']

            inputs = inputs.to(device)
            alpha = alpha.to(device)

            # q = policynet(inputs, alpha, glb, depth)
            q = policynet(inputs, alpha, glb)

            if isinstance(q, (tuple, list)):
                q = q[0]
            if q.dim() == 1:
                q = q.unsqueeze(0)

            pred = q.argmax(dim=-1).item()
            ea = self.atoi(expert_a)

            debug(f"[ZRL] ---> This is pred: {pred} and this is expert action: {ea}")

            correct += int(pred == ea)
            total += 1

        accuracy = correct / max(1, total)
        return accuracy
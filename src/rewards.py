import logging
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs

logger = logging.getLogger(__name__)

_qed_module = None


def _get_qed():
    global _qed_module
    if _qed_module is None:
        from rdkit.Chem import QED as _qed
        _qed_module = _qed
    return _qed_module


_sa_scorer = None


def _get_sa_scorer():
    global _sa_scorer
    if _sa_scorer is None:
        try:
            from rdkit.Chem import RDConfig
            import os, sys
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.insert(0, sa_path)
            import sascorer as _sa
            _sa_scorer = _sa
        except Exception as e:
            logger.warning(f"Could not load SA scorer: {e}. SAReward will return 0.")
    return _sa_scorer


class RewardFunction:

    def __call__(self, mol: Optional[Chem.Mol]) -> float:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ValidityReward(RewardFunction):

    def __call__(self, mol):
        if mol is None:
            return 0.0
        try:
            Chem.SanitizeMol(mol)
            smi = Chem.MolToSmiles(mol)
            if smi is None:
                return 0.0
            frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            return 1.0 if len(frags) == 1 else 0.0
        except Exception:
            return 0.0


class QEDReward(RewardFunction):

    def __call__(self, mol):
        if mol is None:
            return 0.0
        try:
            Chem.SanitizeMol(mol)
            return float(_get_qed().qed(mol))
        except Exception:
            return 0.0


class SAReward(RewardFunction):

    def __call__(self, mol):
        if mol is None:
            return 0.0
        scorer = _get_sa_scorer()
        if scorer is None:
            return 0.0
        try:
            Chem.SanitizeMol(mol)
            sa = scorer.calculateScore(mol)
            return max(0.0, (10.0 - sa) / 9.0)
        except Exception:
            return 0.0


class LogPReward(RewardFunction):

    def __init__(self, target: float = 2.5, low: float = 0.0, high: float = 5.0):
        self.target = target
        self.low = low
        self.high = high

    def __call__(self, mol):
        if mol is None:
            return 0.0
        try:
            logp = Descriptors.MolLogP(mol)
            if self.low <= logp <= self.high:
                half_range = (self.high - self.low) / 2.0
                return max(0.0, 1.0 - abs(logp - self.target) / half_range)
            return 0.0
        except Exception:
            return 0.0

    def __repr__(self):
        return f"LogPReward(target={self.target}, low={self.low}, high={self.high})"


class MolecularWeightReward(RewardFunction):

    def __init__(self, target: float = 350.0, low: float = 150.0, high: float = 500.0):
        self.target = target
        self.low = low
        self.high = high

    def __call__(self, mol):
        if mol is None:
            return 0.0
        try:
            mw = Descriptors.ExactMolWt(mol)
            if self.low <= mw <= self.high:
                half_range = (self.high - self.low) / 2.0
                return max(0.0, 1.0 - abs(mw - self.target) / half_range)
            return 0.0
        except Exception:
            return 0.0


class TanimotoToTargetReward(RewardFunction):

    def __init__(self, target_smiles: str, radius: int = 2, nbits: int = 2048):
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")
        self.target_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        self.radius = radius
        self.nbits = nbits

    def __call__(self, mol):
        if mol is None:
            return 0.0
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nbits)
            return float(DataStructs.TanimotoSimilarity(fp, self.target_fp))
        except Exception:
            return 0.0


class CompositeReward(RewardFunction):

    def __init__(self, rewards: List[RewardFunction], weights: List[float]):
        assert len(rewards) == len(weights), "rewards and weights must have same length"
        self.rewards = rewards
        self.weights = weights

    def __call__(self, mol):
        return sum(w * r(mol) for w, r in zip(self.weights, self.rewards))

    def __repr__(self):
        parts = [f"{w}*{r}" for w, r in zip(self.weights, self.rewards)]
        return f"CompositeReward([{', '.join(parts)}])"


def build_reward_from_cfg(cfg) -> RewardFunction:
    name = getattr(cfg.train, 'rl_reward', 'qed')

    if name == 'validity':
        return ValidityReward()
    elif name == 'qed':
        return QEDReward()
    elif name == 'sa':
        return SAReward()
    elif name == 'logp':
        return LogPReward()
    elif name == 'mw':
        return MolecularWeightReward()
    elif name == 'tanimoto':
        target = cfg.train.rl_reward_target_smiles
        return TanimotoToTargetReward(target)
    elif name == 'composite':
        weights = list(cfg.train.rl_reward_weights)
        rewards = [ValidityReward(), QEDReward(), SAReward()]
        assert len(weights) == len(rewards), \
            f"rl_reward_weights has {len(weights)} entries, expected {len(rewards)}"
        return CompositeReward(rewards, weights)
    else:
        raise ValueError(f"Unknown reward: {name}")


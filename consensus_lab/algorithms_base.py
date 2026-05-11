from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

@dataclass
class ConsensusResult:
    data_name: str
    algorithm: str
    method: str
    m: int
    runs: int
    seed: int
    nmi_mean: float
    nmi_std: float
    ari_mean: float
    ari_std: float
    f_mean: float
    f_std: float
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        extra = out.pop('extra')
        out.update(extra)
        return out

class ConsensusAlgorithm(ABC):
    name: ClassVar[str]
    display_name: ClassVar[str]
    accepts_params: ClassVar[set[str]] = set()

    @abstractmethod
    def run(self, dataset_path: str | Path, m: int=40, runs: int=20, method: str='average', seed: int=19, **kwargs: Any) -> ConsensusResult:
        ...

    def __repr__(self) -> str:
        return f'{type(self).__name__}(name={self.name!r})'

class _RunFnAdapter(ConsensusAlgorithm):

    def __init__(self, name: str, display_name: str, run_fn: Callable[..., dict[str, Any]], accepts_params: set[str] | None=None, algorithm_label: str | None=None) -> None:
        self.name = name
        self.display_name = display_name
        self._run_fn = run_fn
        self.accepts_params = accepts_params or set()
        self._algorithm_label = algorithm_label or name

    def run(self, dataset_path: str | Path, m: int=40, runs: int=20, method: str='average', seed: int=19, **kwargs: Any) -> ConsensusResult:
        passable = {k: v for k, v in kwargs.items() if k in self.accepts_params}
        ignored = set(kwargs) - passable.keys()
        raw = self._run_fn(dataset_path=dataset_path, seed=seed, m=m, cnt_times=runs, method=method, **passable)
        known = {'data_name', 'nmi_mean', 'nmi_std', 'ari_mean', 'ari_std', 'f_mean', 'f_std'}
        extra = {k: v for k, v in raw.items() if k not in known}
        if ignored:
            extra['ignored_params'] = sorted(ignored)
        return ConsensusResult(data_name=str(raw.get('data_name', Path(dataset_path).stem)), algorithm=self._algorithm_label, method=method, m=int(m), runs=int(runs), seed=int(seed), nmi_mean=float(raw['nmi_mean']), nmi_std=float(raw['nmi_std']), ari_mean=float(raw['ari_mean']), ari_std=float(raw['ari_std']), f_mean=float(raw['f_mean']), f_std=float(raw['f_std']), extra=extra)

class AlgorithmRegistry:
    _instances: ClassVar[dict[str, ConsensusAlgorithm] | None] = None

    @classmethod
    def _build(cls) -> dict[str, ConsensusAlgorithm]:
        from hierarchical_consensus import run_hierarchical_consensus
        from hierarchical_consensus_modified import run_weighted_hierarchical_consensus
        from sdgca import run_sdgca
        from sdgca_modified import run_sdgca_modified
        return {'hierarchical_baseline': _RunFnAdapter(name='hierarchical_baseline', display_name='Иерархическая базовая версия', run_fn=run_hierarchical_consensus, accepts_params={'selection_strategy', 'qd_alpha'}, algorithm_label='hierarchical_baseline'), 'hierarchical_weighted': _RunFnAdapter(name='hierarchical_weighted', display_name='Иерархическая взвешенная версия', run_fn=run_weighted_hierarchical_consensus, accepts_params={'sharpen', 'selection_strategy', 'qd_alpha'}, algorithm_label='hierarchical_weighted'), 'sdgca': _RunFnAdapter(name='sdgca', display_name='SDGCA', run_fn=run_sdgca, accepts_params={'nwca_para', 'eta', 'theta', 'selection_strategy', 'qd_alpha'}, algorithm_label='sdgca'), 'sdgca_modified': _RunFnAdapter(name='sdgca_modified', display_name='SDGCA, модифицированная версия', run_fn=run_sdgca_modified, accepts_params={'nwca_para', 'eta', 'theta', 'diffusion_time', 'adaptive_tau', 'tau_percentile', 'selection_strategy', 'qd_alpha'}, algorithm_label='sdgca_modified')}

    @classmethod
    def get(cls, name: str) -> ConsensusAlgorithm:
        if cls._instances is None:
            cls._instances = cls._build()
        try:
            return cls._instances[name]
        except KeyError as exc:
            raise KeyError(f'Неизвестный алгоритм: {name!r}. Доступны: {sorted(cls._instances)}') from exc

    @classmethod
    def all(cls) -> list[ConsensusAlgorithm]:
        if cls._instances is None:
            cls._instances = cls._build()
        return list(cls._instances.values())

    @classmethod
    def names(cls) -> list[str]:
        if cls._instances is None:
            cls._instances = cls._build()
        return sorted(cls._instances)
__all__ = ['AlgorithmRegistry', 'ConsensusAlgorithm', 'ConsensusResult']

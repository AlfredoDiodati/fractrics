# Mid-level classes for Hidden Markov Models

from dataclasses import dataclass, field
from fractrics._components.core import ts_metadata

@dataclass(frozen=True)
class hmm_metadata(ts_metadata):
    
    num_latent: int = 1
  
    filtered: dict = field(default_factory=lambda:{
        'current_distribution': None,
        'distribution_list': None,
        'transition_matrix': None,
        'latent_states': None
    })

# # human_scene_transformer/model/is_hidden_generator_pt.py
# # Apache-2.0 © 2024 The human_scene_transformer Authors.

# from __future__ import annotations
# from typing import Protocol

# import torch
# import gin


# class IsHiddenGenerator(Protocol):
#     """
#     Tiny functional interface so ModelParams can accept either class:
#         tensor = generator(num_agents, train_progress)
#     """
#     def __call__(self, num_agents: int, train_progress: float = 0.0) -> torch.Tensor: ...


# # --------------------------------------------------------------------------- #
# @gin.configurable
# class BPIsHiddenGenerator:
#     """
#     Behavior-Prediction (BP) – hide *all* future frames for every agent.
#     """

#     def __init__(self, num_steps: int, num_history_steps: int):
#         self.num_steps         = num_steps
#         self.num_history_steps = num_history_steps

#     # --------------------------------------------------------------------- #
#     def __call__(self,
#                  num_agents: int,
#                  train_progress: float = 0.0,
#                  *,
#                  device: torch.device | None = None,
#                  dtype: torch.dtype = torch.bool) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         num_agents : int
#             Number of agents in the scene.
#         train_progress : float
#             Ignored here but kept for API parity (0 ↔ 1).
#         device, dtype : optional
#             Manually override tensor device / dtype.

#         Returns
#         -------
#         is_hidden : BoolTensor  shape = (1, a, t, 1)
#             False for history+current (0 … k) and True for future (k+1 … t-1).
#         """
#         if device is None:
#             device = torch.device("cpu")

#         k = self.num_history_steps
#         t = self.num_steps

#         is_hidden = torch.ones((1, num_agents, t, 1), dtype=dtype, device=device)
#         is_hidden[..., : k + 1, :] = False
#         return is_hidden


# # --------------------------------------------------------------------------- #
# @gin.configurable
# class CBPIsHiddenGenerator:
#     """
#     Conditional BP – hide futures for **all agents except the 0-th**.
#     """

#     def __init__(self, num_steps: int, num_history_steps: int):
#         self.num_steps         = num_steps
#         self.num_history_steps = num_history_steps

#     # --------------------------------------------------------------------- #
#     def __call__(self,
#                  num_agents: int,
#                  train_progress: float = 0.0,
#                  *,
#                  device: torch.device | None = None,
#                  dtype: torch.dtype = torch.bool) -> torch.Tensor:
#         """
#         Same signature as BPIsHiddenGenerator.

#         • History/current frames (0…k) are *never* hidden.
#         • Agent index 0 is *never* hidden (conditioned trajectory).
#         • All other agents: future part is hidden.
#         """
#         if device is None:
#             device = torch.device("cpu")

#         k = self.num_history_steps
#         t = self.num_steps

#         is_hidden = torch.ones((1, num_agents, t, 1), dtype=dtype, device=device)

#         # History & current
#         is_hidden[..., : k + 1, :] = False
#         # 0-th agent (often robot)
#         is_hidden[:, 0, :, :]      = False

#         return is_hidden

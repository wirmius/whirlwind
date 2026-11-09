"""Simulated external agentic system for the async proxy-rollout mode.

EXPERIMENTAL — see ``docs/async_proxy_rollout.md``.

Stands in for the real external agent during validation
(single-turn, GSM8K): drives the rollout proxy
exactly the way a real agentic consumer would — fetch prompts, serve
itself completions via ``generate_external``, score them against reference
answers with its OWN reward function, and report scalar rewards out of band
via ``report_reward`` (batched, immediately — the harshest reward-timing
case is the delayed one, configurable via ``--report-delay``).

    prompt source:  the dataloader node's "train" dataset (few-shot system
                    prompt + reference answers — the validated TRL-matched
                    seeding; the sim does NOT re-tokenize or re-seed)
    reward:         local GSM8K accuracy (same regex as the reward node's
                    ``accuracy`` type, read from the reward config YAML) —
                    a real external agent computes its own rewards; the sim
                    mirrors the canonical one so its runs are comparable
                    with the synchronous recipes on the same data

This node is NOT part of the training loop: the async orchestrator never
talks to it. It is pure external traffic — stop it and the updater simply
idles (the design's "proxy keeps serving" property).

Usage (with the cluster already up):
    python3 -m whirlwind.training.sim_external_agent \
        --group-size 4 --batches 200 --sleep 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from typing import Any

from whirlwind.config import RewardConfig
from whirlwind.messages import (
    BatchRequest,
    BatchResult,
    ExternalGenerateRequest,
    ExternalGenerateResult,
    ReportRewardRequest,
    _from_dict,
    _to_dict,
)
from whirlwind.whirlcoms import Node, RpcError

logger = logging.getLogger(__name__)


class SimExternalAgent:
    """Drives the rollout proxy like a real external agentic system."""

    def __init__(
        self,
        *,
        node_id: str = "sim-external-agent",
        proxy_node: str = "rollout",
        dataloader_node: str = "dataloader",
        group_size: int = 4,
        accuracy_regex: str = r"#####?\s*(\d[\d,]*)",
        batches: int = 0,
        sleep_s: float = 0.0,
        report_delay_s: float = 0.0,
        client_id: str = "sim",
    ) -> None:
        self.node_id = node_id
        self.proxy_node = proxy_node
        self.dataloader_node = dataloader_node
        self.group_size = max(1, group_size)
        self.accuracy_regex = accuracy_regex
        self.batches = batches  # 0 = until stopped
        self.sleep_s = sleep_s
        self.report_delay_s = report_delay_s
        self.client_id = client_id
        self._node: Node | None = None

    @property
    def is_running(self) -> bool:
        return self._node.is_running if self._node is not None else False

    async def start(self) -> None:
        self._node = Node(self.node_id, auto_process=True)
        await self._node.start()
        logger.info("Sim external agent '%s' started (proxy=%s)", self.node_id, self.proxy_node)

    async def stop(self) -> None:
        if self._node is not None:
            await self._node.stop()

    # ------------------------------------------------------------------
    # Reward (local, agent-owned — mirrors the reward node's accuracy fn)
    # ------------------------------------------------------------------

    def _score(self, completions: list[str], references: list[str]) -> list[float]:
        pattern = re.compile(self.accuracy_regex)
        rewards: list[float] = []
        for completion, reference in zip(completions, references, strict=True):
            ref_match = pattern.search(reference)
            if not ref_match:
                rewards.append(0.0)
                continue
            ref_answer = ref_match.group(1).replace(",", "")
            comp_match = pattern.search(completion)
            rewards.append(
                1.0
                if comp_match and comp_match.group(1).replace(",", "") == ref_answer
                else 0.0
            )
        return rewards

    # ------------------------------------------------------------------
    # The agent loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        assert self._node is not None
        # Expand reference answers to the served sample count (each prompt
        # produces group_size completions; reference repeats with them).
        served = 0
        n_batches = 0
        while self.is_running and (self.batches <= 0 or n_batches < self.batches):
            t0 = time.monotonic()
            try:
                # 1. Fetch prompts the way any data consumer would.
                raw = await self._node.call(
                    self.dataloader_node,
                    "next_batch",
                    _to_dict(BatchRequest(dataset_name="train", step=n_batches)),
                )
                if raw is None:
                    raise RpcError("dataloader returned None")
                batch = _from_dict(BatchResult, raw)
                if not batch.ok:
                    raise RpcError(f"dataloader error: {batch.error}")
                prompts = batch.to_prompt_batch()

                # 2. Serve ourselves completions through the proxy.
                result = await self._node.call_typed(
                    self.proxy_node,
                    "generate_external",
                    ExternalGenerateResult,
                    ExternalGenerateRequest(
                        prompts=prompts.prompts,
                        system_prompt=prompts.system_prompt,
                        group_size=self.group_size,
                        client_id=self.client_id,
                    ),
                )
                served += len(result.rollout_ids)

                # 3. Score with the agent's own reward function. References
                #    repeat per completion (prompt i → group_size samples).
                references = [
                    prompts.responses[i // self.group_size]
                    if prompts.responses
                    else ""
                    for i in range(len(result.completions))
                ]
                rewards = self._score(result.completions, references)

                # 4. Report rewards out of band (optionally delayed).
                if self.report_delay_s > 0:
                    await asyncio.sleep(self.report_delay_s)
                reply = await self._node.call(
                    self.proxy_node,
                    "report_reward",
                    _to_dict(
                        ReportRewardRequest(
                            rollout_ids=result.rollout_ids, rewards=rewards
                        )
                    ),
                )
                if not (isinstance(reply, dict) and reply.get("ok")):
                    logger.warning("report_reward not ok: %r", reply)

                n_batches += 1
                elapsed = time.monotonic() - t0
                mean_r = sum(rewards) / len(rewards) if rewards else 0.0
                self._node.call_bg_no_return(
                    "tracker",
                    "log_metrics",
                    {
                        "metrics": {
                            "sim/reward_mean": mean_r,
                            "sim/batches_served": float(n_batches),
                            "sim/samples_served": float(served),
                            "sim/serve_time_s": elapsed,
                        },
                        "node_id": self.node_id,
                    },
                )
                logger.info(
                    "sim batch %d: %d completions, reward_mean=%.3f (%.1fs)",
                    n_batches,
                    len(result.completions),
                    mean_r,
                    elapsed,
                )
            except RpcError as exc:
                logger.error("sim batch %d failed: %s — retrying in 5s", n_batches, exc)
                await asyncio.sleep(5.0)
                continue

            if self.sleep_s > 0:
                await asyncio.sleep(self.sleep_s)

        logger.info("Sim external agent done: %d batches, %d samples", n_batches, served)


def _load_accuracy_regex(reward_config_path: str | None) -> str:
    if reward_config_path:
        try:
            return RewardConfig.from_yaml(reward_config_path).accuracy_regex
        except Exception as exc:  # fall through to the GSM8K default
            logger.warning(
                "Could not load reward config %s (%s) — using default regex",
                reward_config_path,
                exc,
            )
    return r"#####?\s*(\d[\d,]*)"


async def _main(args: argparse.Namespace) -> None:
    agent = SimExternalAgent(
        node_id=args.node_id,
        proxy_node=args.proxy,
        dataloader_node=args.dataloader,
        group_size=args.group_size,
        accuracy_regex=_load_accuracy_regex(args.reward_config),
        batches=args.batches,
        sleep_s=args.sleep,
        report_delay_s=args.report_delay,
        client_id=args.client_id,
    )
    await agent.start()
    try:
        await agent.run()
    except KeyboardInterrupt:
        pass
    finally:
        await agent.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated external agent (experimental async proxy mode)")
    parser.add_argument("--node-id", default="sim-external-agent")
    parser.add_argument("--proxy", default="rollout", help="rollout proxy node id")
    parser.add_argument("--dataloader", default="dataloader", help="dataloader node id")
    parser.add_argument("--group-size", type=int, default=4, help="completions per prompt")
    parser.add_argument(
        "--reward-config",
        default=None,
        help="reward config YAML to read the accuracy regex from (default: GSM8K pattern)",
    )
    parser.add_argument("--batches", type=int, default=0, help="stop after N batches (0 = until stopped)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds between batches")
    parser.add_argument(
        "--report-delay", type=float, default=0.0, help="delay reward reporting (test the async path)"
    )
    parser.add_argument("--client-id", default="sim")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    asyncio.run(_main(args))

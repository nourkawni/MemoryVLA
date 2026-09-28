"""
xf_serve_policy.py

Thin mirror of robomme_policy_learning/scripts/serve_policy.py that serves
an XFPolicy instead of MME_VLA_Policy. examples/robomme/eval.py (used
UNCHANGED as the client -- its --subgoal_type/--use_oracle flags already
support everything v1 needs) talks to whatever server is listening on
--host/--port; this script is what that server is, for XF.

Role in the system: run this to serve an XF checkpoint (or, for the v1 smoke
test, a randomly-initialized XFModel -- see smoke_test.py, which builds the
policy directly rather than launching this script as a subprocess, matching
how arm_d_dynamic_fusion's smoke test avoids the checkpoint-loading path
entirely since no trained checkpoint exists yet), then point
examples/robomme/eval.py at the printed host:port.
"""

import dataclasses
import enum
import logging
import socket
from pathlib import Path

import tyro

from mme_vla_suite.serving import websocket_policy_server
from mme_vla_suite.training import config as _config

from xattn_fusion.mme_vla_suite.policies.xf_policy import XFPolicy
from xattn_fusion.mme_vla_suite.policies.xf_policy_config import create_xf_trained_policy


class EnvMode(enum.Enum):
    """Supported environments (mirrors scripts/serve_policy.py's EnvMode)."""

    HISTORY_BENCH = "history_bench"


@dataclasses.dataclass
class Checkpoint:
    """Load an XF policy from a trained checkpoint."""

    config: str  # str, training config name registered for this XF run
    dir: str  # str, checkpoint directory (e.g. "runs/ckpts/xf_smoke_test/exp/10000")

    def __post_init__(self):
        self.dir = Path(self.dir)


@dataclasses.dataclass
class Args:
    """Arguments for xf_serve_policy (mirrors scripts/serve_policy.py's Args)."""

    env: EnvMode = EnvMode.HISTORY_BENCH  # EnvMode
    default_prompt: str | None = None  # str | None
    port: int = 8000  # int
    record: bool = False  # bool
    seed: int = 42  # int
    policy: Checkpoint = dataclasses.field(default_factory=lambda: Checkpoint(config="xf_smoke_test", dir="runs/ckpts/xf_smoke_test/exp/latest"))


def create_policy(args: Args) -> XFPolicy:
    """
    What it does: builds an XFPolicy from the requested checkpoint config/dir.

    Returns:
        XFPolicy.

    Example input:
        create_policy(Args(policy=Checkpoint(config="xf_smoke_test", dir="runs/ckpts/xf_smoke_test/exp/10000")))

    Example output:
        <XFPolicy ...>
    """
    return create_xf_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt, seed=args.seed
    )


def main(args: Args) -> None:
    """
    What it does: builds the policy and serves it over websockets, matching
    scripts/serve_policy.py exactly (only the policy construction differs).

    Returns:
        None -- blocks in server.serve_forever().

    Example input:
        main(tyro.cli(Args))

    Example output:
        None
    """
    policy = create_policy(args)
    policy_metadata = policy.metadata

    hostname = socket.gethostname()  # str
    local_ip = socket.gethostbyname(hostname)  # str
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy_metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

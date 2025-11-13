import jax
import jax.numpy as jnp
import os
import orbax.checkpoint as ocp


def _absdir(path: str) -> str:
    """Resolve an absolute path and ensure its parent directory exists."""
    apath = os.path.abspath(path)
    os.makedirs(os.path.dirname(apath), exist_ok=True)
    return apath


def save_checkpoint(ckpt_dir: str, params, W_mean: float, W_std: float) -> str:
    """
    Save a Linen checkpoint using Orbax.
    """
    ckpt_dir = _absdir(ckpt_dir)

    payload = {
        "params": params,
        "W_mean": float(W_mean),
        "W_std": float(W_std),
    }

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir, payload, force=True)
    checkpointer.wait_until_finished()
    print(f"Saved checkpoint to {ckpt_dir}")


def load_checkpoint(ckpt_dir: str, model, example_graph):
    """
    Restore a Linen checkpoint using Orbax.
    """
    ckpt_dir = _absdir(ckpt_dir)

    n = int(example_graph.n_node[0])
    e = int(example_graph.n_edge[0])
    node_mask = jnp.ones((n, 1), dtype=jnp.float32)
    edge_mask = jnp.ones((e, 1), dtype=jnp.float32)

    params_template = model.init(
        jax.random.key(0),
        (example_graph, node_mask, edge_mask),
    )

    template = {
        "params": params_template,
        "W_mean": 0.0,
        "W_std": 1.0,
    }

    checkpointer = ocp.StandardCheckpointer()
    payload = checkpointer.restore(ckpt_dir, target=template)
    return payload

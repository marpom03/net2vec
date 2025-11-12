import flax.serialization as serialization
import jax
import jax.numpy as jnp
import os

def save_checkpoint(path, params, W_mean, W_std):
    """
    Serialize Linen parameters together with normalization statistics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "params": params,
        "W_mean": W_mean,
        "W_std": W_std,
    }
    bytes_data = serialization.to_bytes(payload)
    with open(path, "wb") as f:
        f.write(bytes_data)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, example_graph):
    """
    Restore a Linen parameter PyTree and normalization stats from bytes.   
    """
    with open(path, "rb") as f:
        bytes_data = f.read()

    n = int(example_graph.n_node[0])
    e = int(example_graph.n_edge[0])
    node_mask = jnp.ones((n, 1))
    edge_mask = jnp.ones((e, 1))

    params_template = model.init(jax.random.key(0), (example_graph, node_mask, edge_mask))

    template = {
        "params": params_template,
        "W_mean": 0.0,
        "W_std": 1.0,
    }

    payload = serialization.from_bytes(template, bytes_data)
    return payload


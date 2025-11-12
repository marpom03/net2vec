import jax
import jax.numpy as jnp
import numpy as np
import jraph
from config import NormConfig

def sample_to_jraph(mu, L, R, W, norm: NormConfig):
    """
    Convert a single synthetic queueing-network sample (μ, Λ, R, W) into a
    `jraph.GraphsTuple` used by the MPNN.
    """
    mu = jnp.asarray(np.asarray(mu, np.float32))
    L  = jnp.asarray(np.asarray(L, np.float32))
    R  = jnp.asarray(np.asarray(R, np.float32))
    W  = jnp.asarray(W)

    mu_norm = (mu - norm.mu_shift) / norm.mu_scale
    W_norm  = (W  - norm.W_shift) / norm.W_scale

    nodes = jnp.concatenate([mu_norm, L], axis=1)                
    senders, receivers = jnp.where(R > 0)
    edges = R[senders, receivers].reshape(-1, 1)

    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders.astype(jnp.int32),
        receivers=receivers.astype(jnp.int32),
        globals=jnp.array([W_norm], dtype=jnp.float32),
        n_node=jnp.array([nodes.shape[0]]),
        n_edge=jnp.array([edges.shape[0]])
    )


def load_npz_as_graphs(path, norm: NormConfig):
    """
    Load a dataset of samples from a compressed NPZ file and convert each
    sample to a `jraph.GraphsTuple`.
    """

    data = np.load(path, allow_pickle=True)

    mus = np.array(data["mu"])
    Ls  = data["L"]
    Rs  = data["R"]
    Ws  = data["W"]
    return [sample_to_jraph(mu, L, R, W, norm) for mu, L, R, W in zip(mus, Ls, Rs, Ws)]


def pad_graph_fixed_size(g, max_nodes, max_edges):
    """
    Pad a single `GraphsTuple` to fixed sizes (max_nodes, max_edges) and
    build corresponding masks. This enables XLA-friendly static shapes.
    """

    n = int(g.n_node[0]); e = int(g.n_edge[0])
    pad_n = max_nodes - n; pad_e = max_edges - e

    nodes = jnp.pad(g.nodes, ((0, pad_n), (0, 0)))
    node_mask = jnp.concatenate([jnp.ones((n, 1)), jnp.zeros((pad_n, 1))], axis=0)

    edges = jnp.pad(g.edges, ((0, pad_e), (0, 0)))
    senders = jnp.pad(g.senders, (0, pad_e))
    receivers = jnp.pad(g.receivers, (0, pad_e))
    edge_mask = jnp.concatenate([jnp.ones((e, 1)), jnp.zeros((pad_e, 1))], axis=0)

    g_pad = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=g.globals,
        n_node=jnp.array([max_nodes]),
        n_edge=jnp.array([max_edges]),
    )
    return g_pad, node_mask, edge_mask


def make_loader(graphs, max_nodes, max_edges, batch_size, key):
    """
    Build an infinite randomized loader that:
      1. pads each graph to fixed size with masks,
      2. forms batched graphs via `jraph.batch`,
      3. shuffles the order of prebuilt batches on each iteration.
    """
    
    padded, nms, ems = [], [], []
    for g in graphs:
        gp, nm, em = pad_graph_fixed_size(g, max_nodes, max_edges)
        padded.append(gp); nms.append(nm); ems.append(em)

    num = len(padded)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, num)
    perm = perm[: (num // batch_size) * batch_size].reshape(-1, batch_size)

    batches = []
    nmask_batches = []
    emask_batches = []
    for row in perm:
        batch_graph = jraph.batch([padded[int(i)] for i in row])
        batch_nmask = jnp.concatenate([nms[int(i)] for i in row], axis=0)
        batch_emask = jnp.concatenate([ems[int(i)] for i in row], axis=0)
        batches.append(batch_graph); nmask_batches.append(batch_nmask); emask_batches.append(batch_emask)

    while True:
        key, subkey = jax.random.split(key)
        order = jax.random.permutation(subkey, len(batches))
        for j in order:
            yield (batches[int(j)], nmask_batches[int(j)], emask_batches[int(j)]), key


import torch
import numpy as np
from torch import nn
import math


class FourierMLP(nn.Module):
    def __init__(
        self,
        in_shape=2,
        out_shape=2,
        num_layers=2,
        channels=128,
        zero_init=True,
    ):
        super().__init__()

        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            torch.cat([sin_embed_cond, cos_embed_cond], dim=1)
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int = 64):
        super(TimeEncoding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).float() * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
            nn.GELU(),
        )
        self.register_buffer("pe", pe)

    def forward(self, t: torch.Tensor):
        """
        Arguments:
            t: Tensor [B, ...].
        """
        original_shape = t.shape
        t_flat = t.reshape(-1, 1)
        t_sin = (t_flat * self.pe[None, :]).sin()
        t_cos = (t_flat * self.pe[None, :]).cos()

        t_sin = t_sin.squeeze(1)
        t_cos = t_cos.squeeze(1)

        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        t_emb = self.t_model(t_emb)

        output_shape = original_shape + (t_emb.shape[-1],)
        return t_emb.reshape(output_shape)


class StateEncoding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncoding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU(),
        )

    def forward(self, s):
        return self.x_model(s)


class JointPolicy(nn.Module):
    def __init__(
        self,
        dim: int,
        s_emb_dim: int,
        t_emb_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        zero_init: bool = False,
        state_encoding: nn.Module = None,  # Partially intialized object
        time_encoding: nn.Module = None,
    ):
        super(JointPolicy, self).__init__()

        self.state_encoding = state_encoding(dim=dim)
        self.time_encoding = time_encoding

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                )
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, x, t):
        if t.dim() == 0:
            # Special case for scalar time.
            # Used in flow matching library ODESolver.
            t = t.unsqueeze(-1).expand(x.shape[0])

        if self.state_encoding is not None:
            x = self.state_encoding(x)
        if self.time_encoding is not None:
            t = self.time_encoding(t)

        return self.model(torch.cat([x, t], dim=-1))


class FlowModel(nn.Module):
    def __init__(
        self,
        dim: int,
        s_emb_dim: int,
        t_emb_dim: int,
        hidden_dim: int = 64,
    ):
        super(FlowModel, self).__init__()

        self.state_encoding = StateEncoding(
            dim=dim, hidden_dim=hidden_dim, s_emb_dim=s_emb_dim
        )
        self.time_encoding = TimeEncoding(
            harmonics_dim=t_emb_dim, hidden_dim=hidden_dim, t_emb_dim=t_emb_dim
        )

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, t):
        s = self.state_encoding(s)
        t = self.time_encoding(t)
        return self.model(torch.cat([s, t], dim=-1)).squeeze(-1)


class LangevinScalingModel(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        zero_init: bool = False,
    ):
        super(LangevinScalingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.01)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, t_emb_dim: int, hidden_dim: int = 64):
        super(TimeEncodingPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_emb_dim),
        )
        self.register_buffer("pe", pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int = 64):
        super(StateEncodingPIS, self).__init__()

        self.x_model = nn.Linear(s_dim, s_emb_dim)

    def forward(self, s):
        return self.x_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(
        self,
        s_dim: int,
        s_emb_dim: int,
        t_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        zero_init: bool = False,
    ):
        super(JointPolicyPIS, self).__init__()

        assert s_emb_dim == t_dim, print(
            "Dimensionality of state embedding and time embedding should be the same!"
        )

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Sequential(nn.Linear(s_emb_dim, hidden_dim), nn.GELU()),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, 2 * s_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class FlowModelPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        zero_init: bool = False,
    ):
        super(FlowModelPIS, self).__init__()

        assert s_emb_dim == t_dim, print(
            "Dimensionality of state embedding and time embedding should be the same!"
        )

        self.model = nn.Sequential(
            nn.GELU(),
            nn.Sequential(nn.Linear(s_emb_dim, hidden_dim), nn.GELU()),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers - 1)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class LangevinScalingModelPIS(nn.Module):
    def __init__(
        self,
        s_emb_dim: int,
        t_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 3,
        zero_init: bool = False,
    ):
        super(LangevinScalingModelPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.register_buffer("pe", pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)


def remove_mean(samples, n_particles, n_dimensions):
    """Makes a configuration of many particle system mean-free.

    Parameters
    ----------
    samples : torch.Tensor
        Positions of n_particles in n_dimensions.

    Returns
    -------
    samples : torch.Tensor
        Mean-free positions of n_particles in n_dimensions.
    """
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        condition_time=True,
        tanh=False,
        agg="sum",
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=1,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        # Count function calls
        self.counter = 0

    def forward(self, x, t):
        if t.dim() == 0:
            # Special case for scalar time.
            # Used in flow matching library ODESolver.
            t = t.unsqueeze(-1).expand(x.shape[0])

        n_batch = x.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0].to(x.device), edges[1].to(x.device)]
        x = x.view(n_batch * self._n_particles, self._n_dimension).clone()
        h = torch.ones(n_batch, self._n_particles).to(x.device)

        if self.condition_time:
            t_expanded = t.reshape(n_batch, 1).expand(n_batch, self._n_particles)
            h = h * t_expanded

        h = h.reshape(n_batch * self._n_particles, 1)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel, self._n_particles, self._n_dimension)
        self.counter += 1
        return vel.view(n_batch, self._n_particles * self._n_dimension)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


class EGNN_flow(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        condition_time=True,
        tanh=False,
        agg="sum",
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=1,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time

        self.additional_scalar = nn.Parameter(torch.zeros(1))

        # Count function calls
        self.counter = 0

    def forward(self, x, t):
        assert x.shape[:-1] == t.shape

        batch_dims = x.shape[:-1]
        x = x.view(-1, self._n_particles * self._n_dimension)
        t = t.reshape(-1)

        n_batch = x.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0].to(x.device), edges[1].to(x.device)]
        x = x.view(n_batch * self._n_particles, self._n_dimension).clone()

        h = torch.ones(n_batch, self._n_particles).to(x.device)
        if self.condition_time:
            t_expanded = t.reshape(n_batch, 1).expand(n_batch, self._n_particles)
            h = h * t_expanded
        h = h.reshape(n_batch * self._n_particles, 1)

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        vel = remove_mean(vel, self._n_particles, self._n_dimension)
        flow = vel.sum(dim=(-1, -2)) + self.additional_scalar

        self.counter += 1
        return flow.view(*batch_dims)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.

    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        # print("edge_model", radial, edge_attr)
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        # print("node_model", edge_attr)
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(
        self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
    ):
        # print("coord_model", coord_diff, radial, edge_feat)
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        # trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.agg_type == "mean":
            if node_mask is not None:
                # raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(
                    node_mask[col], row, num_segments=coord.size(0)
                )
                agg = agg / (M - 1)
            else:
                agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception("Wrong coordinates aggregation type")
        # print("update", coord, coord_diff,edge_feat, self.coord_mlp(edge_feat), self.coords_range, agg, self.tanh)
        coord = coord + agg
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        # print("h", h)
        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)

        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)

        return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

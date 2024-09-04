from layers.Mix import *
from layers.Embed import *
from torchdiffeq import odeint as odeint


class boundarypad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.pad(F.pad(input, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: str = "gelu",
            norm: bool = False,
            n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(p=0.1)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        x_mod = F.pad(F.pad(x, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')
        h = self.activation(self.bn1(self.conv1(self.norm1(x_mod))))
        # Second convolution layer
        h1 = F.pad(F.pad(h, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')
        h2 = self.activation(self.bn2(self.conv2(self.norm2(h1))))
        h3 = self.drop(h2)
        # Add the shortcut connection and return
        return h3 + self.shortcut(x)


class ResNet_2D(nn.Module):
    def __init__(self, num_channels, layers, hidden_size):
        super().__init__()
        layers_cnn = []
        activation_fns = []
        self.block = ResidualBlock
        self.inplanes = num_channels

        for idx in range(len(layers)):
            if idx == 0:
                layers_cnn.append(self.make_layer(self.block, num_channels, hidden_size[idx], layers[idx]))
            else:
                layers_cnn.append(self.make_layer(self.block, hidden_size[idx - 1], hidden_size[idx], layers[idx]))

        self.layer_cnn = nn.ModuleList(layers_cnn)
        self.activation_cnn = nn.ModuleList(activation_fns)

    def make_layer(self, block, in_channels, out_channels, reps):
        layers = []
        layers.append(block(in_channels, out_channels))
        self.inplanes = out_channels
        for i in range(1, reps):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, data):
        dx_final = data.float()
        for l, layer in enumerate(self.layer_cnn):
            dx_final = layer(dx_final)
        return dx_final


class Self_attn_conv_reg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Self_attn_conv_reg, self).__init__()
        self.query = self._conv(in_channels, in_channels // 8, stride=1)
        self.key = self.key_conv(in_channels, in_channels // 8, stride=2)
        self.value = self.key_conv(in_channels, out_channels, stride=2)
        self.post_map = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0))
        self.out_ch = out_channels

    def _conv(self, n_in, n_out, stride):
        return nn.Sequential(boundarypad(), nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
                             nn.LeakyReLU(0.3), boundarypad(),
                             nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
                             nn.LeakyReLU(0.3), boundarypad(),
                             nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=stride, padding=0))

    def key_conv(self, n_in, n_out, stride):
        return nn.Sequential(boundarypad(), nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
                             nn.LeakyReLU(0.3), boundarypad(),
                             nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
                             nn.LeakyReLU(0.3), boundarypad(),
                             nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=1, padding=0))

    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = self.query(x).flatten(-2, -1), self.key(x).flatten(-2, -1), self.value(x).flatten(-2, -1)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_ch, size[-2], size[-1]).contiguous())
        return o


class Model(nn.Module):
    def __init__(self, configs, gcn_depth=2,
                 static_feat=None, subgraph_size=20,
                 node_dim=40, node_emb_dim=16, dilation_exponential=1, conv_channels=32, residual_channels=32,
                 in_dim=6, propalpha=0.05, tanhalpha=3,
                 layer_norm_affline=True):
        super(Model, self).__init__()
        self.device = 'cuda:' + str(configs.gpu)
        self.sc_true = configs.sc_true
        self.buildA_true = configs.buildA_true
        self.num_pairs = configs.num_pairs
        self.num_nodes = configs.num_nodes
        self.num_categories = configs.num_categories
        self.dropout = configs.dropout
        self.res_channels = configs.res_channels
        self.end_channels = configs.end_channels
        self.layers = configs.n_layers
        self.seq_length = configs.seq_len
        self.pred_len = configs.pred_len

        self.residual_convs = nn.ModuleList()
        self.line = nn.ModuleList()
        self.res1 = nn.ModuleList()
        self.res2 = nn.ModuleList()
        self.res3 = nn.ModuleList()
        self.res4 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.estimation_gate = nn.ModuleList()
        self.tc_enc = nn.ModuleList()
        self.sc_enc = nn.ModuleList()

        self.idx = torch.arange(int(self.num_pairs)).to(self.device)
        self.gc = graph_constructor(self.num_pairs, subgraph_size, node_dim, self.device, alpha=tanhalpha,
                                    static_feat=static_feat)
        self.st_emb = st_embedding(self.num_pairs, self.num_categories, node_emb_dim, residual_channels,
                                   static_feat=static_feat)
        self.res0 = nn.Conv2d(in_channels=in_dim, out_channels=self.res_channels,
                              kernel_size=(1, self.seq_length), bias=True)

        new_dilation = 1
        for j in range(1, self.layers + 1):
            self.norm.append(
                LayerNorm((residual_channels, self.num_pairs, self.seq_length),
                          elementwise_affine=layer_norm_affline))

            new_dilation *= dilation_exponential

            self.estimation_gate.append(EstimationGate(node_emb_dim,
                                                       4, 64, self.dropout))

            self.tc_enc.append(
                tc_encoder(residual_channels, conv_channels, new_dilation, self.seq_length, self.dropout))
            if self.sc_true:
                self.sc_enc.append(sc_encoder(conv_channels, residual_channels, gcn_depth, self.dropout, propalpha))

            self.line.append(nn.Linear(1, self.seq_length))
            self.res1.append(
                nn.Conv2d(in_channels=residual_channels, out_channels=self.res_channels,
                          kernel_size=(1, self.seq_length), bias=True))
            self.res2.append(
                nn.Conv2d(in_channels=residual_channels, out_channels=self.res_channels,
                          kernel_size=(1, self.seq_length), bias=True))
            self.res3.append(nn.Conv2d(in_channels=conv_channels,
                                       out_channels=self.res_channels,
                                       kernel_size=(1, self.seq_length)))
            self.res4.append(nn.Conv2d(in_channels=conv_channels,
                                       out_channels=self.res_channels,
                                       kernel_size=(1, self.seq_length)))
            self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

        self.end_conv_1 = nn.Conv2d(in_channels=self.res_channels * self.layers,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=configs.pred_len,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.method = 'euler'
        self.out_ch = 2
        self.field_size = 20
        self.pos = True
        self.vel_layers = [1, 1, 1]  # [5, 3, 2]
        self.vel_hidden = [32, 16, 2]  # [128, 64, 2]
        vel_input_channels = 13
        self.vel_f = ResNet_2D(vel_input_channels, self.vel_layers, self.vel_hidden)
        self.use_att = configs.use_att
        if self.use_att:
            self.vel_att = Self_attn_conv_reg(vel_input_channels, 2)
            self.gamma = nn.Parameter(torch.tensor([0.1]))

    def diffusion(self, time_step, velocity_states):
        diffusion_states = velocity_states[:, -self.out_ch:, :, :].float().view(-1, self.out_ch,
                                                                                *velocity_states.shape[2:]).float()
        velocity_vectors = velocity_states[:, :2 * self.out_ch, :, :].float().view(-1, self.out_ch,
                                                                                   *velocity_states.shape[2:]).float()

        t_emb = ((time_step * 100) % 24).view(1, 1, 1, 1).expand(diffusion_states.shape[0], 1,
                                                                 diffusion_states.shape[2], diffusion_states.shape[3])
        sin_t_emb = torch.sin(torch.pi * t_emb / 12 - torch.pi / 2)
        cos_t_emb = torch.cos(torch.pi * t_emb / 12 - torch.pi / 2)

        sin_seas_emb = torch.sin(torch.pi * t_emb / (12 * 365) - torch.pi / 2)
        cos_seas_emb = torch.cos(torch.pi * t_emb / (12 * 365) - torch.pi / 2)

        day_emb = torch.cat([sin_t_emb, cos_t_emb], dim=1)
        seas_emb = torch.cat([sin_seas_emb, cos_seas_emb], dim=1)

        diffusion_grad_x = torch.gradient(diffusion_states, dim=3)[0]
        diffusion_grad_y = torch.gradient(diffusion_states, dim=2)[0]
        nabla_velocity = torch.cat([diffusion_grad_x, diffusion_grad_y], dim=1)

        combined_representation = torch.cat(
            [t_emb / 24, day_emb, seas_emb, nabla_velocity, velocity_vectors, diffusion_states], dim=1)

        if self.use_att:
            velocity_change = self.vel_f(combined_representation) + self.gamma * self.vel_att(combined_representation)
        else:
            velocity_change = self.vel_f(combined_representation)

        v_x, v_y = velocity_vectors.chunk(2, dim=1)

        adv1 = (v_x * diffusion_grad_x).sum(dim=1, keepdim=True) + (v_y * diffusion_grad_y).sum(dim=1, keepdim=True)
        adv2 = diffusion_states * (torch.gradient(v_x, dim=3)[0].sum(dim=1, keepdim=True) +
                                   torch.gradient(v_y, dim=2)[0].sum(dim=1, keepdim=True))

        diffusion_update = adv1 + adv2
        updated_velocity = velocity_change + diffusion_update / 24
        return updated_velocity.view_as(velocity_states[:, -self.out_ch:, :, :])

    def io2od_flow(self, od_flow_mask, index):
        batch_size, pred_len = od_flow_mask.shape[:2]
        num_nodes = self.num_nodes
        device = self.device
        index = index.to(device)
        index = index.expand(-1, pred_len, -1)
        od_flow_zeros = torch.zeros(batch_size, pred_len, num_nodes * num_nodes).to(device)
        od_flow_zeros.scatter_(2, index, od_flow_mask)
        od_flow = od_flow_zeros.view(batch_size, pred_len, 1, num_nodes, num_nodes)
        inflow = od_flow.sum(dim=4)
        outflow = od_flow.sum(dim=3)
        io_flow = torch.cat([inflow, outflow], dim=2)
        return io_flow

    def construct_field(self, data_x, indices, field):
        batch_size = data_x.shape[0]
        empty_field = torch.zeros((batch_size, 2, field * field)).to(self.device)
        indices = indices.repeat_interleave(2, dim=1)
        empty_field.scatter_add_(2, indices, data_x)
        empty_field = empty_field.view(batch_size, 2, field, field)
        field = (empty_field - empty_field.mean()) / empty_field.std()
        return field

    def re_construct_field(self, io_result, indices, field):
        batch_idx = io_result.size(0)
        io_result = io_result.view(batch_idx, self.pred_len, 2, field * field)
        io_x_re = torch.empty(batch_idx, self.pred_len, 2, self.num_nodes).to(self.device)
        indices = indices.unsqueeze(1).repeat_interleave(self.pred_len, dim=1).repeat_interleave(2, dim=2)
        io_x_re[:] = io_result.gather(-1, indices)
        return io_x_re

    def neural_diffusion(self, x, columns_to_observe, local_stamp, indices):
        step = 2
        od_x = x.squeeze_(dim=3)
        indices = indices[:, 0:1]
        io_x = self.io2od_flow(od_x, columns_to_observe)

        T = local_stamp[0, :]
        init_time = T[-1].item() + 1
        final_time = T[-1].item() + self.pred_len * step
        steps_val = final_time - init_time
        new_time_steps = torch.linspace(init_time, final_time, steps=int(steps_val) + 1)
        t = 0.01 * new_time_steps.float().flatten().float()

        field = self.construct_field(io_x[:, 0], indices, self.field_size)

        pde_rhs = lambda t, vs: self.diffusion(t, vs)
        io_result = odeint(pde_rhs, field, t, method=self.method, atol=1e-6, rtol=1e-3)
        io_result = io_result[0:len(io_result):step]
        io_result = io_result.transpose(0, 1)

        io_x_re = self.re_construct_field(io_result, indices, self.field_size)
        io_x += io_x_re
        return io_x

    def forward(self, input, predefined_A, local_poi, local_stamp, columns_to_observe, indices, idx=None):
        # ==================== Prepare Input Data ==================== #

        history_data, node_emb_od, time_in_day_emb, day_in_week_emb = self.st_emb(
            input, local_poi, columns_to_observe)
        predict = []
        # ========================= Construct Graphs ========================== #
        adp = self.gc(self.idx) if self.buildA_true else predefined_A
        # ========================= Spatio-Temporal Encoder ========================== #
        for i in range(self.layers):
            res = self.res0(F.dropout(input, self.dropout, training=self.training))
            x = history_data if i == 0 else history_data + torch.sigmoid(x)
            res += self.res1[i](x)

            gated_x, gated_x_res = self.estimation_gate[i](node_emb_od, time_in_day_emb,
                                                           day_in_week_emb, x)
            res += self.res2[i](gated_x)

            tc_x, tc_x_res = self.tc_enc[i](gated_x_res)
            # res += self.res3[i](tc_x)

            sc_x = self.sc_enc[i](tc_x_res, adp) if self.sc_true else self.residual_convs[i](tc_x_res)
            res += self.res4[i](sc_x)
            x = self.norm[i](sc_x, self.idx)

            predict.append(res)
        # ========================= Output Module ========================== #
        res = torch.cat(predict, dim=1)
        forecast_od = self.end_conv_2(F.relu(self.end_conv_1(F.relu(res))))
        # ========================= Diffusion Process ========================== #
        forecast_io = self.neural_diffusion(forecast_od, columns_to_observe, local_stamp, indices)

        return forecast_od, forecast_io

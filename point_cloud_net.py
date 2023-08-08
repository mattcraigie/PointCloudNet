import torch
import torch.nn as nn


def rotate(x, angle):
    assert len(angle) == 1 or len(angle) == 2, 'angle must be a 1D or 2D tensor'

    if len(angle) == 1:
        # 2D case
        rot_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                                      [torch.sin(angle), torch.cos(angle)]])

    else:
        # 3D case
        # is this right? it was written by github copilot
        rot_matrix = torch.tensor([[torch.cos(angle[0]), -torch.sin(angle[0]), 0],
                                        [torch.sin(angle[0]), torch.cos(angle[0]), 0],
                                        [0, 0, 1]])

        rot_matrix = torch.matmul(torch.tensor([[1, 0, 0],
                                                [0, torch.cos(angle[1]), -torch.sin(angle[1])],
                                                [0, torch.sin(angle[1]), torch.cos(angle[1])]]), rot_matrix)

    return torch.matmul(x, rot_matrix.to(x.device))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        if hidden_sizes is None:
            self.model = nn.Sequential(
                nn.Linear(input_size, output_size)
            )
        else:
            self.layers = []
            for i in range(len(hidden_sizes)):
                if i == 0:
                    self.layers.append(nn.Linear(input_size, hidden_sizes[i]))
                else:
                    self.layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
                self.layers.append(activation())

            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

            self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class PointEmbedder(nn.Module):
    def __init__(self, num_input_dims, embedding_size, hidden_sizes):
        super(PointEmbedder, self).__init__()

        self.embedder = MLP(num_input_dims, hidden_sizes, embedding_size, activation=nn.LeakyReLU)

    def forward(self, x, attention=None):
        # x is shape batch, num_points, num_input_dims
        if attention is None:
            point_embeddings = self.embedder(x)  # z is shape batch, num_points, embedding_size
            global_embedding = torch.mean(point_embeddings, dim=1)  # mean over points for input order invariance
        else:
            # attention is shape batch, num_points
            point_embeddings = self.embedder(x)
            global_embedding = torch.sum(point_embeddings * attention.unsqueeze(-1), dim=1)  # sum over points

        return global_embedding


class AnchorSelector(nn.Module):
    def __init__(self, num_input_dims, num_anchors, embedding_size=32, anchor_hidden_sizes=(64, 64),
                 embedding_hidden_sizes=(64, 64)):
        super(AnchorSelector, self).__init__()

        self.num_input_dims = num_input_dims
        self.num_anchors = num_anchors

        # add in the extra dimensions for rotation
        if num_input_dims == 2:
            self.num_output_dims = num_input_dims + 1
        elif num_input_dims == 3:
            self.num_output_dims = num_input_dims + 2
        else:
            raise ValueError('Only 2D and 3D point clouds supported')

        self.embedder = PointEmbedder(num_input_dims, embedding_size, embedding_hidden_sizes)
        self.anchor_selector = MLP(embedding_size, anchor_hidden_sizes, self.num_output_dims*self.num_anchors,
                                   activation=nn.LeakyReLU)

    def forward(self, x):
        x = self.embedder(x)
        x = self.anchor_selector(x)
        x = x.reshape(-1, self.num_anchors, self.num_output_dims)
        return x


class Pointy(nn.Module):
    # processes the
    def __init__(self, num_input_dims, num_anchors, scale_threshold, anchor_embedding_size=32, anchor_hidden_sizes=(32, 32,),
                 anchor_embedding_hidden_sizes=(32, 32), embedding_size=32, embedding_hidden_sizes=(32, 32)):
        super(Pointy, self).__init__()

        self.num_input_dims = num_input_dims
        self.num_anchors = num_anchors
        self.scale_threshold = scale_threshold
        self.embedding_size = embedding_size

        self.anchor_selector = AnchorSelector(num_input_dims, num_anchors, anchor_embedding_size, anchor_hidden_sizes,
                                              anchor_embedding_hidden_sizes)

        self.point_embedder = PointEmbedder(num_input_dims, embedding_size, embedding_hidden_sizes)

    def forward(self, x):
        # x is shape batch, num_points, num_input_dims
        anchors = self.anchor_selector(x)  # anchors are shape batch, num_anchors, num_input_dims
        deltas = x.unsqueeze(1) - anchors[:, :, :self.num_input_dims].unsqueeze(2)  # deltas are shape batch, num_anchors, num_points, num_input_dims

        # now, we begin working with different numbers of points per batch so we need to use a for loop

        all_anchor_embeddings = []
        for i in range(x.shape[0]):
            anchor_embeddings = []
            for j in range(self.num_anchors):
                anchor_deltas = deltas[i, j]  # shape num_points, num_input_dims

                # get the norm of each delta vector
                delta_norms = torch.norm(anchor_deltas, dim=1)

                # weight the points by their distance from the anchor
                # doing it this way is slow, but it allows a masking while retaining the gradients
                attention_weights = None # nn.functional.softmax(-delta_norms / self.scale_threshold, dim=0).unsqueeze(0)

                # rotate the remaining points according to the angle of the anchor
                anchor_angle = anchors[i, j, self.num_input_dims:]
                anchor_rotated_deltas = rotate(anchor_deltas, anchor_angle)

                # embed the points
                anchor_embedding = self.point_embedder(anchor_rotated_deltas.unsqueeze(0), attention_weights)  # shape 1, embedding_size
                anchor_embeddings.append(anchor_embedding)
            anchor_embeddings = torch.cat(anchor_embeddings, dim=0)  # shape num_anchors, embedding_size
            all_anchor_embeddings.append(anchor_embeddings)

        all_anchor_embeddings = torch.stack(all_anchor_embeddings)  # shape batch, num_anchors, embedding_size
        all_anchor_embeddings = all_anchor_embeddings.mean(1)  # mean along anchors

        return all_anchor_embeddings

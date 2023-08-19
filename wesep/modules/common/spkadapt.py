import torch
import torch.nn as nn

from wesep.modules.common import FiLM


class SpeakerTransform(nn.Module):
    def __init__(self, embed_dim=256, num_layers=3, hid_dim=128):
        """
        Transform the pretrained speaker embeddings, keep the dimension the same
        :param embed_dim:
        :param num_layers:
        :param hid_dim:
        :return:
        """
        super(SpeakerTransform, self).__init__()
        self.transforms = []
        self.transforms.append(nn.Conv1d(embed_dim, hid_dim, 1))
        for _ in range(num_layers - 2):
            self.transforms.append(
                nn.Conv1d(hid_dim, hid_dim, 1)
            )
            self.transforms.append(
                nn.Tanh()
            )
        self.transforms.append(nn.Conv1d(hid_dim, embed_dim, 1))
        self.transforms = nn.Sequential(*self.transforms)

    def forward(self, x):
        return self.transforms(x)


class SpeakerFuseLayer(nn.Module):
    def __init__(self, embed_dim=256, feat_dim=512, fuse_type='concat'):
        super(SpeakerFuseLayer, self).__init__()
        assert fuse_type in ["concat", "additive", "FiLM", "None"]

        self.fuse_type = fuse_type
        if fuse_type == 'concat':
            self.fc = nn.Linear(embed_dim + feat_dim, feat_dim)
        elif fuse_type == 'additive':
            self.fc = nn.Linear(embed_dim, feat_dim)
        elif fuse_type == 'FiLM':
            self.film = FiLM(feat_dim, embed_dim)
        else:
            raise ValueError("Fuse type not defined.")

    def forward(self, x, embed):
        """

        :param x: batch x dimension x length
        :param embed: batch x dimension x 1
        :return:
        """
        if self.fuse_type == "concat":
            # For Conv
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                y = torch.cat([x, embed_t], 1)
                y = torch.transpose(y, 1, 2)
                x = torch.transpose(self.fc(y), 1, 2)
            else:
                # len(x.size() == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                y = torch.cat([x, embed_t], 2)
                y = torch.transpose(y, 2, 3)
                x = torch.transpose(self.fc(y), 2, 3).contiguous()
                # print(x.size())
        elif self.fuse_type == "additive":
            if len(x.size()) == 3:
                embed_t = embed.expand(-1, -1, x.size(2))
                embed_t = torch.transpose(embed_t, 1, 2)
                x = x + torch.transpose(self.fc(embed_t), 1, 2)
            else:
                # len(x.size() == 4
                embed_t = embed.expand(-1, x.size(1), -1, x.size(3))
                embed_t = torch.transpose(embed_t, 2, 3)
                x = x + torch.transpose(self.fc(embed_t), 2, 3)
        else:
            embed = embed.squeeze(-1)
            x = self.film(embed, x)
        return x


def test_speaker_fuse():
    st = SpeakerTransform(embed_dim=256, num_layers=3, hid_dim=128)
    sfl = SpeakerFuseLayer(fuse_type='additive')

    embeds = torch.rand(4, 256, 1)
    encoder_output = torch.rand(4, 512, 1000)

    print(embeds.size())
    embeds = st(embeds)
    print(embeds.size())
    output = sfl(encoder_output, embeds)
    print(output.size())


if __name__ == "__main__":
    test_speaker_fuse()

import torch
import numpy as np


def area(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def triangle_area(A, B, C):
    out = (C - A).flip([-1]) * (B - A)
    out = out[..., 1] - out[..., 0]
    return out


def xing_loss(x_list, scale=1.0):  # x[ npoints,2]
    loss = 0.
    for x in x_list:
        x1 = torch.cat([x[1:, :], x[:1, :]], dim=0)
        segments = torch.cat([x.unsqueeze(dim=-1), x1.unsqueeze(dim=-1)],
                             dim=-1)  # [npoints, 2, 2], npoints, xy, start-end
        mutual_segments = segments.unsqueeze(dim=1).expand(-1, x.shape[0], -1, -1)  # [npoints,npoints,2,2]
        mutual_segments_2 = torch.transpose(mutual_segments, 0, 1)
        mutual_segments = torch.cat([mutual_segments, mutual_segments_2], dim=-1)  # [npoints,npoints,2,4] 4 is A,B,C,D
        Area_AB_C = triangle_area(mutual_segments[:, :, :, 0], mutual_segments[:, :, :, 1], mutual_segments[:, :, :, 2])
        Area_AB_D = triangle_area(mutual_segments[:, :, :, 0], mutual_segments[:, :, :, 1], mutual_segments[:, :, :, 3])
        Area_CD_A = triangle_area(mutual_segments[:, :, :, 2], mutual_segments[:, :, :, 3], mutual_segments[:, :, :, 0])
        Area_CD_B = triangle_area(mutual_segments[:, :, :, 2], mutual_segments[:, :, :, 3], mutual_segments[:, :, :, 1])

        condition1 = ((Area_AB_C * Area_AB_D) <= 0.).float()
        condition2 = ((Area_CD_A * Area_CD_B) <= 0.).float()

        four_areas = torch.cat([
            abs(Area_AB_C.unsqueeze(dim=-1)), abs(Area_AB_D.unsqueeze(dim=-1)),
            abs(Area_CD_A.unsqueeze(dim=-1)), abs(Area_CD_B.unsqueeze(dim=-1))
        ], dim=-1)
        # areas can describe the distance bwtween one point and a segment.
        four_areas = four_areas.min(dim=-1, keepdim=False)[0]
        four_areas = torch.relu(-torch.log(four_areas + 1e-5))

        # Tensor_X = Area_AB_C*Area_AB_D
        # Tensor_Y = Area_CD_A*Area_CD_B
        # angel = torch.atan2(Tensor_X + 1e-5,Tensor_Y+ 1e-5)
        # mask2 = torch.tanh(angel+1.5708)
        # mask2 = torch.relu(-mask2)
        # mask2 = torch.triu(mask2, diagonal=2)
        # print(mask2)

        mask = condition1 * condition2  # mask is without gradient.
        area_AB_1 = (abs(Area_AB_C)) / (abs(Area_AB_D) + 1e-5)
        area_AB_2 = (abs(Area_AB_D)) / (abs(Area_AB_C) + 1e-5)
        area_AB, _ = torch.cat([area_AB_1.unsqueeze(dim=-1), area_AB_2.unsqueeze(dim=-1)], dim=-1).min(dim=-1)
        area_AB = torch.clip(area_AB, 0.0, 1.0)
        area_AB = torch.nan_to_num(area_AB, nan=0.0)

        area_CD_1 = (abs(Area_CD_A)) / (abs(Area_CD_B) + 1e-5)
        area_CD_2 = (abs(Area_CD_B)) / (abs(Area_CD_A) + 1e-5)
        area_CD, _ = torch.cat([area_CD_1.unsqueeze(dim=-1), area_CD_2.unsqueeze(dim=-1)], dim=-1).min(dim=-1)
        area_CD = torch.clip(area_CD, 0.0, 1.0)
        area_CD = torch.nan_to_num(area_CD, nan=0.0)

        area_loss, _ = torch.cat([area_AB.unsqueeze(dim=-1), area_CD.unsqueeze(dim=-1)], dim=-1).min(dim=-1)

        area_loss = (area_loss + four_areas) * mask
        # print(f"mask is: {mask}")
        # print(f"area_loss is: {area_loss}")
        area_loss = area_loss.sum() / ((x.shape[0] - 2) ** 2)

        loss += area_loss * scale

    return loss / (len(x_list))


if __name__ == "__main__":
    x = torch.rand([6, 2])
    scale = 0.5
    y = xing_loss([x], scale)
    print(y)

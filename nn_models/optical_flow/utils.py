import torch


def warp_optical_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    if str(img.device) != str(flow.device):
        raise RuntimeError('Both image anf flow device must be the same.')
    device = img.device

    if img.shape[0] != flow.shape[0]:
        raise RuntimeError('Two input batches has different batch sizes.')
    if img.shape[-2] != flow.shape[-2]:
        raise RuntimeError('Two input batches has different heights.')
    if img.shape[-1] != flow.shape[-1]:
        raise RuntimeError('Two input batches has different widths.')

    h, w = img.shape[-2:]

    grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
    grid = torch.stack((grid_x, grid_y), dim=-1).float().to(device)
    grid = grid.unsqueeze(dim=0).repeat(img.shape[0], 1, 1, 1)
    flow = flow.permute(0, 2, 3, 1)
    warped_grid = grid - flow
    warped_grid = warped_grid * 2 / torch.tensor([w, h]).float().to(device) - 1
    warped_image = torch.nn.functional.grid_sample(img.float(), warped_grid, padding_mode='zeros',
                                                   align_corners=True)
    return warped_image

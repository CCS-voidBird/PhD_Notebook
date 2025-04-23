import torch

## Externally convert chromosome number and position to 2D discrete coordinates, particular for large position values
def chr_pos_scaling(map_info, d_model, method='logical',base = 1000, eps = 1e-6):
    # pos: a vector of position values from map file
    # chr_num: a vector of chromosome numbers from map file
    # return: 2D discrete coordinates, large position values would be scaled down and converted to discrete values (integer)
    # method: 'same' or 'different', 'same' mode will scale pos into percentage of the maximum position value in the same chromosome, and then convert to discrete values;
    # 'different' mode will scale pos via log transformation and then convert to discrete values
    # chromosome number and position in first two columns of map file

    pos = map_info[:, 1]
    chr_num = map_info[:, 0]

    if method == 'logical':
        # in this mode, chromosome number is same as 1, and position is a vector of logical values from 1 to n
        chr_num = torch.ones_like(chr_num)
        pos = range(1, len(pos) + 1)

    max_pos = torch.max(pos)
    max_chr = torch.max(chr_num)
    
    # scale down the position values to avoid large values
    if method == 'same':
        pos = pos / max_pos
    elif method == 'different':
        pos = torch.log(pos + eps) / torch.log(base)
    elif method == 'logical':
        pass

    chr_num = chr_num / max_chr

    map_info[:, 0] = chr_num
    map_info[:, 1] = pos

    return map_info


    


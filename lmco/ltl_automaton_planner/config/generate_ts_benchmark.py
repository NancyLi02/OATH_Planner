import yaml
import numpy as np
import sys

data = {
    'state_dim': ["2d_pose_region", "robot_state"],
    'state_models': {
        '2d_pose_region': {
            'ts_type': "2d_pose_region",
            'initial': "c0_r0",
            'nodes': {
            }
        }
    },
    'actions': {
    }
}

N = 10

# region = {
#     (0, 5, 1): set([]), (1, 5, 1): set([]), (2, 5, 1): set(['p4']), (3, 5, 1): set(['l1']), (4, 5, 1): set([]), (5, 5, 1): set([]),
#     (0, 4, 1): set([]), (1, 4, 1): set(['i']), (2, 4, 1): set([]), (3, 4, 1): set([]), (4, 4, 1): set([]), (5, 4, 1): set([]),
#     (0, 3, 1): set(['p3']), (1, 3, 1): set([]), (2, 3, 1): set([]), (3, 3, 1): set([]), (4, 3, 1): set(['f']), (5, 3, 1): set([]),
#     (0, 2, 1): set([]), (1, 2, 1): set([]), (2, 2, 1): set(['h']), (3, 2, 1): set([]), (4, 2, 1): set([]), (5, 2, 1): set(['p2']),
#     (0, 1, 1): set(['c', 'unloaded']), (1, 1, 1): set([]), (2, 1, 1): set([]), (3, 1, 1): set([]), (4, 1, 1): set([]), (5, 1, 1): set([]),
#     (0, 0, 1): set([]), (1, 0, 1): set([]), (2, 0, 1): set([]), (3, 0, 1): set(['p1']), (4, 0, 1): set([]), (5, 0, 1): set(['l2']),
# }

region = dict()
coords = np.linspace(0,N-1,N)
coords = coords.astype(int)

for i in np.flip(coords):
    for j in coords:
        region_label = set([])
        center = int(np.rint(0.5*N))-1
        coord_key = (j,i)
        
        # if coord_key == (0,0):
        #     region_label.add('a')
        # elif coord_key ==  (N-1,N-1):
        #     region_label.add('c')
        # elif coord_key ==  (N-1,0):
        #     region_label.add('b')
        # elif coord_key ==  (0,N-1):
        #     region_label.add('d')
        if coord_key == (center,center):
            region_label.add('e')
        elif coord_key == (center+1,center+1):
            region_label.add('g')
        elif coord_key == (center,center+1):
            region_label.add('h')
        elif coord_key == (center+1,center):
            region_label.add('f')
        
        if coord_key == (center-1, center) or coord_key == (center-1, center+1) or coord_key == (center, center-1) or coord_key == (center+1, center-1) or\
            coord_key == (center+2, center) or coord_key == (center+2, center+1) or coord_key == (center+1, center+2) or coord_key == (center, center+2):
             region_label.add('p')
        region[(j,i,1)] = region_label
        
data['state_models']['2d_pose_region']['initial'] = f'c{center}_r{center}'
neighbor_transforms = [(-1, 0), (0, -1), (1, 0), (0, 1)]
# size_x = 6
# size_y = 6
# y
# ^
# |_> x
count = 0
edges = []
for i in region.keys():
    x = count % N
    y = (N-1) - int(count/N)
    count = count + 1
    cell_key = f'c{x}_r{y}'
    position = [x*10+5, y*10+5]
    data['state_models']['2d_pose_region']['nodes'][cell_key] = {
            'attr': {
                # 'type': 'square',
                'pose': '[' + str(position)+', [0]]',
                # 'length': 10,
                # 'hysteresis': 0.05,
                'labels': list(region[i]),
            },
            'connected_to': {}
        }
    # if region[i]
    #     data['state_models']['2d_pose_region']['nodes'][cell_key]['labels'] = 
    data['state_models']['2d_pose_region']['nodes'][cell_key]['connected_to'][cell_key] = "stay"
    for transform in neighbor_transforms:
        new_x = i[0] + transform[0]
        new_y = i[1] + transform[1]
        if ((0 <= new_x) and (new_x < N) and 
            (0 <= new_y) and (new_y < N)):
            data['state_models']['2d_pose_region']['nodes'][cell_key]['connected_to'][f'c{new_x}_r{new_y}'] \
                = f'goto_c{new_x}_r{new_y}'
            data['actions'][f'goto_c{new_x}_r{new_y}'] = {
                'type': "move",
                'weight': 10,
                'guard': "1",
                'attr': {
                    # 'region': "c8",
                    'pose':  '[' + str([x*10+5, y*10+5, 0]) + ', [0, 0, 0, 1]]'
                }
            }
        
    


data['state_models']['robot_state'] = {
    'ts_type': "robot_state",
    'initial': "loaded",
    'nodes': {
        # 'unloaded': {
        # 'attr':{
        #   'labels': ['unloaded']
        # },
        # 'connected_to': {
        #     'loaded': "load"
        # }
        # },
        'loaded': {
        'attr':{
          'labels': ['loaded']
        },
        'connected_to': {
            'loaded': "stay"
        }
        }
    }
}

data['actions']['stay'] = {
    'type': "stay",
    'guard': "1",
    'weight': 0.1,
}
# data['actions']['load'] = {
#     'type': 'load',
#     'guard': "a || b || c || d",
#     'weight': 5,
# }
# data['actions']['unload'] = {
#     'type': 'unload',
#     'guard': "e || f || g || h",
#     'weight': 5,
# }

## No Barriers

data['state_models']['2d_pose_region']
# with open('your_file.yaml', 'w') as file:
#     yaml.dump(data, file, default_flow_style=False)


## Known Barriers (Black lines)

data_known = data

# Line Coordinates
# ci_rj
# c1_start = int(np.rint(0.2*N))
# c1_end = int(np.rint(c1_start+0.6*N-1))
# dc1 = int(np.rint(0.6*N))
c1_start = 1
c1_end = N-2
dc1 = N-2
c1 = np.linspace(c1_start,c1_end,dc1).astype(int)
c2 = int(np.rint(0.5*N)-1)
print(f'{c1_start},{c1_end},{c2}')
print(c1)

for i in c1:
    cell_key_h1 = f'c{i}_r{c2}'
    cell_key_h2 = f'c{i}_r{c2+1}'
    del data_known['state_models']['2d_pose_region']['nodes'][cell_key_h1]['connected_to'][cell_key_h2]
    del data_known['state_models']['2d_pose_region']['nodes'][cell_key_h2]['connected_to'][cell_key_h1]

    cell_key_v1 = f'c{c2}_r{i}'
    cell_key_v2 = f'c{c2+1}_r{i}'
    del data_known['state_models']['2d_pose_region']['nodes'][cell_key_v1]['connected_to'][cell_key_v2]
    del data_known['state_models']['2d_pose_region']['nodes'][cell_key_v2]['connected_to'][cell_key_v1]


with open('benchmark_known_'+str(N)+'.yaml', 'w') as file:
    yaml.dump(data_known, file, default_flow_style=False)




## Blocks (Red Lines)

# block_start = [(c2+1,c1_start),(c1_end,c2+1),(c2,c1_end),(c1_start,c2)]
# block_end = [(c1_end,c1_start),(c1_end,c1_end),(c1_start,c1_end),(c1_start,c1_start)]
block_start = [(c1_start+1,c1_start),(c1_end,c1_start+1),(c1_start+1,c1_end),(c1_start,c1_start+1)]
block_end = [(c1_end-1,c1_start),(c1_end,c1_end-1),(c1_end-1,c1_end),(c1_start,c1_end-1)]
block_direction = [-1,2,1,-2] # 1 = horizontal, 2 = vertical
                              # Sign is block direction

data_blocks = {
    'blocks': {
    }
}


count = 0
pairs = []
for k in block_direction:
    c_start = block_start[count]
    c_end = block_end[count]
    col = []
    row = []
    direc = int(k/np.abs(k))

    # block_str = f'c{c_start[0]}_r{c_start[1]}_to_c{c_end[0]}_r{c_end[1]}'
    # data_blocks['blocks'][block_str] = {}

    if np.abs(k) == 1:
        dc = np.abs(c_end[0] - c_start[0]) + 1
        col = np.linspace(c_start[0],c_end[0],dc).astype(int)
        row = (c_start[1]*np.ones(dc)).astype(int)
        row_trans = row + direc

        ind = 0
        for i in col:
            # block_coords = [f'({int(i)},{int(row_trans[ind])})',f'({int(i)},{int(row[ind])})']
            # data_blocks['blocks'][block_str][f'c{i}_r{row[ind]}'] = block_coords
            pairs.append([[int(i),int(row_trans[ind])], [int(i),int(row[ind])]])
            ind += 1

    elif np.abs(k) == 2:
        dc = np.abs(c_end[1] - c_start[1]) + 1
        row = np.linspace(c_start[1],c_end[1],dc).astype(int)
        col = (c_start[0]*np.ones(dc)).astype(int)
        col_trans = col + direc

        ind = 0
        for i in row:
            # block_coords = [f'({int(col_trans[ind])},{int(i)})',f'({int(col[ind])},{int(i)})']
            # data_blocks['blocks'][block_str][f'c{col[ind]}_r{i}'] = block_coords
            pairs.append([[int(col_trans[ind]),int(i)], [int(col[ind]),int(i)]])
            ind += 1

    else:
        print(f'Invalid block direction. Incorrect value at ind = {count}')
        break

    count += 1

data_blocks['blocks'] = pairs

with open('benchmark_block_'+str(N)+'.yaml', 'w') as file:
    yaml.dump(data_blocks, file, default_flow_style=False)



## Bumps (Pink Lines)

bump_start = [(c1_start,c1_start),(c1_start,c1_start),(c1_start,c1_end),(c1_start,c1_end), (c1_end,c1_start),(c1_end+1,c1_start),(c1_end,c1_end+1),(c1_end+1,c1_end)]
bump_end = [(c1_start-1,c1_start),(c1_start,c1_start-1),(c1_start,c1_end+1),(c1_start-1,c1_end), (c1_end,c1_start-1),(c1_end,c1_start),(c1_end,c1_end),(c1_end,c1_end)]
bump_direction = [-1,-2,-2,1,2,-1,2,1] # 1 = horizontal, 2 = vertical
                              # Sign is bump direction

data_bumps = {
    'bumps': {
    }
}

count = 0
pairs = []
for k in range(8):
    # c_start = bump_start[count]
    # c_end = bump_end[count]
    # col = []
    # row = []
    # direc = int(k/np.abs(k))

    # # bump_str = f'c{c_start[0]}_r{c_start[1]}_to_c{c_end[0]}_r{c_end[1]}'
    # # data_bumps['bumps'][bump_str] = {'weight': 30}
    # data_bumps['bumps'] = {'weight': 50,
    #                        'points': {}}

    # if np.abs(k) == 1:
    #     dc = np.abs(c_end[0] - c_start[0]) 
    #     col = np.linspace(c_start[0],c_end[0],dc).astype(int)
    #     row = (c_start[1]*np.ones(dc)).astype(int)
    #     row_trans = row + direc
    #     print(row_trans)
        

    #     ind = 0
    #     for i in col:
    #         # bump_coords = [f'({int(i)},{int(row_trans[ind])})',f'({int(i)},{int(row[ind])})']
    #         # data_bumps['bumps'][bump_str][f'c{i}_r{row[ind]}'] = bump_coords
    #         pairs.append([[int(i),int(row_trans[ind])], [int(i),int(row[ind])]])
    #         ind += 1
        

    # elif np.abs(k) == 2:
    #     dc = np.abs(c_end[1] - c_start[1]) 
    #     row = np.linspace(c_start[1],c_end[1],dc).astype(int)
    #     col = (c_start[0]*np.ones(dc)).astype(int)
    #     col_trans = col + direc

    #     ind = 0
    #     for i in row:
    #         # bump_coords = [f'({int(col_trans[ind])},{int(i)})',f'({int(col[ind])},{int(i)})']
    #         # data_bumps['bumps'][bump_str][f'c{col[ind]}_r{i}'] = bump_coords
    #         pairs.append([[int(col_trans[ind]),int(i)], [int(col[ind]),int(i)]])
    #         ind += 1

    # else:
    #     print(f'Invalid bump direction. Incorrect value at ind = {count}')
    #     break

    # count += 1
    pairs.append([list(bump_start[k]), list(bump_end[k])])
data_bumps['bumps']['points'] = pairs

with open('benchmark_bump_'+str(N)+'.yaml', 'w') as file:
    yaml.dump(data_bumps, file, default_flow_style=False)
    
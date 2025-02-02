import yaml

data = {
    'state_dim': ["2d_pose_region", "Drone_state"],
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

region = {
    (0, 5, 1): set([]), (1, 5, 1): set([]), (2, 5, 1): set(['p4']), (3, 5, 1): set(['l1']), (4, 5, 1): set([]), (5, 5, 1): set([]),
    (0, 4, 1): set([]), (1, 4, 1): set(['i']), (2, 4, 1): set([]), (3, 4, 1): set([]), (4, 4, 1): set([]), (5, 4, 1): set([]),
    (0, 3, 1): set(['p3']), (1, 3, 1): set([]), (2, 3, 1): set([]), (3, 3, 1): set([]), (4, 3, 1): set(['f']), (5, 3, 1): set([]),
    (0, 2, 1): set([]), (1, 2, 1): set([]), (2, 2, 1): set(['h']), (3, 2, 1): set([]), (4, 2, 1): set([]), (5, 2, 1): set(['p2']),
    (0, 1, 1): set(['c', 'unloaded']), (1, 1, 1): set([]), (2, 1, 1): set([]), (3, 1, 1): set([]), (4, 1, 1): set([]), (5, 1, 1): set([]),
    (0, 0, 1): set([]), (1, 0, 1): set([]), (2, 0, 1): set([]), (3, 0, 1): set(['p1']), (4, 0, 1): set([]), (5, 0, 1): set(['l2']),
}

neighbor_transforms = [(-1, 0), (0, -1), (1, 0), (0, 1)]
size_x = 6
size_y = 6
# y
# ^
# |_> x
count = 0
edges = []
for i in region.keys():
    x = count % size_x
    y = 5 - int(count/size_x)
    count = count + 1
    cell_key = f'c{x}_r{y}'
    position = [x*10+5, y*10+5]
    data['state_models']['2d_pose_region']['nodes'][cell_key] = {
            'attr': {
                'type': 'square',
                'pose': '[' + str(position)+', [0]]',
                'length': 10,
                'hysteresis': 0.05,
                'labels': str(list(region[i])),
            },
            'connected_to': {}
        }
    # if region[i]
    #     data['state_models']['2d_pose_region']['nodes'][cell_key]['labels'] = 
    #data['state_models']['2d_pose_region']['nodes'][cell_key]['connected_to'][cell_key] = "stay"
    for transform in neighbor_transforms:
        new_x = i[0] + transform[0]
        new_y = i[1] + transform[1]
        if ((0 <= new_x) and (new_x < size_x) and 
            (0 <= new_y) and (new_y < size_y)):
            data['state_models']['2d_pose_region']['nodes'][cell_key]['connected_to'][f'c{new_x}_r{new_y}'] \
                = f'goto_c{new_x}_r{new_y}'
            data['actions'][f'goto_c{new_x}_r{new_y}'] = {
                'type': "move",
                'weight': 10,
                'guard': "1",
                'attr': {
                    'region': "c8",
                    'pose':  '[' + str([x*10+5, y*10+5, 0]) + ', [0, 0, 0, 1]]'
                }
            }
        
    


data['state_models']['Drone_state'] = {
    'ts_type': "Drone_state",
    'initial': "unloaded",
    'nodes': {
        'unloaded': {
        'connected_to': {
            'loaded': "load"
        }
        },
        'loaded': {
        'connected_to': {
            'unloaded': "unload"
        }
        }
    }
}

# data['actions']['stay'] = {
#     'type': "stay",
#     'guard': "1",
#     'weight': 0,
#     'synchronized_transition': {
#         'type': "synchronized_transition",
#         'guard': "1",
#         'weight': 0,
#     }
# }
data['actions']['load'] = {
    'type': 'load',
    'guard': "1",
    'weight': 0,
}
data['actions']['unload'] = {
    'type': 'unload',
    'guard': "1",
    'weight': 0,
}


data['state_models']['2d_pose_region']
with open('your_file.yaml', 'w') as file:
    yaml.dump(data, file, default_flow_style=False)
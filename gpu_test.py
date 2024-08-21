# # import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))

import sys

sys_path=[ p for p in sys.path if not p.startswith('/home')]
print(sys_path)
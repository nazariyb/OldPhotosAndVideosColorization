from torch.cuda import is_available

device = 'cuda' if is_available() else 'cpu'
# device = 'cpu'

print('Using', device)

# content-aware-model

## 实现中的一些记录

```python

real_layout_size = [225, 300]
network_input_layout_size = [45, 60]
one_block_size = [5, 5]

layout = [
  {
    'label': 'text',
    'bb': [151, 26, 205, 280],
    'area': 13716
  },
]

x1, y1, x2, y2 = layout[0]['bb']  # x1=151, y1=26, x2=205, y2=280


```
from psd_tools import PSDImage
from PIL import Image, ImageDraw, ImageFont 
import os

psd_root = './psd'
res_root = './psd/res'
file_list = os.listdir(psd_root)
font_size = 40
font = ImageFont.truetype('./fonts/wqy-microhei.ttc', font_size)

if not os.path.exists('./psd'):
    os.mkdir('./psd')
if not os.path.exists('./psd/res'):
    os.mkdir('./psd/res')

for f in file_list:
    if f.split('.')[-1] != 'psd':
        break

    fn = f[:-4]
    os.mkdir(f'./psd/res/{fn}')
    psdAbsPath = os.path.abspath(f'{psd_root}/{f}')
    pngAbsPath = os.path.abspath(f'{res_root}/{fn}/{fn}.png')
    
    print(f'Dealing with file {psdAbsPath}')
    psd = PSDImage.open(psdAbsPath)
    psd.composite().save(pngAbsPath)
    canvas_size = psd.size
    print('canvas_size', canvas_size)

    im = Image.open(pngAbsPath)
    layout_img = Image.new('RGB', (canvas_size[0], canvas_size[1]), (255, 255, 255))
    
    draw = ImageDraw.Draw(im)
    layout_draw = ImageDraw.Draw(layout_img)

    for layer in psd:
        print(layer)
        layer_name = str(layer.name)
        layer_name = layer_name.replace('"', '')
        layer_name = layer_name.replace(' ', '')
        layer_name = layer_name.replace('.', '')
        layer_name = layer_name.replace('png', '')
        layer.composite().save(f'{res_root}/{fn}/{fn}_{layer_name}.png')
        
        bbox = list(layer.bbox)
        print(bbox)
        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0
        
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='red', width=10)
        layout_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline='red', width=10)

        label_size = (draw.textlength(layer_name, font), font_size)
        
        draw.rectangle(
            [ bbox[0], bbox[1], bbox[0] + 5 + label_size[0], bbox[1] + 5 + label_size[1] ],
            fill = 'yellow',
            width = 200
        )

        draw.text((bbox[0], bbox[1]), layer_name, fill='red', font=font)
        
    im.save(f'{res_root}/{fn}/{fn}_layout_label.png')
    layout_img.save(f'{res_root}/{fn}/{fn}_layout.png')



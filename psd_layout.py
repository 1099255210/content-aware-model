from psd_tools import PSDImage
import os
from PIL import Image, ImageDraw, ImageFont 

psd_root = './psd'
img_root = './psd/res'
file_list = os.listdir(psd_root)
font = ImageFont.truetype(r"C:\Windows\Fonts\simsun.ttc",40)

for f in file_list:
    # f = r'饮品店%暴打椰子柠檬%尺寸（30mm_100mm）%现代风格2.psd'
    if f.split('.')[-1] != 'psd':
        break

    print(f'Dealing with file {f}')
    psd_name = os.path.join(psd_root, f)
    img_name = os.path.join(img_root, f.replace('psd', 'png'))
    print(img_name)
    psd = PSDImage.open(psd_name)
    psd.compose().save(img_name)
    canvas_size = psd.size
    print('canvas_size',canvas_size)

    im = Image.open(img_name)
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
        layer.compose().save(os.path.join(img_root,str(layer_name)+'.png'))
        
        bbox = list(layer.bbox)
        print(bbox)
        for i in range(len(bbox)):
            if bbox[i] < 0:
                bbox[i] = 0
        
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]],outline='red',width=10)
        layout_draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]],outline='red',width=10)

        label_size = draw.textsize(layer_name, font)
        
        draw.rectangle(
            [bbox[0], bbox[1], bbox[0]+5+label_size[0], bbox[1]+5+label_size[1]],
            fill = 'yellow',
            width = 200
        )

        draw.text((bbox[0], bbox[1]), layer_name, fill = 'red', font=font)

    im.save(os.path.join(img_root, "new_result.png"))
    layout_img.save(os.path.join(img_root, 'layout.jpg'))

    break



from rembg import new_session, remove
from PIL import Image

input_path = './boulot-dams.jpeg'
output_path = 'output.png'


input = Image.open(input_path)
output = remove(input, only_mask=True)
mask = Image.eval(output, lambda x: 255 - x)
mask.save(output_path)
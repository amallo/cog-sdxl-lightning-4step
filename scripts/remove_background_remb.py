from rembg import new_session, remove
from PIL import Image

input_path = './data/inputs/mairie-pacs.jpeg'
output_path = 'output.png'


input = Image.open(input_path).convert("RGBA")

output = remove(input)
output.save(output_path)
background = Image.new("RGBA", input.size, (255, 255, 255, 255))
result = Image.alpha_composite(background, input)
result = result.convert("RGB")
#mask = Image.eval(output, lambda x: 255 - x)

# proof of concept python script
from PIL import Image

width, height = 1920, 1080

real_min, real_max = -2.0, 1.0
imag_min, imag_max = -1.5, 1.5

real_inc = (real_max - real_min) / width
imag_inc = (imag_max - imag_min) / height

max_iterations = 100

def mandelbrot(c):
    z = 0
    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

image = Image.new('RGB', (width, height))

real = real_min

for px in range(width):
    imag = imag_min
    for py in range(height):
        c = complex(real, imag)
        m = mandelbrot(c)
        color = 255 - int(m * 255 / max_iterations)
        image.putpixel((px, py), (color, color, color))
        imag += imag_inc
    real += real_inc
    
image.save('mandel-py.png')

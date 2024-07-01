# proof of concept python script
from PIL import Image

width, height = 1920, 1080

real_min, real_max = -2.0, 1.0
imag_min, imag_max = -1.5, 1.5

max_iterations = 1000

def mandelbrot(c):
    z = 0
    for n in range(max_iterations):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iterations

image = Image.new('RGB', (width, height))

for px in range(width):
    for py in range(height):
        a = real_min + (px / (width - 1)) * (real_max - real_min)
        b = imag_min + (py / (height - 1)) * (imag_max - imag_min)
        c = complex(a, b)
        m = mandelbrot(c)
        color = 255 - int(m * 255 / max_iterations)
        image.putpixel((px, py), (color, color, color))
image.save('mandelbrot.png')

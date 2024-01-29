from PIL import Image, ImageDraw, ImageFont
import random


def generate_img():
    image_size = (20, 20)
    num_samples_per_digit = 10
    digits = range(10)

    font = ImageFont.load_default()

    for digit in digits:
        for sample in range(num_samples_per_digit):
            image = Image.new('L', image_size, 'white')
            draw = ImageDraw.Draw(image)

            x, y = random.randint(0, 5), random.randint(0, 5)

            angle = random.randint(-15, 15)
            draw.text((x, y), str(digit), font=font, fill='black')
            image = image.rotate(angle)

            filename = f'test_digit_{digit}_sample_{sample}.png'
            image.save(filename)

    print("Images have been generated.")

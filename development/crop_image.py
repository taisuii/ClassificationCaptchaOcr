from PIL import Image, ImageFont, ImageDraw, ImageOps
from io import BytesIO


def convert_png_to_jpg(png_bytes: bytes) -> bytes:
    # 将传入的 bytes 转换为图像对象
    png_image = Image.open(BytesIO(png_bytes))

    # 创建一个 BytesIO 对象，用于存储输出的 JPG 数据
    output_bytes = BytesIO()

    # 检查图像是否具有透明度通道 (RGBA)
    if png_image.mode == 'RGBA':
        # 创建白色背景
        white_bg = Image.new("RGB", png_image.size, (255, 255, 255))
        # 将 PNG 图像粘贴到白色背景上，透明部分用白色填充
        white_bg.paste(png_image, (0, 0), png_image)
        jpg_image = white_bg
    else:
        # 如果图像没有透明度，直接转换为 RGB 模式
        jpg_image = png_image.convert("RGB")

    # 将转换后的图像保存为 JPG 格式到 BytesIO 对象
    jpg_image.save(output_bytes, format="JPEG")

    # 返回保存后的 JPG 图像的 bytes
    return output_bytes.getvalue()


def crop_image(image_bytes, coordinates):
    img = Image.open(BytesIO(image_bytes))
    width, height = img.size
    grid_width = width // 3
    grid_height = height // 3
    cropped_images = []
    for coord in coordinates:
        y, x = coord
        left = (x - 1) * grid_width
        upper = (y - 1) * grid_height
        right = left + grid_width
        lower = upper + grid_height
        box = (left, upper, right, lower)
        cropped_img = img.crop(box)
        cropped_images.append(cropped_img)
    return cropped_images



if __name__ == "__main__":
    # 切割顺序，这里是从左到右，从上到下[x,y]
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]
    with open("./image_test/bg.jpg", "rb") as rb:
        bg_img = rb.read()
    cropped_images = crop_image(bg_img, coordinates)
    # 一个个保存下来
    for j, img_crop in enumerate(cropped_images):
        img_crop.save(f"./image_test/bg{j}.jpg")
    
    # 图标格式转换
    with open("./image_test/icon.png", "rb") as rb:
        icon_img = rb.read()
    icon_img_jpg = convert_png_to_jpg(icon_img)
    with open("./image_test/icon.jpg", "wb") as wb:
        wb.write(icon_img_jpg)

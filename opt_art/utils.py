from PIL import Image


def join_images(*rows, bg_color=(0, 0, 0, 0), alignment=(0.0, 0.0)):
    rows = [[image.convert("RGBA") for image in row] for row in rows]

    heights = [max(image.height for image in row) for row in rows]

    widths = [max(image.width for image in column) for column in zip(*rows)]

    tmp = Image.new("RGBA", size=(sum(widths), sum(heights)), color=bg_color)

    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            y = sum(heights[:i]) + int((heights[i] - image.height) * alignment[1])
            x = sum(widths[:j]) + int((widths[j] - image.width) * alignment[0])
            tmp.paste(image, (x, y))

    return tmp

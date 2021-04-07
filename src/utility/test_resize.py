from PIL import Image
import requests
import os


def test_function():
    size = 300, 300
    link = "https://steamcdn-a.akamaihd.net/steam/apps/10/0000002538.600x338.jpg?t=1528733245"
    try:

        # Download image into test.jpg
        file = open("test.jpg", "wb")
        response = requests.get(link)
        file.write(response.content)
        file.close()

        # Create a thumbnail of the image with specified method.
        # Resize does not work here.
        im = Image.open("test.jpg")
        im.thumbnail(size, Image.BILINEAR)
        im.save("10_resized.jpg", "JPEG")

        # Remove the downloaded image in order to conserve storage.
        os.remove("test.jpg")

    except IOError as e:
        print(str(e) + " : cannot create thumbnail")


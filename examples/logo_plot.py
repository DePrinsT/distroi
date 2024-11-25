"""
For plotting the logo disk of DISTROI.
"""

import distroi
import matplotlib.pyplot as plt


def logo_plot(image, img_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    img_plot = ax.imshow(
        image.img,
        cmap="gist_heat",
        aspect="auto",
        extent=(
            (image.num_pix_x / 2) * image.pixelscale_x * distroi.auxiliary.constants.RAD2MAS,
            (-image.num_pix_x / 2) * image.pixelscale_x * distroi.auxiliary.constants.RAD2MAS,
            (-image.num_pix_y / 2) * image.pixelscale_y * distroi.auxiliary.constants.RAD2MAS,
            (image.num_pix_y / 2) * image.pixelscale_y * distroi.auxiliary.constants.RAD2MAS,
        ),
        interpolation="bicubic",
    )
    ax.set_aspect(2.5)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(img_path, dpi=700, bbox_inches="tight")
    plt.show()


mod_dir = "./models/IRAS08544-4431_test_model/"
fig_dir = "./figures/single_disk_model/PIONIER"

# FFT test + output info on frequencies
img_dir = "PIONIER/data_1.65/"
img = distroi.read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz", disk_only=True)
logo_plot(img, "/home/toond/Downloads/distroi_logo.png")

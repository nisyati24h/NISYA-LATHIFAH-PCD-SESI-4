import os
import numpy as np
import imageio.v3 as iio                 
from scipy import ndimage as ndi        
import matplotlib.pyplot as plt         

folder = os.path.dirname(os.path.abspath(__file__))
file_gambar = os.path.join(folder, "Treefingers.jpeg")
if not os.path.exists(file_gambar):
    raise SystemExit("Treefingers.jpeg tidak ditemukan di folder: " + folder)


img = iio.imread(file_gambar)
if img.ndim == 3:
    img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
gray = img.astype(np.uint8)


hist = np.bincount(gray.ravel(), minlength=256)   
cdf = hist.cumsum()                               
cdf_min = cdf[cdf > 0].min()
pembagi = max(cdf[-1] - cdf_min, 1)
lut = np.round((cdf - cdf_min) / pembagi * 255)   
equalized = lut.astype(np.uint8)[gray]           


f = gray.astype(float)
blur = ndi.gaussian_filter(f, sigma=2)           
tajam = np.clip(f + 1.5 * (f - blur), 0, 255).astype(np.uint8)   


iio.imwrite(os.path.join(folder, "hasil_ekualisasi.png"), equalized)
iio.imwrite(os.path.join(folder, "hasil_pertajam.png"), tajam)

judul = ["Gambar Asli", "Ekualisasi Histogram", "Pertajam (Unsharp Masking)"]
judul_hist = ["Histogram Asli", "Histogram Ekualisasi", "Histogram Pertajam"]
citra = [gray, equalized, tajam]

plt.figure(figsize=(14, 8))
for i in range(3):
    plt.subplot(2, 3, i + 1)                     
    plt.imshow(citra[i], cmap="gray", vmin=0, vmax=255)
    plt.title(judul[i])
    plt.axis("off")

    plt.subplot(2, 3, i + 4)                      
    plt.hist(citra[i].ravel(), bins=256, range=(0, 256))
    plt.yscale("log")                             
    plt.xlim(0, 255)
    plt.title(judul_hist[i])
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah piksel")
plt.tight_layout()
plt.savefig(os.path.join(folder, "hasil.png"), dpi=150)
plt.show()
import qrcode

vcard = """BEGIN:VCARD
VERSION:3.0
FN:Gilbert Rahme
TEL:+96170931335
EMAIL:gilbert.rahme@gmail.com
END:VCARD"""

qr = qrcode.QRCode(
    version=None,                                  # None = auto-size based on data
    error_correction=qrcode.constants.ERROR_CORRECT_M,  # allows ~15% damage/obstruction and still scan
    box_size=10,                                   # pixel size of each QR "box"
    border=4,                                       # white border thickness (boxes)
)
qr.add_data(vcard)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")
img.save("gilbert_rahme_contact_qr.png")
print("QR code saved!")
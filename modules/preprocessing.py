import numpy as np
import cv2
from PIL import Image
METHOD_DESCRIPTIONS={
"Filtering":"Mengurangi noise menggunakan filter median atau Gaussian.",
"Segmentation & Masking":"Membuat mask lesi dengan threshold adaptif dan operasi morfologi. Area yang terdeteksi bersifat estimasi otomatis.",
"Enhancement":"Meningkatkan kontras lokal menggunakan CLAHE.",
"Hair Removal":"Mengurangi artefak rambut dengan deteksi garis gelap dan inpainting.",
"Grayscale":"Mengubah citra RGB menjadi grayscale untuk melihat intensitas luminance."
}
def preprocess_image(image,method,filter_type="Median",kernel=5,clip_limit=2.0):
    rgb=np.array(image.convert("RGB")); bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    if method=="Filtering":
        if filter_type=="Gaussian": out=cv2.GaussianBlur(bgr,(kernel,kernel),0)
        else: out=cv2.medianBlur(bgr,kernel)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    if method=="Enhancement":
        lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab)
        l=cv2.createCLAHE(clipLimit=float(clip_limit),tileGridSize=(8,8)).apply(l)
        out=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    if method=="Hair Removal":
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
        blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        _,hair=cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
        hair=cv2.dilate(hair,np.ones((2,2),np.uint8),iterations=1)
        out=cv2.inpaint(bgr,hair,3,cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    if method=="Grayscale":
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray)
    if method=="Segmentation & Masking":
        hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
        # Conservative lesion segmentation demo; not a clinical segmentation algorithm.
        s=hsv[:,:,1]; v=hsv[:,:,2]
        candidate=cv2.adaptiveThreshold(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,5)
        candidate=cv2.bitwise_and(candidate,cv2.threshold(s,35,255,cv2.THRESH_BINARY)[1])
        candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=2)
        candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=1)
        contours,_=cv2.findContours(candidate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        mask=np.zeros(gray.shape if (gray:=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)) is not None else bgr.shape[:2],dtype=np.uint8)
        if contours:
            contour=max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour)>0.005*mask.size: cv2.drawContours(mask,[contour],-1,255,thickness=cv2.FILLED)
        masked=cv2.bitwise_and(bgr,bgr,mask=mask)
        return Image.fromarray(cv2.cvtColor(masked,cv2.COLOR_BGR2RGB)),Image.fromarray(mask)
    return image

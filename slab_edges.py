"""Slab outline for gel images: hysteresis threshold -> cleaned mask -> contour.
No shape assumption (handles notches, curved/wavy edges). Flags contour parts that run along the image frame."""
import sys, numpy as np, tifffile, cv2, warnings
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from skimage import measure, morphology, filters

def levels(c):
    h,e=np.histogram(c,bins=np.arange(0,c.max()+1,0.5)); ctr=(e[:-1]+e[1:])/2
    lo=ctr[np.argmax(h[:6])]                                  # surround / camera floor (lowest mode)
    sel=ctr>lo+2; hi=ctr[sel][np.argmax(h[sel])]              # slab glow mode
    return lo,hi

def slab_mask(a, sigma=2, k=3, min_margin=0.5, open_r=9, close_r=9):
    a=a.astype(np.float32); s=cv2.GaussianBlur(a,(0,0),sigma)
    c=np.minimum(s,np.percentile(s,98))                       # cap bands
    lo,hi=levels(c)
    sd=1.4826*np.median(np.abs(c[c<lo+1.5]-np.median(c[c<lo+1.5])))      # surround noise
    high=(lo+hi)/2                                            # confident gel
    low=lo+max(k*sd,min_margin)                               # just above surround: dim edges/corners
    m=filters.apply_hysteresis_threshold(c,low,high)
    m=morphology.binary_closing(m,morphology.disk(close_r))
    m=morphology.binary_opening(m,morphology.disk(open_r))    # cut thin debris bridges
    lab=measure.label(m); big=max(measure.regionprops(lab),key=lambda r:r.area)
    m=ndi.binary_fill_holes(lab==big.label)
    return m,dict(floor=lo,slab=hi,low=low,high=high)

def border_flags(cnt,shape,tol=2):
    H,W=shape; p=cnt.reshape(-1,2)
    return (p[:,0]<=tol)|(p[:,1]<=tol)|(p[:,0]>=W-1-tol)|(p[:,1]>=H-1-tol)

if __name__=='__main__':
    a=tifffile.imread(sys.argv[1]); m,info=slab_mask(a); print({k:round(float(v),2) for k,v in info.items()},'area',int(m.sum()))
    cnt=max(cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)
    bf=border_flags(cnt,a.shape); print('contour pts %d, of which on image frame %d (%.0f%%)'%(len(cnt),bf.sum(),100*bf.mean()))
    poly=cv2.approxPolyDP(cnt,2.0,True); print('polygon @2px tolerance: %d vertices'%len(poly))
    np.save(sys.argv[2]+'.npy',m)
    v=np.clip(a.astype(float)/np.percentile(a,99.5),0,1)**0.5
    rgb=cv2.cvtColor((v*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    p=cnt.reshape(-1,2)
    for i in range(len(p)): cv2.circle(rgb,tuple(int(x) for x in p[i]),1,(255,0,255) if bf[i] else (0,255,0),-1)
    cv2.imwrite(sys.argv[2],rgb)

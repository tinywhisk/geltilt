"""Slab outline via marker watershed on the log-intensity gradient.
Inside seed = confident gel; outside seed = far from gel. Boundary lands on the strongest step between them, so a smooth
stray-light halo outside the slab is not mistaken for gel. Also scores edge confidence (inside/outside level ratio)."""
import os, sys, numpy as np, tifffile, cv2, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from slab_edges_v3 import levels, slab_mask

def slab_outline(a, sigma=2, reach=55, min_ratio=1.4):
    a=a.astype(np.float32); H,W=a.shape
    s=cv2.GaussianBlur(a,(0,0),sigma); c=np.minimum(s,np.percentile(s,98))
    lo,hi=levels(c); mid=(lo+hi)/2
    gen,_=slab_mask(a,sigma=sigma)                       # generous mask: covers all gel (plus halo) incl. dim regions
    edt=ndi.distance_transform_edt(np.pad(gen,1))[1:-1,1:-1]             # frame counts as outside so edges near it can be found
    reach=min(reach,0.5*edt.max())                                       # can't be wider than the slab allows
    inner=edt>reach                                                      # certainly gel
    outer=~gen                                           # certainly background
    L=np.log(c+1.0); gx=cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3); g=np.hypot(gx,gy)
    markers=np.zeros((H,W),np.int32); markers[outer]=1; markers[inner]=2
    ws=segmentation.watershed(g,markers)
    m=ndi.binary_fill_holes(ws==2)
    m=morphology.binary_opening(m,morphology.disk(5))
    lab=measure.label(m); big=max(measure.regionprops(lab),key=lambda r:r.area); m=lab==big.label
    # confidence along the contour
    cnt=max(cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea).reshape(-1,2)
    d=cv2.GaussianBlur((ndi.distance_transform_edt(m)-ndi.distance_transform_edt(~m)).astype(np.float32),(0,0),4)
    gy_,gx_=np.gradient(d); n=np.stack([gx_[cnt[:,1],cnt[:,0]],gy_[cnt[:,1],cnt[:,0]]],1); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    def lvl(t0,t1):
        ts=np.arange(t0,t1+1); P=cnt[:,None,:]+ts[None,:,None]*n[:,None,:]
        return np.median(ndi.map_coordinates(c,[P[...,1],P[...,0]],order=1,mode='nearest'),axis=1)
    ratio=lvl(6,18)/np.maximum(lvl(-18,-6),1e-3)
    ratio=ndi.median_filter(ratio,size=31,mode='wrap')
    on_frame=(cnt[:,0]<=2)|(cnt[:,1]<=2)|(cnt[:,0]>=W-3)|(cnt[:,1]>=H-3)
    cnt0,ratio0=cnt.copy(),ratio.copy()

    # Where no step exists near the frame, the gel most likely runs off the frame: snap to it and keep the flag.
    from scipy.spatial import cKDTree
    weak=(ratio<min_ratio)&~on_frame
    yy,xx=np.mgrid[:H,:W]; dfr=np.minimum.reduce([xx,yy,W-1-xx,H-1-yy])
    near=gen&(dfr<=reach+10)&~m
    if weak.any() and near.any():
        tree=cKDTree(cnt); py,px=np.nonzero(near); dist,idx=tree.query(np.stack([px,py],1))
        take=weak[idx]&(dist<reach+10)
        m=m.copy(); m[py[take],px[take]]=True
        m=morphology.binary_closing(m,morphology.disk(7))
        lab=measure.label(m); big=max(measure.regionprops(lab),key=lambda r:r.area); m=ndi.binary_fill_holes(lab==big.label)
        cnt=max(cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea).reshape(-1,2)
        tree=cKDTree(cnt0); _,j=tree.query(cnt); ratio=ratio0[j]
        on_frame=(cnt[:,0]<=2)|(cnt[:,1]<=2)|(cnt[:,0]>=W-3)|(cnt[:,1]>=H-3)
    return m,cnt,ratio,on_frame,dict(floor=lo,slab=hi,mid=mid)

if __name__=='__main__':
    a=tifffile.imread(sys.argv[1]); out=sys.argv[2]
    m,cnt,ratio,fr,info=slab_outline(a)
    weak=(ratio<1.4)&~fr
    print({k:round(float(v),2) for k,v in info.items()},'area',int(m.sum()))
    print('contour %d pts: %.0f%% on image frame, %.0f%% sharp step (ratio>=1.4), %.0f%% weak/no step'%(len(cnt),100*fr.mean(),100*((ratio>=1.4)&~fr).mean(),100*weak.mean()))
    v=np.clip(a.astype(float)/np.percentile(a,99.5),0,1)**0.5
    rgb=cv2.cvtColor((v*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for p,f,w in zip(cnt,fr,weak): cv2.circle(rgb,(int(p[0]),int(p[1])),1,(255,0,255) if f else ((0,200,255) if w else (0,255,0)),-1)
    cv2.imwrite(out,rgb); np.save(out.replace('.png','_mask.npy'),m)

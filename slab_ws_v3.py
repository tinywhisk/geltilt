"""Slab outline via marker watershed on the log-intensity gradient.
Inside seed = confident gel; outside seed = far from gel. Boundary lands on the strongest step between them, so a smooth
stray-light halo outside the slab is not mistaken for gel. Also scores edge confidence (inside/outside level ratio)."""
import os, sys, numpy as np, tifffile, cv2, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from slab_edges_v3 import levels, slab_mask


def halo_depth(c, gen, default, max_depth=320, min_ratio=1.8, margin=8, step=3):
    """Per generous-mask boundary point: depth (px, inward) of the first strong intensity step = the real slab edge.
    Returns a KD-tree over boundary points and a depth per point (default where no step is found)."""
    from scipy.spatial import cKDTree
    H,W=c.shape
    cnt=max(cv2.findContours(gen.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea).reshape(-1,2)
    d=cv2.GaussianBlur((ndi.distance_transform_edt(gen)-ndi.distance_transform_edt(~gen)).astype(np.float32),(0,0),4)
    gy,gx=np.gradient(d); n=np.stack([gx[cnt[:,1],cnt[:,0]],gy[cnt[:,1],cnt[:,0]]],1); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
    sub=np.arange(0,len(cnt),step); T=max_depth+14; ts=np.arange(-12,T-12)
    P=cnt[sub][:,None,:]+ts[None,:,None]*n[sub][:,None,:]
    prof=ndi.map_coordinates(c,[np.clip(P[...,1],0,H-1),np.clip(P[...,0],0,W-1)],order=1,mode='nearest')
    cs=np.concatenate([np.zeros((len(sub),1)),np.cumsum(prof,1)],1)
    def mean(a,b): return (cs[:,b]-cs[:,a])/(b-a)
    D=np.full(len(sub),np.nan)
    raw_found=np.zeros(len(sub),bool)
    for j in range(len(sub)):
        for t in range(12,T-14):          # index t <-> depth ts[t]
            out=mean(t-10,t-3)[j] if False else (cs[j,t-3]-cs[j,t-10])/7
            inn=(cs[j,t+10]-cs[j,t+3])/7
            if inn/max(out,1e-3)>=min_ratio:
                # take the local maximum of the ratio within the next few px
                best=t; bestr=inn/max(out,1e-3)
                for u in range(t+1,min(t+8,T-14)):
                    o2=(cs[j,u-3]-cs[j,u-10])/7; i2=(cs[j,u+10]-cs[j,u+3])/7; r2=i2/max(o2,1e-3)
                    if r2>bestr: best,bestr=u,r2
                D[j]=ts[best]; raw_found[j]=True; break
    # reject outliers / fill gaps along the contour with a nan-aware running median
    k=41; Dp=np.concatenate([D[-k:],D,D[:k]]); out=np.full(len(D),np.nan)
    for j in range(len(D)):
        w=Dp[j+k-k//2:j+k+k//2+1]; w=w[~np.isnan(w)]
        if len(w)>=k*0.4: out[j]=np.median(w)
    found=~np.isnan(out)
    out=np.where(np.isnan(out),default,out)
    return cKDTree(cnt[sub]),np.maximum(out,0),found,cnt[sub].astype(float),n[sub]

def slab_outline(a, sigma=2, reach=55, min_ratio=1.4, open_sides=()):
    a=a.astype(np.float32); H,W=a.shape
    s=cv2.GaussianBlur(a,(0,0),sigma); c=np.minimum(s,np.percentile(s,98))
    lo,hi=levels(c); mid=(lo+hi)/2
    gen,_=slab_mask(a,sigma=sigma)                       # generous mask: covers all gel (plus halo) incl. dim regions
    edt=ndi.distance_transform_edt(np.pad(gen,1))[1:-1,1:-1]             # frame counts as outside so edges near it can be found
    reach=min(reach,0.5*edt.max())
    tree,Dep,found,bp,bn=halo_depth(c,gen,default=reach)
    py,px=np.nonzero(edt>5); dist,ix=tree.query(np.stack([px,py],1))
    inner=np.zeros(gen.shape,bool); inner[py,px]=edt[py,px]>(Dep[ix]+8)    # beyond the measured halo depth
    # outside seed also covers the measured halo, so only a thin ring around the real edge is left for the watershed
    vec=np.stack([px,py],1)-bp[ix]; cosang=(vec*bn[ix]).sum(1)/(np.linalg.norm(vec,axis=1)+1e-9)
    halo=found[ix]&(edt[py,px]<Dep[ix]-8)&(cosang>0.8)
    outer=~gen.copy(); outer[py[halo],px[halo]]=True
    L=np.log(c+1.0); gx=cv2.Sobel(L,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(L,cv2.CV_32F,0,1,ksize=3); g=np.hypot(gx,gy)
    markers=np.zeros((H,W),np.int32); markers[outer]=1; markers[inner]=2
    ws=segmentation.watershed(g,markers)
    m=ndi.binary_fill_holes(ws==2)
    m=morphology.binary_opening(m,morphology.disk(5))
    lab=measure.label(m); big=max(measure.regionprops(lab),key=lambda r:r.area); m=lab==big.label

    # Sides declared "gel runs off the frame": extend the mask to that frame edge (only in columns/rows that contain gel)
    for side in open_sides:
        # eligible columns/rows: those whose first gel pixel lies near the slab's own end on that side
        # (a slanted side edge must not be extended along its whole length)
        slack=150
        if side in ('top','bottom'):
            cols=np.nonzero(m.any(0))[0]
            first=np.array([np.nonzero(m[:,x])[0].min() if side=='top' else np.nonzero(m[:,x])[0].max() for x in cols])
            ref=np.median(first); ok=np.abs(first-ref)<=slack
            for x,f,k in zip(cols,first,ok):
                if k:
                    if side=='top': m[:f,x]=True
                    else: m[f:,x]=True
        elif side in ('left','right'):
            rows=np.nonzero(m.any(1))[0]
            first=np.array([np.nonzero(m[y])[0].min() if side=='left' else np.nonzero(m[y])[0].max() for y in rows])
            ref=np.median(first); ok=np.abs(first-ref)<=slack
            for y,f,k in zip(rows,first,ok):
                if k:
                    if side=='left': m[y,:f]=True
                    else: m[y,f:]=True
    m=ndi.binary_fill_holes(m)
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
    open_sides=tuple(sys.argv[3:])        # e.g.  top   (sides where the gel runs off the frame)
    m,cnt,ratio,fr,info=slab_outline(a,open_sides=open_sides)
    weak=(ratio<1.4)&~fr
    print({k:round(float(v),2) for k,v in info.items()},'area',int(m.sum()))
    print('contour %d pts: %.0f%% on image frame, %.0f%% sharp step (ratio>=1.4), %.0f%% weak/no step'%(len(cnt),100*fr.mean(),100*((ratio>=1.4)&~fr).mean(),100*weak.mean()))
    v=np.clip(a.astype(float)/np.percentile(a,99.5),0,1)**0.5
    rgb=cv2.cvtColor((v*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    for p,f,w in zip(cnt,fr,weak): cv2.circle(rgb,(int(p[0]),int(p[1])),1,(255,0,255) if f else ((0,200,255) if w else (0,255,0)),-1)
    cv2.imwrite(out,rgb); np.save(out.replace('.png','_mask.npy'),m)

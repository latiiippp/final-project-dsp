import cv2
import numpy as np

def draw_grid(image, graph_width, graph_height, color=(200, 200, 200), thickness=1):
    for i in range(1, 5): 
        y = int(i * graph_height / 5)
        cv2.line(image, (0, y), (graph_width, y), color, thickness)
    for i in range(1, 10): 
        x = int(i * graph_width / 10)
        cv2.line(image, (x, 0), (x, graph_height), color, thickness)

def cpu_POS(rgb_signal_segment, **kargs):
    eps = 10**-9
    X = rgb_signal_segment 
    if X.ndim == 2 and X.shape[0] == 3: X = np.expand_dims(X, axis=0)
    elif X.ndim != 3 or X.shape[0] != 1 or X.shape[1] != 3:
        raise ValueError(f"Input signal for cpu_POS must be shape (1,3,N) or (3,N), got {X.shape}")
    e, c, f = X.shape
    fps = kargs.get('fps', 30.0) 
    if fps <= 0: fps = 30.0
    w = int(1.6 * fps)   
    if w == 0: w = 1 
    if f < w : return np.zeros((e, f))
    P = np.array([[0,1,-1],[-2,1,1]]); Q = np.stack([P for _ in range(e)],axis=0)
    H = np.zeros((e,f)) 
    for n in np.arange(w-1,f): 
        m=n-w+1; Cn=X[:,:,m:(n+1)]; M=1.0/(np.mean(Cn,axis=2)+eps)
        Cn_norm=np.multiply(np.expand_dims(M,axis=2),Cn)
        S=np.diagonal(np.tensordot(Q,Cn_norm,axes=([2],[1])),axis1=0,axis2=2).T
        S1=S[:,0,:]; S2=S[:,1,:]; alpha=np.std(S1,axis=1)/(eps+np.std(S2,axis=1))
        Hn_w=S1+np.expand_dims(alpha,axis=1)*S2
        Hnm_w=Hn_w-np.expand_dims(np.mean(Hn_w,axis=1),axis=1)
        if Hnm_w.shape[1]>0: H[:,n]=Hnm_w[:,-1] 
    return H
# NumPy version of YOLO11 so the model runs without PyTorch.
# Only does inference (no training). Checked against 14 prediction files the real Ultralytics runs saved:
# same counts, boxes off by < 0.0001 px. Copied from experiments/Barnacle_Improvement_Study/npyolo.
"""PyTorch-free YOLO11 inference (NumPy) for checkpoints saved by Ultralytics.

Why this exists: this workspace cannot install PyTorch/Ultralytics. The forward
pass follows Ultralytics 8.4 modules (Conv+BN+SiLU, C3k2/C3k/Bottleneck, SPPF,
C2PSA attention, Detect with DFL) and its predict pre/post-processing
(letterbox, sigmoid scores, greedy NMS, box rescaling). Correctness is checked
by reproducing prediction CSVs that the original Ultralytics runs saved
(see validate_np.py). Training is NOT possible with this module.
"""
import numpy as np
import cv2
from .ptload import load, modules, params, cls

F32 = np.float32


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60))))


def _conv_raw(conv, bn=None):
    p = params(conv)
    W = p['weight'].astype(F32)
    b = p['bias'].astype(F32) if 'bias' in p else np.zeros(W.shape[0], F32)
    if bn is not None:
        q = params(bn)
        s = q['weight'].astype(F32) / np.sqrt(q['running_var'].astype(F32) + F32(bn.eps))
        W = W * s[:, None, None, None]
        b = (b - q['running_mean'].astype(F32)) * s + q['bias'].astype(F32)
    d = conv.__dict__
    return dict(W=np.ascontiguousarray(W), b=b, stride=int(d['stride'][0]),
                pad=int(d['padding'][0]), groups=int(d['groups']))


def conv2d(x, c):
    """x: (N, C, H, W) float32. Cross-correlation like torch.nn.Conv2d."""
    W, b, s, p, g = c['W'], c['b'], c['stride'], c['pad'], c['groups']
    O, Cg, kh, kw = W.shape
    N, C, H, Wd = x.shape
    if p:
        x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho = (H + 2 * p - kh) // s + 1
    Wo = (Wd + 2 * p - kw) // s + 1
    if g == 1:
        if kh == 1 and kw == 1 and s == 1:
            out = np.matmul(W.reshape(O, C), x.reshape(N, C, -1)).reshape(N, O, Ho, Wo)
        else:
            win = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
            win = win[:, :, ::s, ::s][:, :, :Ho, :Wo]            # N C Ho Wo kh kw
            col = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * kh * kw)
            out = (col @ W.reshape(O, -1).T).reshape(N, Ho, Wo, O).transpose(0, 3, 1, 2)
    elif g == C and O == C and Cg == 1:
        out = np.zeros((N, O, Ho, Wo), F32)
        for i in range(kh):
            for j in range(kw):
                out += W[:, 0, i, j][None, :, None, None] * \
                    x[:, :, i:i + s * (Ho - 1) + 1:s, j:j + s * (Wo - 1) + 1:s]
    else:
        raise NotImplementedError('grouped conv')
    out = out + b[None, :, None, None]
    return np.ascontiguousarray(out, dtype=F32)


class Model:
    def __init__(self, path):
        ck = load(path)
        m = ck.get('ema') if ck.get('ema') is not None else ck['model']
        self.meta = dict(epoch=ck.get('epoch'), version=ck.get('version'),
                         train_args=ck.get('train_args'))
        self.layers = []
        for _, layer in modules(modules(m)['model']).items():
            self.layers.append((layer.__dict__['f'], cls(layer), self._build(layer)))
        self.save = set(m.__dict__['save'])
        det = self.layers[-1][2]
        self.stride = det['stride']
        self.reg_max = det['reg_max']

    # ---- module builders -------------------------------------------------
    def _conv(self, s):
        mods = modules(s)
        c = _conv_raw(mods['conv'], mods.get('bn'))
        c['act'] = cls(mods['act']) == 'SiLU' if 'act' in mods else False
        return c

    def _build(self, s):
        k = cls(s)
        mods = modules(s)
        if k in ('Conv', 'DWConv'):
            return self._conv(s)
        if k == 'Bottleneck':
            return dict(cv1=self._conv(mods['cv1']), cv2=self._conv(mods['cv2']), add=s.add)
        if k == 'C3k':
            return dict(cv1=self._conv(mods['cv1']), cv2=self._conv(mods['cv2']),
                        cv3=self._conv(mods['cv3']),
                        m=[('Bottleneck', self._build(b)) for b in modules(mods['m']).values()])
        if k in ('C3k2', 'C2f'):
            return dict(cv1=self._conv(mods['cv1']), cv2=self._conv(mods['cv2']),
                        m=[(cls(b), self._build(b)) for b in modules(mods['m']).values()])
        if k == 'SPPF':
            mp = mods['m'].__dict__
            return dict(cv1=self._conv(mods['cv1']), cv2=self._conv(mods['cv2']),
                        k=int(mp['kernel_size']), n=int(s.__dict__.get('n', 3)),
                        add=bool(s.__dict__.get('add', False)))
        if k == 'C2PSA':
            blocks = []
            for b in modules(mods['m']).values():
                bm = modules(b)
                at = bm['attn']
                am = modules(at)
                blocks.append(dict(add=b.add, nh=at.num_heads, kd=at.key_dim, hd=at.head_dim,
                                   scale=F32(at.scale), qkv=self._conv(am['qkv']),
                                   proj=self._conv(am['proj']), pe=self._conv(am['pe']),
                                   ffn=[self._conv(c) for c in modules(bm['ffn']).values()]))
            return dict(cv1=self._conv(mods['cv1']), cv2=self._conv(mods['cv2']), c=s.c, m=blocks)
        if k == 'Upsample':
            return dict(scale=int(s.scale_factor))
        if k == 'Concat':
            return dict(d=s.d)
        if k == 'Detect':
            def seq(q):
                out = []
                for c in modules(q).values():
                    ck_ = cls(c)
                    if ck_ == 'Sequential':
                        out.extend(seq(c))
                    elif ck_ in ('Conv', 'DWConv'):
                        out.append(self._conv(c))
                    elif ck_ == 'Conv2d':
                        cc = _conv_raw(c)
                        cc['act'] = False
                        out.append(cc)
                    else:
                        raise NotImplementedError(ck_)
                return out
            if s.__dict__.get('end2end'):
                raise NotImplementedError('end2end head')
            return dict(cv2=[seq(q) for q in modules(mods['cv2']).values()],
                        cv3=[seq(q) for q in modules(mods['cv3']).values()],
                        reg_max=int(s.reg_max), nc=int(s.nc),
                        stride=np.asarray(s.stride, F32))
        raise NotImplementedError(k)

    # ---- forward -----------------------------------------------------------
    @staticmethod
    def _c(x, c):
        y = conv2d(x, c)
        return _silu(y) if c['act'] else y

    def _run(self, k, p, x):
        C = self._c
        if k in ('Conv', 'DWConv'):
            return C(x, p)
        if k == 'Bottleneck':
            y = C(C(x, p['cv1']), p['cv2'])
            return x + y if p['add'] else y
        if k == 'C3k':
            a = C(x, p['cv1'])
            for kk, pp in p['m']:
                a = self._run(kk, pp, a)
            return C(np.concatenate([a, C(x, p['cv2'])], 1), p['cv3'])
        if k in ('C3k2', 'C2f'):
            y = C(x, p['cv1'])
            h = y.shape[1] // 2
            ys = [y[:, :h], y[:, h:]]
            for kk, pp in p['m']:
                ys.append(self._run(kk, pp, ys[-1]))
            return C(np.concatenate(ys, 1), p['cv2'])
        if k == 'SPPF':
            y = [C(x, p['cv1'])]
            r = p['k'] // 2
            for _ in range(p['n']):
                z = np.pad(y[-1], ((0, 0), (0, 0), (r, r), (r, r)), constant_values=-np.inf)
                H, Wd = y[-1].shape[2:]
                m1 = z[:, :, :, 0:Wd]
                for j in range(1, p['k']):
                    m1 = np.maximum(m1, z[:, :, :, j:j + Wd])
                m2 = m1[:, :, 0:H]
                for i in range(1, p['k']):
                    m2 = np.maximum(m2, m1[:, :, i:i + H])
                y.append(np.ascontiguousarray(m2))
            out = C(np.concatenate(y, 1), p['cv2'])
            return out + x if p['add'] else out
        if k == 'C2PSA':
            y = C(x, p['cv1'])
            a, b = y[:, :p['c']], y[:, p['c']:]
            for blk in p['m']:
                b = b + self._attn(b, blk) if blk['add'] else self._attn(b, blk)
                f = b
                for c in blk['ffn']:
                    f = C(f, c)
                b = b + f if blk['add'] else f
            return C(np.concatenate([a, b], 1), p['cv2'])
        if k == 'Upsample':
            s = p['scale']
            return x.repeat(s, axis=2).repeat(s, axis=3)
        raise NotImplementedError(k)

    def _attn(self, x, p):
        N, Cc, H, W = x.shape
        n = H * W
        qkv = self._c(x, p['qkv']).reshape(N, p['nh'], p['kd'] * 2 + p['hd'], n)
        q, k, v = qkv[:, :, :p['kd']], qkv[:, :, p['kd']:2 * p['kd']], qkv[:, :, 2 * p['kd']:]
        a = np.matmul(q.transpose(0, 1, 3, 2), k) * p['scale']        # N h n n
        a = a - a.max(-1, keepdims=True)
        a = np.exp(a)
        a /= a.sum(-1, keepdims=True)
        out = np.matmul(v, a.transpose(0, 1, 3, 2)).reshape(N, Cc, H, W)
        out = out + self._c(np.ascontiguousarray(v.reshape(N, Cc, H, W)), p['pe'])
        return self._c(out, p['proj'])

    def raw(self, x, return_features=False):
        """x: (N,3,H,W) RGB float32 in [0,1]. Returns (N, 4+nc, A) like Ultralytics."""
        y = []
        for idx, (f, k, p) in enumerate(self.layers):
            if f != -1:
                inp = y[f] if isinstance(f, int) else [x if j == -1 else y[j] for j in f]
            else:
                inp = x
            if k == 'Concat':
                x = np.concatenate(inp, p['d'])
            elif k == 'Detect':
                feats = inp
                x = self._detect(inp, p)
            else:
                x = self._run(k, p, inp)
            y.append(x if idx in self.save else None)
        return (x, feats) if return_features else x

    def _detect(self, xs, p):
        outs, anchors, strides = [], [], []
        self._pen = []
        for i, xi in enumerate(xs):
            a = xi
            for c in p['cv2'][i]:
                a = self._c(a, c)
            b = xi
            for c in p['cv3'][i][:-1]:
                b = self._c(b, c)
            self._pen.append(b.reshape(b.shape[0], b.shape[1], -1))
            b = self._c(b, p['cv3'][i][-1])
            N, _, H, W = a.shape
            outs.append(np.concatenate([a, b], 1).reshape(N, -1, H * W))
            sy, sx = np.meshgrid(np.arange(H, dtype=F32) + 0.5, np.arange(W, dtype=F32) + 0.5,
                                 indexing='ij')
            anchors.append(np.stack([sx.ravel(), sy.ravel()], 0))
            strides.append(np.full((1, H * W), p['stride'][i], F32))
        z = np.concatenate(outs, 2)                       # N, 4*reg+nc, A
        self._pen = np.concatenate(self._pen, 2)          # N, 80, A penultimate class features
        self._logit = z[:, -p['nc']:]
        anc = np.concatenate(anchors, 1)
        st = np.concatenate(strides, 1)
        R = p['reg_max']
        box, logit = z[:, :4 * R], z[:, 4 * R:]
        N, _, A = box.shape
        bb = box.reshape(N, 4, R, A)
        bb = bb - bb.max(2, keepdims=True)
        bb = np.exp(bb)
        bb /= bb.sum(2, keepdims=True)
        dist = (bb * np.arange(R, dtype=F32)[None, None, :, None]).sum(2)   # N,4,A
        lt, rb = dist[:, :2], dist[:, 2:]
        x1y1 = anc[None] - lt
        x2y2 = anc[None] + rb
        cxy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        dbox = np.concatenate([cxy, wh], 1) * st[None]
        return np.concatenate([dbox, 1.0 / (1.0 + np.exp(-logit))], 1)

    # ---- Ultralytics-like predict -----------------------------------------
    def predict(self, imgs_bgr, imgsz=512, conf=0.001, iou=0.5, max_det=2000, batch=4, feats=False):
        shapes = {im.shape for im in imgs_bgr}
        auto = len(shapes) == 1
        results = []
        for s0 in range(0, len(imgs_bgr), batch):
            chunk = imgs_bgr[s0:s0 + batch]
            lb = [_letterbox(im, imgsz, auto) for im in chunk]
            x = np.stack([l[0] for l in lb]).astype(F32)
            x = x[..., ::-1].transpose(0, 3, 1, 2) / F32(255.0)   # BGR->RGB, NCHW
            pred = self.raw(np.ascontiguousarray(x, dtype=F32))
            for j, im in enumerate(chunk):
                boxes, scores, idx = _nms(pred[j], conf, iou, max_det, return_index=True)
                gain, (padw, padh) = lb[j][1], lb[j][2]
                boxes[:, [0, 2]] -= padw
                boxes[:, [1, 3]] -= padh
                boxes /= gain
                h, w = im.shape[:2]
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
                if feats:
                    results.append((boxes, scores, self._pen[j][:, idx].T.copy(), self._logit[j][0, idx].copy()))
                else:
                    results.append((boxes, scores))
        return results


def _letterbox(im, new, auto, stride=32):
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = new - nw, new - nh
    if auto:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2
    if (w, h) != (nw, nh):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if top or bottom or left or right:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))
    padw = (new - nw) / 2 if not auto else ((new - nw) % stride) / 2
    padh = (new - nh) / 2 if not auto else ((new - nh) % stride) / 2
    return im, r, (round(padw - 0.1), round(padh - 0.1))


def _nms(p, conf, iou, max_det, max_nms=30000, return_index=False):
    """p: (4+nc, A) with xywh + class scores (single class here)."""
    sc = p[4:].max(0)
    keep = sc > conf
    b = p[:4, keep].T
    s = sc[keep]
    anchor_ids = np.flatnonzero(keep)
    if not len(s):
        e = (np.zeros((0, 4), F32), np.zeros(0, F32))
        return e + (np.zeros(0, int),) if return_index else e
    order = np.argsort(-s, kind='stable')[:max_nms]
    anchor_ids = anchor_ids[order]
    b, s = b[order], s[order]
    xyxy = np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    alive = np.ones(len(s), bool)
    out = []
    for i in range(len(s)):
        if not alive[i]:
            continue
        out.append(i)
        if len(out) >= max_det:
            break
        rest = np.flatnonzero(alive[i + 1:]) + i + 1
        if not len(rest):
            continue
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        ov = inter / (area[i] + area[rest] - inter + 1e-12)
        alive[rest[ov > iou]] = False
    out = np.asarray(out, int)
    if return_index:
        return xyxy[out].astype(F32), s[out].astype(F32), anchor_ids[out]
    return xyxy[out].astype(F32), s[out].astype(F32)
